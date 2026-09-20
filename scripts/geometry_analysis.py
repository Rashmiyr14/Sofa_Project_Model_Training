"""
sofa_geometry_analysis.py

Sofa component detection + geometry analysis + seater classification.

YOLO classes:
    0 -> back_cushion
    1 -> base
    2 -> left_arm
    3 -> legs
    4 -> right_arm
    5 -> seat_cushion

IMPORTANT:
    YOLO component count is NOT the same as sofa seat count.

Example:
    One continuous seat cushion can represent a 2-seater or 3-seater.

Therefore the final seater type is estimated from geometry:
    - overall sofa width
    - physical arm positions
    - usable seating span
    - seat cushion width
    - seat/sofa width ratio
    - seat/usable-span ratio
    - back cushion count as supporting evidence

OUTPUTS:
    runs/
        geometry/
            geometry_result.json
            geometry_result.txt
            geometry_result.jpg
            sofa_background_removed.png

Usage:
    python .\scripts\sofa_geometry_analysis.py

The program will ask:
    Enter path to sofa image:

Optional:
    python .\scripts\sofa_geometry_analysis.py --no-bg-removal
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


# ============================================================================
# OPTIONAL BACKGROUND REMOVAL
# ============================================================================

try:
    from rembg import remove as rembg_remove

    REMBG_AVAILABLE = True

except ImportError:
    REMBG_AVAILABLE = False


# ============================================================================
# PROJECT PATHS
# ============================================================================

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_ROOT = os.path.dirname(
    SCRIPT_DIR
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "runs",
    "geometry"
)

DEFAULT_WEIGHTS = os.path.join(
    PROJECT_ROOT,
    "runs",
    "segment",
    "models",
    "sofa_yolo11n_seg_v1-2",
    "weights",
    "best.pt"
)


# ============================================================================
# YOLO CLASSES
# ============================================================================

CLASS_NAMES = [
    "back_cushion",
    "base",
    "left_arm",
    "legs",
    "right_arm",
    "seat_cushion",
]


# ============================================================================
# GEOMETRY SETTINGS
# ============================================================================

# Very small masks are normally noise.
MIN_COMPONENT_AREA = 100

# Geometry confidence threshold.
GEOMETRY_CONFIDENCE = 0.50

# Detection confidence.
DEFAULT_CONFIDENCE = 0.25

# ---------------------------------------------------------------------------
# Continuous-cushion classification thresholds.
#
# These are image-space ratios, so no cm calibration is required.
#
# Main ratio:
#
#     seat cushion width / overall sofa width
#
# Typical interpretation:
#
#     < 0.55  -> 1-seater
#     0.55-0.76 -> 2-seater
#     0.76-0.92 -> 3-seater
#     > 0.92 -> 4-seater
#
# These are deliberately used together with other geometry signals rather
# than as a single hard rule.
# ---------------------------------------------------------------------------

SEAT_RATIO_1_MAX = 0.55
SEAT_RATIO_2_MAX = 0.76
SEAT_RATIO_3_MAX = 0.92

# Arm inner span ratio.
INNER_SPAN_1_MAX = 0.48
INNER_SPAN_2_MAX = 0.72
INNER_SPAN_3_MAX = 0.86

# Seat cushion aspect ratio.
# Very short/wide cushions are unlikely to represent a single seat.
SEAT_AR_2_MIN = 2.5
SEAT_AR_3_MIN = 4.0
SEAT_AR_4_MIN = 5.5


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ComponentInstance:

    cls_name: str

    mask: np.ndarray

    bbox: Tuple[int, int, int, int]

    area_px: int

    confidence: float

    @property
    def width_px(self) -> int:
        return max(
            0,
            self.bbox[2] - self.bbox[0]
        )

    @property
    def height_px(self) -> int:
        return max(
            0,
            self.bbox[3] - self.bbox[1]
        )

    @property
    def center_x(self) -> float:
        return (
            self.bbox[0] +
            self.bbox[2]
        ) / 2.0

    @property
    def center_y(self) -> float:
        return (
            self.bbox[1] +
            self.bbox[3]
        ) / 2.0

    @property
    def aspect_ratio(self) -> float:

        if self.height_px <= 0:
            return 0.0

        return (
            self.width_px /
            self.height_px
        )


@dataclass
class SeaterCandidate:

    source: str

    count: int

    confidence: float

    note: str = ""


@dataclass
class SofaGeometryReport:

    seater_type: str

    seat_count_final: int

    seater_confidence: float

    chosen_source: str

    overall_width_px: float

    overall_height_px: float

    arm_inner_span_px: Optional[float]

    seat_width_px: Optional[float]

    seat_height_px: Optional[float]

    seat_aspect_ratio: Optional[float]

    seat_to_overall_ratio: Optional[float]

    seat_to_inner_span_ratio: Optional[float]

    inner_span_to_overall_ratio: Optional[float]

    seat_cushion_count: int

    back_cushion_count: int

    physical_left_arm_class: Optional[str]

    physical_right_arm_class: Optional[str]

    physical_left_arm_confidence: Optional[float]

    physical_right_arm_confidence: Optional[float]

    visible_leg_count: int

    shape: str

    is_sectional: bool

    solidity: float

    components_detected: List[str] = field(
        default_factory=list
    )

    warnings: List[str] = field(
        default_factory=list
    )

    explanation: List[str] = field(
        default_factory=list
    )

    def to_dict(self) -> Dict:

        return self.__dict__


# ============================================================================
# ANALYZER
# ============================================================================

class SofaGeometryAnalyzer:

    def __init__(
        self,
        weights_path: str,
        conf: float = DEFAULT_CONFIDENCE,
    ):

        print("\nLoading YOLO model...")

        self.model = YOLO(
            weights_path
        )

        self.conf = conf

        print(
            "Model loaded successfully."
        )

        print(
            f"Model: {weights_path}"
        )

        print(
            f"Prediction confidence: {conf}"
        )

        print(
            f"Geometry confidence: "
            f"{GEOMETRY_CONFIDENCE}"
        )


    # ========================================================================
    # BACKGROUND REMOVAL
    # ========================================================================

    def remove_background(
        self,
        image_path: str
    ) -> str:

        """
        Remove background using rembg.

        Returns a temporary RGB image path.

        If rembg is unavailable, returns original image.
        """

        if not REMBG_AVAILABLE:

            print(
                "\n[WARNING] rembg is not installed."
            )

            print(
                "Skipping background removal."
            )

            print(
                'Install with: '
                'python -m pip install "rembg[cpu]"'
            )

            return image_path

        try:

            with open(
                image_path,
                "rb"
            ) as f:

                input_bytes = f.read()

            output_bytes = rembg_remove(
                input_bytes
            )

            from PIL import Image
            import io

            fg = Image.open(
                io.BytesIO(
                    output_bytes
                )
            ).convert("RGBA")

            # Transparent pixels become white.
            background = Image.new(
                "RGB",
                fg.size,
                (255, 255, 255)
            )

            background.paste(
                fg,
                mask=fg.getchannel("A")
            )

            temp_fd, temp_path = tempfile.mkstemp(
                suffix="_nobg.png"
            )

            os.close(
                temp_fd
            )

            background.save(
                temp_path
            )

            return temp_path

        except Exception as exc:

            print(
                f"\n[WARNING] Background removal failed: "
                f"{exc}"
            )

            return image_path


    # ========================================================================
    # SAVE BACKGROUND REMOVED IMAGE
    # ========================================================================

    def save_background_removed_image(
        self,
        processed_path: str,
        original_path: str
    ) -> str:

        os.makedirs(
            OUTPUT_DIR,
            exist_ok=True
        )

        output_path = os.path.join(
            OUTPUT_DIR,
            "sofa_background_removed.png"
        )

        try:

            if processed_path != original_path:

                shutil.copy2(
                    processed_path,
                    output_path
                )

            else:

                image = cv2.imread(
                    original_path
                )

                if image is not None:

                    cv2.imwrite(
                        output_path,
                        image
                    )

        except Exception as exc:

            print(
                f"[WARNING] Could not save "
                f"background-removed image: {exc}"
            )

        return output_path


    # ========================================================================
    # YOLO INFERENCE
    # ========================================================================

    def run_inference(
        self,
        image_path: str
    ) -> List[ComponentInstance]:

        results = self.model.predict(
            image_path,
            conf=self.conf,
            verbose=False,
            retina_masks=True
        )[0]

        instances = []

        if results.masks is None:

            return instances

        img_h, img_w = results.orig_shape

        masks = (
            results.masks.data
            .cpu()
            .numpy()
        )

        boxes = (
            results.boxes.xyxy
            .cpu()
            .numpy()
        )

        classes = (
            results.boxes.cls
            .cpu()
            .numpy()
            .astype(int)
        )

        confidences = (
            results.boxes.conf
            .cpu()
            .numpy()
        )

        for i in range(
            len(classes)
        ):

            class_id = int(
                classes[i]
            )

            if (
                class_id < 0
                or class_id >= len(CLASS_NAMES)
            ):
                continue

            mask = cv2.resize(
                masks[i].astype(
                    np.uint8
                ),
                (
                    img_w,
                    img_h
                ),
                interpolation=cv2.INTER_NEAREST
            )

            x1, y1, x2, y2 = boxes[i]

            instance = ComponentInstance(

                cls_name=CLASS_NAMES[
                    class_id
                ],

                mask=mask,

                bbox=(
                    int(x1),
                    int(y1),
                    int(x2),
                    int(y2)
                ),

                area_px=int(
                    mask.sum()
                ),

                confidence=float(
                    confidences[i]
                )
            )

            instances.append(
                instance
            )

        return instances


    # ========================================================================
    # IOU
    # ========================================================================

    @staticmethod
    def calculate_iou(
        a: ComponentInstance,
        b: ComponentInstance
    ) -> float:

        ax1, ay1, ax2, ay2 = a.bbox

        bx1, by1, bx2, by2 = b.bbox

        ix1 = max(
            ax1,
            bx1
        )

        iy1 = max(
            ay1,
            by1
        )

        ix2 = min(
            ax2,
            bx2
        )

        iy2 = min(
            ay2,
            by2
        )

        iw = max(
            0,
            ix2 - ix1
        )

        ih = max(
            0,
            iy2 - iy1
        )

        intersection = (
            iw * ih
        )

        if intersection <= 0:

            return 0.0

        area_a = (
            max(
                0,
                ax2 - ax1
            )
            *
            max(
                0,
                ay2 - ay1
            )
        )

        area_b = (
            max(
                0,
                bx2 - bx1
            )
            *
            max(
                0,
                by2 - by1
            )
        )

        union = (
            area_a
            + area_b
            - intersection
        )

        if union <= 0:

            return 0.0

        return (
            intersection /
            union
        )


    # ========================================================================
    # DEDUPLICATION
    # ========================================================================

    def deduplicate_instances(
        self,
        instances: List[ComponentInstance],
        iou_threshold: float = 0.55
    ) -> List[ComponentInstance]:

        by_class = {}

        for instance in instances:

            by_class.setdefault(
                instance.cls_name,
                []
            ).append(
                instance
            )

        kept = []

        for cls_name, group in by_class.items():

            group = sorted(
                group,
                key=lambda x: x.confidence,
                reverse=True
            )

            chosen = []

            for candidate in group:

                duplicate = False

                for existing in chosen:

                    if (
                        self.calculate_iou(
                            candidate,
                            existing
                        )
                        >= iou_threshold
                    ):

                        duplicate = True
                        break

                if not duplicate:

                    chosen.append(
                        candidate
                    )

            kept.extend(
                chosen
            )

        return kept


    # ========================================================================
    # GEOMETRY FILTER
    # ========================================================================

    def geometry_valid_components(
        self,
        instances: List[ComponentInstance]
    ) -> List[ComponentInstance]:

        valid = []

        for instance in instances:

            if (
                instance.confidence
                < GEOMETRY_CONFIDENCE
            ):
                continue

            if (
                instance.area_px
                < MIN_COMPONENT_AREA
            ):
                continue

            valid.append(
                instance
            )

        return valid


    # ========================================================================
    # LEG FILTER
    # ========================================================================

    def filter_legs(
        self,
        instances: List[ComponentInstance]
    ) -> List[ComponentInstance]:

        """
        Keep visible/detected legs without inventing hidden legs.

        A leg should generally:
            - be reasonably small
            - be below the main sofa
            - have non-zero height
        """

        raw_legs = [
            i
            for i in instances
            if i.cls_name == "legs"
            and i.confidence >= GEOMETRY_CONFIDENCE
            and i.area_px >= MIN_COMPONENT_AREA
        ]

        if not raw_legs:

            return []

        non_legs = [
            i
            for i in instances
            if i.cls_name != "legs"
        ]

        if not non_legs:

            return raw_legs

        sofa_bottom = max(
            i.bbox[3]
            for i in non_legs
        )

        candidates = []

        for leg in raw_legs:

            # Leg should normally be near/below sofa bottom.
            if leg.bbox[1] < sofa_bottom - 20:

                continue

            # Extremely large regions are probably false positives.
            if (
                leg.width_px
                > 0.25
                * max(
                    1,
                    max(
                        x.width_px
                        for x in non_legs
                    )
                )
            ):

                continue

            candidates.append(
                leg
            )

        # Remove duplicate leg detections.
        candidates = sorted(
            candidates,
            key=lambda x: x.confidence,
            reverse=True
        )

        filtered = []

        for candidate in candidates:

            duplicate = False

            for existing in filtered:

                center_distance = abs(
                    candidate.center_x
                    - existing.center_x
                )

                if (
                    self.calculate_iou(
                        candidate,
                        existing
                    ) > 0.2
                    or center_distance < 8
                ):

                    duplicate = True
                    break

            if not duplicate:

                filtered.append(
                    candidate
                )

        return filtered


    # ========================================================================
    # PHYSICAL ARM ASSIGNMENT
    # ========================================================================

    def assign_physical_arms(
        self,
        instances: List[ComponentInstance]
    ) -> Tuple[
        Optional[ComponentInstance],
        Optional[ComponentInstance]
    ]:

        """
        IMPORTANT:

        Do not trust YOLO left_arm/right_arm class to determine
        physical image side.

        Smaller X = physical left.
        Larger X = physical right.
        """

        arms = [
            i
            for i in instances
            if i.cls_name in (
                "left_arm",
                "right_arm"
            )
            and i.confidence >= GEOMETRY_CONFIDENCE
            and i.area_px >= MIN_COMPONENT_AREA
        ]

        if len(arms) < 2:

            return None, None

        arms = sorted(
            arms,
            key=lambda x: x.center_x
        )

        return (
            arms[0],
            arms[-1]
        )


    # ========================================================================
    # OVERALL SOFA BOUNDING BOX
    # ========================================================================

    def calculate_overall_bbox(
        self,
        instances: List[ComponentInstance]
    ) -> Optional[
        Tuple[int, int, int, int]
    ]:

        useful = [
            i
            for i in instances
            if i.cls_name != "legs"
        ]

        if not useful:

            useful = instances

        if not useful:

            return None

        x1 = min(
            i.bbox[0]
            for i in useful
        )

        y1 = min(
            i.bbox[1]
            for i in useful
        )

        x2 = max(
            i.bbox[2]
            for i in useful
        )

        y2 = max(
            i.bbox[3]
            for i in useful
        )

        # Include visible legs in height only.
        legs = [
            i
            for i in instances
            if i.cls_name == "legs"
        ]

        if legs:

            y2 = max(
                y2,
                max(
                    i.bbox[3]
                    for i in legs
                )
            )

        return (
            x1,
            y1,
            x2,
            y2
        )


    # ========================================================================
    # SEAT CUSHION
    # ========================================================================

    def get_main_seat_cushion(
        self,
        instances: List[ComponentInstance]
    ) -> Optional[ComponentInstance]:

        cushions = [
            i
            for i in instances
            if i.cls_name == "seat_cushion"
            and i.confidence >= GEOMETRY_CONFIDENCE
            and i.area_px >= MIN_COMPONENT_AREA
        ]

        if not cushions:

            return None

        # For continuous cushion sofas, use the largest seat cushion.
        return max(
            cushions,
            key=lambda x: x.area_px
        )


    # ========================================================================
    # ARM INNER SPAN
    # ========================================================================

    def calculate_arm_inner_span(
        self,
        physical_left: Optional[ComponentInstance],
        physical_right: Optional[ComponentInstance]
    ) -> Optional[float]:

        if (
            physical_left is None
            or physical_right is None
        ):

            return None

        left_inner = (
            physical_left.bbox[2]
        )

        right_inner = (
            physical_right.bbox[0]
        )

        span = (
            right_inner
            - left_inner
        )

        return max(
            0.0,
            float(span)
        )


    # ========================================================================
    # SEATER ESTIMATION
    # ========================================================================

    def estimate_seater_from_geometry(
        self,
        overall_width: float,
        inner_span: Optional[float],
        seat: Optional[ComponentInstance],
        seat_count: int,
        back_count: int
    ) -> Tuple[
        int,
        float,
        str,
        List[str],
        Dict
    ]:

        """
        Main seater classifier.

        IMPORTANT:
            It does NOT assume:
                one cushion = one seat.

        Geometry is the main signal.
        """

        explanation = []

        if overall_width <= 0:

            return (
                0,
                0.0,
                "none",
                [
                    "Overall sofa width unavailable."
                ],
                {}
            )

        if seat is None:

            # No seat cushion.
            # Use inner span if available.
            if inner_span is not None:

                inner_ratio = (
                    inner_span /
                    overall_width
                )

                if inner_ratio < INNER_SPAN_1_MAX:

                    count = 1

                elif inner_ratio < INNER_SPAN_2_MAX:

                    count = 2

                elif inner_ratio < INNER_SPAN_3_MAX:

                    count = 3

                else:

                    count = 4

                confidence = 0.55

                explanation.append(
                    "No seat cushion was detected. "
                    "Seater type estimated from "
                    "physical arm-to-arm span."
                )

                return (
                    count,
                    confidence,
                    "arm_inner_span",
                    explanation,
                    {
                        "inner_span_ratio":
                            inner_ratio
                    }
                )

            return (
                0,
                0.0,
                "none",
                [
                    "Insufficient geometry for "
                    "seater classification."
                ],
                {}
            )

        seat_width = float(
            seat.width_px
        )

        seat_height = float(
            seat.height_px
        )

        seat_ratio = (
            seat_width /
            overall_width
        )

        seat_ar = (
            seat_width /
            seat_height
            if seat_height > 0
            else 0
        )

        inner_ratio = None
        seat_inner_ratio = None

        if inner_span is not None:

            inner_ratio = (
                inner_span /
                overall_width
            )

            if inner_span > 0:

                seat_inner_ratio = (
                    seat_width /
                    inner_span
                )

        explanation.append(
            f"Main seat cushion width = "
            f"{seat_width:.1f}px."
        )

        explanation.append(
            f"Overall sofa width = "
            f"{overall_width:.1f}px."
        )

        explanation.append(
            f"Seat/overall width ratio = "
            f"{seat_ratio:.3f}."
        )

        if inner_span is not None:

            explanation.append(
                f"Usable arm-to-arm span = "
                f"{inner_span:.1f}px."
            )

            explanation.append(
                f"Inner-span/overall ratio = "
                f"{inner_ratio:.3f}."
            )

            explanation.append(
                f"Seat/inner-span ratio = "
                f"{seat_inner_ratio:.3f}."
            )

        explanation.append(
            f"Seat cushion aspect ratio = "
            f"{seat_ar:.2f}."
        )

        # ------------------------------------------------------------------
        # SIGNAL 1: seat/overall ratio
        # ------------------------------------------------------------------

        if seat_ratio < SEAT_RATIO_1_MAX:

            ratio_count = 1

        elif seat_ratio < SEAT_RATIO_2_MAX:

            ratio_count = 2

        elif seat_ratio < SEAT_RATIO_3_MAX:

            ratio_count = 3

        else:

            ratio_count = 4

        # ------------------------------------------------------------------
        # SIGNAL 2: arm inner span
        # ------------------------------------------------------------------

        if inner_ratio is not None:

            if inner_ratio < INNER_SPAN_1_MAX:

                inner_count = 1

            elif inner_ratio < INNER_SPAN_2_MAX:

                inner_count = 2

            elif inner_ratio < INNER_SPAN_3_MAX:

                inner_count = 3

            else:

                inner_count = 4

        else:

            inner_count = None

        # ------------------------------------------------------------------
        # SIGNAL 3: cushion aspect ratio
        # ------------------------------------------------------------------

        if seat_ar < SEAT_AR_2_MIN:

            aspect_count = 1

        elif seat_ar < SEAT_AR_3_MIN:

            aspect_count = 2

        elif seat_ar < SEAT_AR_4_MIN:

            aspect_count = 3

        else:

            aspect_count = 4

        # ------------------------------------------------------------------
        # Weighted geometry vote
        # ------------------------------------------------------------------

        votes = []

        # Seat width ratio is strongest.
        votes.append(
            (
                ratio_count,
                0.50
            )
        )

        # Arm span is second strongest.
        if inner_count is not None:

            votes.append(
                (
                    inner_count,
                    0.30
                )
            )

        # Aspect ratio is supporting evidence.
        votes.append(
            (
                aspect_count,
                0.20
            )
        )

        scores = {
            1: 0.0,
            2: 0.0,
            3: 0.0,
            4: 0.0
        }

        for count, weight in votes:

            scores[count] += weight

        predicted = max(
            scores,
            key=scores.get
        )

        score = scores[
            predicted
        ]

        # ------------------------------------------------------------------
        # Continuous cushion correction
        # ------------------------------------------------------------------

        if seat_count == 1:

            explanation.append(
                "Only one continuous seat cushion "
                "was detected."
            )

            explanation.append(
                "The cushion count is NOT interpreted "
                "as the final seat count."
            )

        # ------------------------------------------------------------------
        # Important correction for 2-seater geometry
        # ------------------------------------------------------------------

        # If the geometry strongly supports 2 seats,
        # do not allow one-cushion counting to reduce it to 1.
        if (
            predicted == 1
            and ratio_count == 2
        ):

            predicted = 2
            score = max(
                score,
                0.70
            )

            explanation.append(
                "Continuous-cushion correction: "
                "geometry supports 2 seats even "
                "though only one seat-cushion "
                "instance was detected."
            )

        # ------------------------------------------------------------------
        # Additional consistency rules
        # ------------------------------------------------------------------

        if (
            predicted == 1
            and seat_ar >= SEAT_AR_2_MIN
            and seat_ratio >= 0.55
        ):

            predicted = 2

            score = max(
                score,
                0.65
            )

            explanation.append(
                "Seat cushion is too wide to be "
                "treated as a single seating position."
            )

        if (
            predicted == 4
            and seat_ratio < 0.86
        ):

            # Prevent an overly aggressive 4-seat result
            # from a wide but normal 2/3-seater cushion.
            predicted = 3

            score = max(
                score,
                0.65
            )

            explanation.append(
                "4-seat estimate was reduced because "
                "the seat/overall width ratio does "
                "not strongly support four seats."
            )

        # ------------------------------------------------------------------
        # Confidence
        # ------------------------------------------------------------------

        # Agreement increases confidence.
        signal_counts = [
            ratio_count,
            aspect_count
        ]

        if inner_count is not None:

            signal_counts.append(
                inner_count
            )

        agreement = (
            signal_counts.count(
                predicted
            )
            /
            len(signal_counts)
        )

        confidence = (
            0.55
            +
            0.40 * agreement
        )

        confidence = min(
            confidence,
            0.95
        )

        # If exactly one cushion is detected,
        # reduce confidence slightly because the model
        # does not directly observe individual seats.
        if seat_count == 1:

            confidence -= 0.05

        confidence = max(
            0.50,
            confidence
        )

        explanation.append(
            f"Geometry vote = "
            f"{predicted}-seater."
        )

        explanation.append(
            f"Geometry confidence = "
            f"{confidence * 100:.1f}%."
        )

        diagnostics = {

            "ratio_count":
                ratio_count,

            "inner_span_count":
                inner_count,

            "aspect_count":
                aspect_count,

            "scores":
                scores,

            "seat_ratio":
                seat_ratio,

            "inner_ratio":
                inner_ratio,

            "seat_inner_ratio":
                seat_inner_ratio,

            "seat_aspect_ratio":
                seat_ar
        }

        return (
            predicted,
            confidence,
            "geometry",
            explanation,
            diagnostics
        )


    # ========================================================================
    # SHAPE CLASSIFICATION
    # ========================================================================

    def classify_shape(
        self,
        instances: List[ComponentInstance]
    ) -> Tuple[
        str,
        float,
        bool
    ]:

        structural = [
            i
            for i in instances
            if i.cls_name in (
                "base",
                "left_arm",
                "right_arm"
            )
        ]

        if not structural:

            return (
                "unknown",
                0.0,
                False
            )

        combined = np.zeros_like(
            structural[0].mask,
            dtype=np.uint8
        )

        for instance in structural:

            combined = np.logical_or(
                combined,
                instance.mask
            ).astype(
                np.uint8
            )

        contours, _ = cv2.findContours(
            combined,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:

            return (
                "unknown",
                0.0,
                False
            )

        largest = max(
            contours,
            key=cv2.contourArea
        )

        area = cv2.contourArea(
            largest
        )

        hull = cv2.convexHull(
            largest
        )

        hull_area = cv2.contourArea(
            hull
        )

        if hull_area <= 0:

            solidity = 1.0

        else:

            solidity = (
                area /
                hull_area
            )

        if solidity > 0.90:

            return (
                "straight",
                round(
                    float(solidity),
                    3
                ),
                False
            )

        if solidity > 0.75:

            return (
                "chaise / extended",
                round(
                    float(solidity),
                    3
                ),
                True
            )

        return (
            "L-shaped / sectional",
            round(
                float(solidity),
                3
            ),
            True
        )


    # ========================================================================
    # VISUALIZATION
    # ========================================================================

    def create_visualization(
        self,
        image_path: str,
        instances: List[ComponentInstance],
        physical_left: Optional[ComponentInstance],
        physical_right: Optional[ComponentInstance],
        legs: List[ComponentInstance],
        report: SofaGeometryReport
    ) -> str:

        image = cv2.imread(
            image_path
        )

        if image is None:

            raise ValueError(
                f"Could not read image: "
                f"{image_path}"
            )

        # ---------------------------------------------------------------
        # Draw component masks / boxes
        # ---------------------------------------------------------------

        for instance in instances:

            # Only visualize geometry-valid components.
            if (
                instance.confidence
                < GEOMETRY_CONFIDENCE
            ):
                continue

            if (
                instance.area_px
                < MIN_COMPONENT_AREA
            ):
                continue

            color = (
                255,
                0,
                0
            )

            if instance.cls_name == "seat_cushion":

                color = (
                    0,
                    165,
                    255
                )

            elif instance.cls_name == "back_cushion":

                color = (
                    255,
                    0,
                    0
                )

            elif instance.cls_name == "base":

                color = (
                    0,
                    255,
                    0
                )

            elif instance.cls_name in (
                "left_arm",
                "right_arm"
            ):

                color = (
                    255,
                    0,
                    255
                )

            elif instance.cls_name == "legs":

                color = (
                    0,
                    255,
                    255
                )

            # Contour.
            contours, _ = cv2.findContours(
                instance.mask.astype(
                    np.uint8
                ),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            cv2.drawContours(
                image,
                contours,
                -1,
                color,
                2
            )

            x1, y1, x2, y2 = (
                instance.bbox
            )

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                color,
                1
            )

            label = (
                f"{instance.cls_name} "
                f"{instance.confidence:.2f}"
            )

            cv2.putText(
                image,
                label,
                (
                    x1,
                    max(
                        15,
                        y1 - 5
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA
            )

        # ---------------------------------------------------------------
        # Physical arm labels
        # ---------------------------------------------------------------

        if physical_left is not None:

            x = int(
                physical_left.center_x
            )

            y = max(
                20,
                int(
                    physical_left.bbox[1]
                    - 20
                )
            )

            cv2.putText(
                image,
                "PHYSICAL LEFT ARM",
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 0, 255),
                2,
                cv2.LINE_AA
            )

        if physical_right is not None:

            x = int(
                physical_right.center_x
            )

            y = max(
                20,
                int(
                    physical_right.bbox[1]
                    - 20
                )
            )

            cv2.putText(
                image,
                "PHYSICAL RIGHT ARM",
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 0, 255),
                2,
                cv2.LINE_AA
            )

        # ---------------------------------------------------------------
        # Top result panel
        # ---------------------------------------------------------------

        panel_height = 115

        overlay = image.copy()

        cv2.rectangle(
            overlay,
            (0, 0),
            (
                image.shape[1],
                panel_height
            ),
            (0, 0, 0),
            -1
        )

        image = cv2.addWeighted(
            overlay,
            0.70,
            image,
            0.30,
            0
        )

        cv2.putText(
            image,
            f"SOFA TYPE: {report.seater_type}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            image,
            f"CONFIDENCE: "
            f"{report.seater_confidence * 100:.1f}%",
            (10, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            image,
            f"SEAT CUSHIONS: "
            f"{report.seat_cushion_count}",
            (10, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            image,
            f"VISIBLE LEGS: "
            f"{report.visible_leg_count}",
            (10, 108),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        # ---------------------------------------------------------------
        # Geometry measurements
        # ---------------------------------------------------------------

        if (
            physical_left is not None
            and physical_right is not None
        ):

            left_inner = (
                physical_left.bbox[2]
            )

            right_inner = (
                physical_right.bbox[0]
            )

            y = int(
                max(
                    physical_left.bbox[3],
                    physical_right.bbox[3]
                )
                + 15
            )

            y = min(
                y,
                image.shape[0] - 10
            )

            cv2.line(
                image,
                (
                    left_inner,
                    y
                ),
                (
                    right_inner,
                    y
                ),
                (255, 255, 255),
                2
            )

            cv2.putText(
                image,
                "USABLE SEATING SPAN",
                (
                    left_inner,
                    max(
                        15,
                        y - 5
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

        # ---------------------------------------------------------------
        # Save
        # ---------------------------------------------------------------

        os.makedirs(
            OUTPUT_DIR,
            exist_ok=True
        )

        output_path = os.path.join(
            OUTPUT_DIR,
            "geometry_result.jpg"
        )

        cv2.imwrite(
            output_path,
            image
        )

        return os.path.abspath(
            output_path
        )


    # ========================================================================
    # ANALYSIS
    # ========================================================================

    def analyze(
        self,
        image_path: str,
        remove_bg: bool = True,
        verbose: bool = True
    ) -> Tuple[
        SofaGeometryReport,
        List[ComponentInstance],
        Optional[ComponentInstance],
        Optional[ComponentInstance],
        List[ComponentInstance],
        str
    ]:

        start_total = time.time()

        # ---------------------------------------------------------------
        # Background removal
        # ---------------------------------------------------------------

        if verbose:

            print(
                "\n[1/5] Background processing..."
            )

        processed_path = image_path

        temp_created = False

        if remove_bg:

            processed_path = (
                self.remove_background(
                    image_path
                )
            )

            temp_created = (
                processed_path
                != image_path
            )

        background_output = (
            self.save_background_removed_image(
                processed_path,
                image_path
            )
        )

        # ---------------------------------------------------------------
        # YOLO
        # ---------------------------------------------------------------

        if verbose:

            print(
                "[2/5] Running YOLO segmentation..."
            )

        instances = self.run_inference(
            processed_path
        )

        if verbose:

            print(
                f"      Raw detections: "
                f"{len(instances)}"
            )

        # ---------------------------------------------------------------
        # Cleanup temporary file
        # ---------------------------------------------------------------

        if (
            temp_created
            and os.path.exists(
                processed_path
            )
        ):

            try:

                os.remove(
                    processed_path
                )

            except Exception:

                pass

        # ---------------------------------------------------------------
        # Deduplication
        # ---------------------------------------------------------------

        if verbose:

            print(
                "[3/5] Removing duplicate detections..."
            )

        instances = (
            self.deduplicate_instances(
                instances
            )
        )

        # ---------------------------------------------------------------
        # Geometry valid
        # ---------------------------------------------------------------

        geometry_instances = (
            self.geometry_valid_components(
                instances
            )
        )

        if verbose:

            print(
                f"      Geometry-valid detections: "
                f"{len(geometry_instances)}"
            )

        # ---------------------------------------------------------------
        # Arms
        # ---------------------------------------------------------------

        physical_left, physical_right = (
            self.assign_physical_arms(
                geometry_instances
            )
        )

        # ---------------------------------------------------------------
        # Legs
        # ---------------------------------------------------------------

        visible_legs = (
            self.filter_legs(
                geometry_instances
            )
        )

        # ---------------------------------------------------------------
        # Overall bbox
        # ---------------------------------------------------------------

        overall_bbox = (
            self.calculate_overall_bbox(
                geometry_instances
            )
        )

        if overall_bbox is None:

            raise RuntimeError(
                "Could not determine sofa geometry."
            )

        ox1, oy1, ox2, oy2 = (
            overall_bbox
        )

        overall_width = float(
            ox2 - ox1
        )

        overall_height = float(
            oy2 - oy1
        )

        # ---------------------------------------------------------------
        # Seat
        # ---------------------------------------------------------------

        main_seat = (
            self.get_main_seat_cushion(
                geometry_instances
            )
        )

        seat_cushions = [
            i
            for i in geometry_instances
            if i.cls_name == "seat_cushion"
        ]

        back_cushions = [
            i
            for i in geometry_instances
            if i.cls_name == "back_cushion"
        ]

        # ---------------------------------------------------------------
        # Arm inner span
        # ---------------------------------------------------------------

        inner_span = (
            self.calculate_arm_inner_span(
                physical_left,
                physical_right
            )
        )

        # ---------------------------------------------------------------
        # Seater classification
        # ---------------------------------------------------------------

        if verbose:

            print(
                "[4/5] Calculating seater geometry..."
            )

        (
            seat_count,
            seater_confidence,
            chosen_source,
            seater_explanation,
            diagnostics
        ) = self.estimate_seater_from_geometry(

            overall_width=overall_width,

            inner_span=inner_span,

            seat=main_seat,

            seat_count=len(
                seat_cushions
            ),

            back_count=len(
                back_cushions
            )
        )

        seater_names = {
            1: "1-seater",
            2: "2-seater",
            3: "3-seater",
            4: "4-seater"
        }

        seater_type = seater_names.get(
            seat_count,
            "unknown"
        )

        # ---------------------------------------------------------------
        # Seat measurements
        # ---------------------------------------------------------------

        seat_width = None
        seat_height = None
        seat_ar = None
        seat_overall_ratio = None
        seat_inner_ratio = None
        inner_overall_ratio = None

        if main_seat is not None:

            seat_width = float(
                main_seat.width_px
            )

            seat_height = float(
                main_seat.height_px
            )

            if seat_height > 0:

                seat_ar = (
                    seat_width /
                    seat_height
                )

            seat_overall_ratio = (
                seat_width /
                overall_width
            )

            if inner_span is not None:

                if inner_span > 0:

                    seat_inner_ratio = (
                        seat_width /
                        inner_span
                    )

                inner_overall_ratio = (
                    inner_span /
                    overall_width
                )

        # ---------------------------------------------------------------
        # Shape
        # ---------------------------------------------------------------

        shape, solidity, sectional = (
            self.classify_shape(
                geometry_instances
            )
        )

        # ---------------------------------------------------------------
        # Warnings
        # ---------------------------------------------------------------

        warnings = []

        if len(
            seat_cushions
        ) == 1:

            warnings.append(
                "One continuous seat cushion detected. "
                "Seat cushion count was not used as "
                "the final seat count."
            )

        if (
            physical_left is None
            or physical_right is None
        ):

            warnings.append(
                "Two physical arms were not reliably "
                "detected. Arm-to-arm span is unavailable."
            )

        if not back_cushions:

            warnings.append(
                "No back cushion detected."
            )

        if not visible_legs:

            warnings.append(
                "No geometry-valid visible legs detected."
            )

        # ---------------------------------------------------------------
        # Components
        # ---------------------------------------------------------------

        components_detected = sorted(
            {
                i.cls_name
                for i in geometry_instances
            }
        )

        # ---------------------------------------------------------------
        # Report
        # ---------------------------------------------------------------

        report = SofaGeometryReport(

            seater_type=seater_type,

            seat_count_final=seat_count,

            seater_confidence=seater_confidence,

            chosen_source=chosen_source,

            overall_width_px=overall_width,

            overall_height_px=overall_height,

            arm_inner_span_px=inner_span,

            seat_width_px=seat_width,

            seat_height_px=seat_height,

            seat_aspect_ratio=seat_ar,

            seat_to_overall_ratio=seat_overall_ratio,

            seat_to_inner_span_ratio=seat_inner_ratio,

            inner_span_to_overall_ratio=inner_overall_ratio,

            seat_cushion_count=len(
                seat_cushions
            ),

            back_cushion_count=len(
                back_cushions
            ),

            physical_left_arm_class=(
                physical_left.cls_name
                if physical_left
                else None
            ),

            physical_right_arm_class=(
                physical_right.cls_name
                if physical_right
                else None
            ),

            physical_left_arm_confidence=(
                physical_left.confidence
                if physical_left
                else None
            ),

            physical_right_arm_confidence=(
                physical_right.confidence
                if physical_right
                else None
            ),

            visible_leg_count=len(
                visible_legs
            ),

            shape=shape,

            is_sectional=sectional,

            solidity=solidity,

            components_detected=components_detected,

            warnings=warnings,

            explanation=seater_explanation
        )

        if verbose:

            print(
                "[5/5] Analysis complete."
            )

        print(
            f"\nFINAL SEATER TYPE: "
            f"{report.seater_type}"
        )

        print(
            f"CONFIDENCE: "
            f"{report.seater_confidence * 100:.1f}%"
        )

        print(
            f"Seat cushions detected: "
            f"{report.seat_cushion_count}"
        )

        print(
            f"Back cushions detected: "
            f"{report.back_cushion_count}"
        )

        print(
            f"Visible legs: "
            f"{report.visible_leg_count}"
        )

        print(
            f"Overall width: "
            f"{report.overall_width_px:.1f}px"
        )

        if report.arm_inner_span_px is not None:

            print(
                f"Usable seating span: "
                f"{report.arm_inner_span_px:.1f}px"
            )

        if report.seat_to_overall_ratio is not None:

            print(
                f"Seat/overall ratio: "
                f"{report.seat_to_overall_ratio:.3f}"
            )

        # ---------------------------------------------------------------
        # Visualization
        # ---------------------------------------------------------------

        visualization_path = (
            self.create_visualization(
                image_path=processed_path
                if os.path.exists(processed_path)
                else image_path,
                instances=geometry_instances,
                physical_left=physical_left,
                physical_right=physical_right,
                legs=visible_legs,
                report=report
            )
        )

        # If temporary background image no longer exists,
        # regenerate visualization from original.
        if not os.path.exists(
            visualization_path
        ):

            visualization_path = (
                self.create_visualization(
                    image_path=image_path,
                    instances=geometry_instances,
                    physical_left=physical_left,
                    physical_right=physical_right,
                    legs=visible_legs,
                    report=report
                )
            )

        print(
            f"\nAnalysis time: "
            f"{time.time() - start_total:.1f}s"
        )

        return (
            report,
            geometry_instances,
            physical_left,
            physical_right,
            visible_legs,
            background_output
        )


# ============================================================================
# SAVE TEXT REPORT
# ============================================================================

def save_text_report(
    report: SofaGeometryReport,
    path: str
) -> None:

    lines = []

    lines.append(
        "SOFA GEOMETRY ANALYSIS"
    )

    lines.append(
        "=" * 70
    )

    lines.append(
        f"Final seater type: "
        f"{report.seater_type}"
    )

    lines.append(
        f"Seater confidence: "
        f"{report.seater_confidence * 100:.1f}%"
    )

    lines.append(
        f"Decision source: "
        f"{report.chosen_source}"
    )

    lines.append("")

    lines.append(
        "COMPONENT COUNTS"
    )

    lines.append(
        "-" * 70
    )

    lines.append(
        f"Seat cushions: "
        f"{report.seat_cushion_count}"
    )

    lines.append(
        f"Back cushions: "
        f"{report.back_cushion_count}"
    )

    lines.append(
        f"Visible legs: "
        f"{report.visible_leg_count}"
    )

    lines.append("")

    lines.append(
        "PHYSICAL ARM ASSIGNMENT"
    )

    lines.append(
        "-" * 70
    )

    lines.append(
        f"Physical LEFT arm YOLO class: "
        f"{report.physical_left_arm_class}"
    )

    lines.append(
        f"Physical RIGHT arm YOLO class: "
        f"{report.physical_right_arm_class}"
    )

    lines.append("")

    lines.append(
        "GEOMETRY"
    )

    lines.append(
        "-" * 70
    )

    lines.append(
        f"Overall width: "
        f"{report.overall_width_px:.2f}px"
    )

    lines.append(
        f"Overall height: "
        f"{report.overall_height_px:.2f}px"
    )

    if report.arm_inner_span_px is not None:

        lines.append(
            f"Arm inner span: "
            f"{report.arm_inner_span_px:.2f}px"
        )

    if report.seat_width_px is not None:

        lines.append(
            f"Seat width: "
            f"{report.seat_width_px:.2f}px"
        )

    if report.seat_height_px is not None:

        lines.append(
            f"Seat height: "
            f"{report.seat_height_px:.2f}px"
        )

    if report.seat_aspect_ratio is not None:

        lines.append(
            f"Seat aspect ratio: "
            f"{report.seat_aspect_ratio:.3f}"
        )

    if report.seat_to_overall_ratio is not None:

        lines.append(
            f"Seat/overall ratio: "
            f"{report.seat_to_overall_ratio:.3f}"
        )

    if report.seat_to_inner_span_ratio is not None:

        lines.append(
            f"Seat/inner-span ratio: "
            f"{report.seat_to_inner_span_ratio:.3f}"
        )

    if report.inner_span_to_overall_ratio is not None:

        lines.append(
            f"Inner-span/overall ratio: "
            f"{report.inner_span_to_overall_ratio:.3f}"
        )

    lines.append("")

    lines.append(
        "SHAPE"
    )

    lines.append(
        "-" * 70
    )

    lines.append(
        f"Shape: "
        f"{report.shape}"
    )

    lines.append(
        f"Sectional: "
        f"{report.is_sectional}"
    )

    lines.append(
        f"Solidity: "
        f"{report.solidity:.3f}"
    )

    lines.append("")

    lines.append(
        "SEATER REASONING"
    )

    lines.append(
        "-" * 70
    )

    for explanation in report.explanation:

        lines.append(
            explanation
        )

    lines.append("")

    lines.append(
        "WARNINGS"
    )

    lines.append(
        "-" * 70
    )

    if report.warnings:

        for warning in report.warnings:

            lines.append(
                warning
            )

    else:

        lines.append(
            "None"
        )

    lines.append("")

    lines.append(
        "DETECTED COMPONENTS"
    )

    lines.append(
        "-" * 70
    )

    for component in report.components_detected:

        lines.append(
            component
        )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "\n".join(
                lines
            )
        )


# ============================================================================
# INTERACTIVE IMAGE PATH
# ============================================================================

def prompt_for_image_path() -> str:

    while True:

        path = input(
            "\nEnter path to sofa image: "
        ).strip()

        path = (
            path
            .strip('"')
            .strip("'")
        )

        if not path:

            print(
                "Please enter a path."
            )

            continue

        if not os.path.isfile(path):

            print(
                f"File not found:\n{path}"
            )

            continue

        return os.path.abspath(
            path
        )


# ============================================================================
# MAIN
# ============================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Sofa geometry and seater analysis"
        )
    )

    parser.add_argument(
        "--image",
        default=None,
        help=(
            "Path to sofa image. "
            "If omitted, program asks interactively."
        )
    )

    parser.add_argument(
        "--weights",
        default=DEFAULT_WEIGHTS,
        help=(
            "Path to trained YOLO segmentation model."
        )
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONFIDENCE,
        help=(
            "YOLO prediction confidence. "
            "Default: 0.25"
        )
    )

    parser.add_argument(
        "--no-bg-removal",
        action="store_true",
        help=(
            "Skip rembg background removal."
        )
    )

    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Image
    # -----------------------------------------------------------------------

    if args.image:

        image_path = (
            args.image
            .strip()
            .strip('"')
            .strip("'")
        )

    else:

        image_path = (
            prompt_for_image_path()
        )

    image_path = os.path.abspath(
        image_path
    )

    if not os.path.isfile(
        image_path
    ):

        print(
            f"ERROR: image does not exist:\n"
            f"{image_path}"
        )

        sys.exit(1)

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------

    weights_path = args.weights

    if not os.path.isabs(
        weights_path
    ):

        weights_path = os.path.abspath(
            weights_path
        )

    if not os.path.isfile(
        weights_path
    ):

        print(
            "\nERROR: YOLO weights not found:"
        )

        print(
            weights_path
        )

        print(
            "\nExpected:"
        )

        print(
            DEFAULT_WEIGHTS
        )

        sys.exit(1)

    # -----------------------------------------------------------------------
    # Output directory
    # -----------------------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "SOFA GEOMETRY ANALYSIS"
    )

    print(
        "=" * 70
    )

    print(
        f"\nInput image:"
    )

    print(
        image_path
    )

    print(
        f"\nOutput directory:"
    )

    print(
        OUTPUT_DIR
    )

    # -----------------------------------------------------------------------
    # Analyzer
    # -----------------------------------------------------------------------

    analyzer = SofaGeometryAnalyzer(
        weights_path=weights_path,
        conf=args.conf
    )

    # -----------------------------------------------------------------------
    # Analyze
    # -----------------------------------------------------------------------

    (
        report,
        instances,
        physical_left,
        physical_right,
        visible_legs,
        background_output
    ) = analyzer.analyze(

        image_path=image_path,

        remove_bg=(
            not args.no_bg_removal
        ),

        verbose=True
    )

    # -----------------------------------------------------------------------
    # Save JSON
    # -----------------------------------------------------------------------

    json_path = os.path.join(
        OUTPUT_DIR,
        "geometry_result.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report.to_dict(),
            f,
            indent=4,
            ensure_ascii=False
        )

    # -----------------------------------------------------------------------
    # Save TXT
    # -----------------------------------------------------------------------

    txt_path = os.path.join(
        OUTPUT_DIR,
        "geometry_result.txt"
    )

    save_text_report(
        report,
        txt_path
    )

    # -----------------------------------------------------------------------
    # Final output
    # -----------------------------------------------------------------------

    visualization_path = os.path.join(
        OUTPUT_DIR,
        "geometry_result.jpg"
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FINAL RESULT"
    )

    print(
        "=" * 70
    )

    print(
        f"\nSEATER TYPE: "
        f"{report.seater_type}"
    )

    print(
        f"CONFIDENCE: "
        f"{report.seater_confidence * 100:.1f}%"
    )

    print(
        f"\nJSON:"
    )

    print(
        json_path
    )

    print(
        f"\nTEXT REPORT:"
    )

    print(
        txt_path
    )

    print(
        f"\nVISUALIZATION:"
    )

    print(
        visualization_path
    )

    print(
        f"\nBACKGROUND REMOVED:"
    )

    print(
        background_output
    )

    print(
        "\n"
        + "=" * 70
    )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":

    main()