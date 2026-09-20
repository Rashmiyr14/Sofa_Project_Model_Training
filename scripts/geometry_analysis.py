"""
sofa_geometry_analysis.py

YOLO11 segmentation + sofa geometry + seater classification.

FINAL VISUALIZATION:
    SOFA TYPE: 2-seater
    CONFIDENCE: 90.0%

    Original YOLO class names are preserved:
        back_cushion
        base
        left_arm
        legs
        right_arm
        seat_cushion

    No PHYSICAL LEFT/RIGHT labels are displayed.
    No bounding boxes are displayed.

OUTPUT DIRECTORY:
    runs/geometry/

OUTPUT FILES:
    runs/geometry/sofa_geometry_result.png
    runs/geometry/geometry_result.txt
    runs/geometry/geometry_report.json

USAGE:

    python scripts/geometry_analysis.py

or:

    python scripts/geometry_analysis.py --image "path/to/sofa.jpg"

Model:
    runs/segment/models/sofa_yolo11n_seg_v1-2/weights/best.pt
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import List, Dict, Tuple


import cv2
import numpy as np
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

CLASS_NAMES = [
    "back_cushion",
    "base",
    "left_arm",
    "legs",
    "right_arm",
    "seat_cushion",
]


MODEL_PATH = (
    r".\runs\segment\models\sofa_yolo11n_seg_v1-2\weights\best.pt"
)


OUTPUT_DIR = r".\runs\geometry"


# YOLO detection threshold.
PREDICTION_CONFIDENCE = 0.25


# Only detections above this threshold are used for geometry.
GEOMETRY_CONFIDENCE = 0.50


# Minimum segmentation mask area.
MIN_COMPONENT_AREA = 100


# ============================================================
# DATA STRUCTURE
# ============================================================

@dataclass
class Component:

    cls_name: str

    confidence: float

    mask: np.ndarray

    bbox: Tuple[int, int, int, int]

    area_px: float

    @property
    def x1(self):
        return self.bbox[0]

    @property
    def y1(self):
        return self.bbox[1]

    @property
    def x2(self):
        return self.bbox[2]

    @property
    def y2(self):
        return self.bbox[3]

    @property
    def width(self):
        return max(
            0,
            self.x2 - self.x1
        )

    @property
    def height(self):
        return max(
            0,
            self.y2 - self.y1
        )

    @property
    def center_x(self):
        return (
            self.x1 + self.x2
        ) / 2.0

    @property
    def center_y(self):
        return (
            self.y1 + self.y2
        ) / 2.0


# ============================================================
# MAIN ANALYZER
# ============================================================

class SofaGeometryAnalyzer:

    def __init__(
        self,
        model_path=MODEL_PATH,
        prediction_confidence=PREDICTION_CONFIDENCE,
        geometry_confidence=GEOMETRY_CONFIDENCE,
    ):

        print()
        print("=" * 70)
        print("LOADING SOFA YOLO MODEL")
        print("=" * 70)

        if not os.path.isfile(model_path):

            raise FileNotFoundError(
                f"\nModel not found:\n{os.path.abspath(model_path)}"
            )

        self.model = YOLO(model_path)

        self.prediction_confidence = (
            prediction_confidence
        )

        self.geometry_confidence = (
            geometry_confidence
        )

        print(
            f"Model path: {model_path}"
        )

        print(
            f"Classes: {self.model.names}"
        )

        print(
            "Model loaded successfully."
        )

    # ========================================================
    # YOLO SEGMENTATION
    # ========================================================

    def run_inference(
        self,
        image_path
    ):

        results = self.model.predict(
            source=image_path,
            conf=self.prediction_confidence,
            retina_masks=True,
            verbose=False,
        )

        if not results:

            return None, []

        result = results[0]

        image = result.orig_img.copy()

        if result.masks is None:

            return image, []

        image_height, image_width = (
            image.shape[:2]
        )

        boxes = (
            result.boxes.xyxy
            .cpu()
            .numpy()
        )

        class_ids = (
            result.boxes.cls
            .cpu()
            .numpy()
            .astype(int)
        )

        confidences = (
            result.boxes.conf
            .cpu()
            .numpy()
        )

        polygons = result.masks.xy

        components = []

        for index in range(
            len(class_ids)
        ):

            class_id = class_ids[index]

            if (
                class_id < 0
                or class_id >= len(CLASS_NAMES)
            ):
                continue

            class_name = CLASS_NAMES[
                class_id
            ]

            confidence = float(
                confidences[index]
            )

            polygon = polygons[index]

            mask = np.zeros(
                (
                    image_height,
                    image_width,
                ),
                dtype=np.uint8,
            )

            if (
                polygon is not None
                and len(polygon) >= 3
            ):

                polygon_points = (
                    np.round(
                        polygon
                    ).astype(
                        np.int32
                    )
                )

                cv2.fillPoly(
                    mask,
                    [polygon_points],
                    255,
                )

            area = float(
                cv2.countNonZero(
                    mask
                )
            )

            x1, y1, x2, y2 = (
                boxes[index]
            )

            component = Component(
                cls_name=class_name,
                confidence=confidence,
                mask=mask,
                bbox=(
                    int(x1),
                    int(y1),
                    int(x2),
                    int(y2),
                ),
                area_px=area,
            )

            components.append(
                component
            )

        return image, components

    # ========================================================
    # MASK IOU
    # ========================================================

    @staticmethod
    def mask_iou(
        first,
        second
    ):

        mask_a = first.mask > 0
        mask_b = second.mask > 0

        intersection = np.logical_and(
            mask_a,
            mask_b,
        ).sum()

        union = np.logical_or(
            mask_a,
            mask_b,
        ).sum()

        if union == 0:

            return 0.0

        return float(
            intersection / union
        )

    # ========================================================
    # DEDUPLICATION
    # ========================================================

    def deduplicate_components(
        self,
        components,
        iou_threshold=0.60,
    ):

        grouped = {}

        for component in components:

            grouped.setdefault(
                component.cls_name,
                []
            ).append(
                component
            )

        final_components = []

        for class_name, group in (
            grouped.items()
        ):

            group = sorted(
                group,
                key=lambda item:
                    item.confidence,
                reverse=True,
            )

            kept = []

            for candidate in group:

                duplicate = False

                for existing in kept:

                    if (
                        self.mask_iou(
                            candidate,
                            existing,
                        )
                        >= iou_threshold
                    ):

                        duplicate = True

                        break

                if not duplicate:

                    kept.append(
                        candidate
                    )

            final_components.extend(
                kept
            )

        return final_components

    # ========================================================
    # GEOMETRY VALID COMPONENTS
    # ========================================================

    def get_geometry_components(
        self,
        components
    ):

        valid = []

        for component in components:

            if (
                component.confidence
                < self.geometry_confidence
            ):
                continue

            if (
                component.area_px
                < MIN_COMPONENT_AREA
            ):
                continue

            valid.append(
                component
            )

        return valid

    # ========================================================
    # LEG FILTERING
    # ========================================================

    def filter_visible_legs(
        self,
        components,
        image_height,
    ):

        raw_legs = [
            component
            for component in components
            if (
                component.cls_name
                == "legs"
            )
            and (
                component.confidence
                >= self.geometry_confidence
            )
            and (
                component.area_px
                >= MIN_COMPONENT_AREA
            )
        ]

        candidates = []

        for leg in raw_legs:

            # Legs normally occur in the lower
            # part of the sofa.
            if (
                leg.center_y
                < image_height * 0.45
            ):
                continue

            # Reject extremely wide regions
            # that are unlikely to be legs.
            if (
                leg.width
                > max(
                    leg.height * 3.0,
                    30
                )
            ):
                continue

            candidates.append(
                leg
            )

        final_legs = []

        for leg in sorted(
            candidates,
            key=lambda item:
                item.confidence,
            reverse=True,
        ):

            duplicate = False

            for existing in final_legs:

                if (
                    self.mask_iou(
                        leg,
                        existing
                    )
                    > 0.30
                ):

                    duplicate = True

                    break

            if not duplicate:

                final_legs.append(
                    leg
                )

        final_legs.sort(
            key=lambda item:
                item.center_x
        )

        return final_legs

    # ========================================================
    # ARM DETECTION
    # ========================================================

    def get_arms(
        self,
        components
    ):

        return [
            component
            for component in components
            if component.cls_name
            in (
                "left_arm",
                "right_arm",
            )
        ]

    # ========================================================
    # OVERALL SOFA GEOMETRY
    # ========================================================

    def calculate_overall_geometry(
        self,
        components,
    ):

        structural = [
            component
            for component in components
            if component.cls_name
            in (
                "back_cushion",
                "base",
                "left_arm",
                "right_arm",
                "legs",
            )
        ]

        if not structural:

            return (
                0.0,
                0.0,
                0.0,
            )

        non_leg_components = [
            component
            for component in structural
            if component.cls_name
            != "legs"
        ]

        if non_leg_components:

            top_y = min(
                component.y1
                for component
                in non_leg_components
            )

        else:

            top_y = min(
                component.y1
                for component
                in structural
            )

        left_x = min(
            component.x1
            for component
            in structural
        )

        right_x = max(
            component.x2
            for component
            in structural
        )

        bottom_y = max(
            component.y2
            for component
            in structural
        )

        width = float(
            right_x - left_x
        )

        height = float(
            bottom_y - top_y
        )

        aspect_ratio = (
            width / height
            if height > 0
            else 0.0
        )

        return (
            width,
            height,
            aspect_ratio,
        )

    # ========================================================
    # SEAT GEOMETRY FROM MASKS
    # ========================================================

    def calculate_seat_geometry(
        self,
        components,
    ):

        cushions = [
            component
            for component in components
            if (
                component.cls_name
                == "seat_cushion"
            )
        ]

        if not cushions:

            return {
                "count": 0,
                "width": 0.0,
                "height": 0.0,
                "aspect_ratio": 0.0,
                "area": 0.0,
                "mask": None,
            }

        combined_mask = np.zeros_like(
            cushions[0].mask
        )

        for cushion in cushions:

            combined_mask = cv2.bitwise_or(
                combined_mask,
                cushion.mask,
            )

        ys, xs = np.where(
            combined_mask > 0
        )

        if len(xs) == 0:

            return {
                "count": len(cushions),
                "width": 0.0,
                "height": 0.0,
                "aspect_ratio": 0.0,
                "area": 0.0,
                "mask": combined_mask,
            }

        mask_x1 = int(
            xs.min()
        )

        mask_x2 = int(
            xs.max()
        )

        mask_y1 = int(
            ys.min()
        )

        mask_y2 = int(
            ys.max()
        )

        width = float(
            mask_x2 - mask_x1 + 1
        )

        height = float(
            mask_y2 - mask_y1 + 1
        )

        aspect_ratio = (
            width / height
            if height > 0
            else 0.0
        )

        area = float(
            cv2.countNonZero(
                combined_mask
            )
        )

        return {
            "count": len(cushions),
            "width": width,
            "height": height,
            "aspect_ratio": aspect_ratio,
            "area": area,
            "mask": combined_mask,
        }

    # ========================================================
    # ARM INNER SPAN
    # ========================================================

    def calculate_inner_seating_span(
        self,
        components,
        overall_width,
    ):

        arms = self.get_arms(
            components
        )

        if len(arms) < 2:

            return (
                None,
                None,
                None,
            )

        arms_sorted = sorted(
            arms,
            key=lambda item:
                item.center_x
        )

        left_arm = arms_sorted[0]
        right_arm = arms_sorted[-1]

        left_inner_x = left_arm.x2
        right_inner_x = right_arm.x1

        inner_span = max(
            0,
            right_inner_x
            - left_inner_x
        )

        return (
            float(inner_span),
            left_arm,
            right_arm,
        )

    # ========================================================
    # SEATER CLASSIFICATION
    # ========================================================

    def classify_seater(
        self,
        components,
        overall_width,
        overall_height,
    ):

        seat_geometry = (
            self.calculate_seat_geometry(
                components
            )
        )

        seat_count = (
            seat_geometry["count"]
        )

        seat_width = (
            seat_geometry["width"]
        )

        seat_height = (
            seat_geometry["height"]
        )

        seat_aspect = (
            seat_geometry["aspect_ratio"]
        )

        # ----------------------------------------------------
        # Sofa aspect ratio
        # ----------------------------------------------------

        sofa_aspect = (
            overall_width
            / overall_height
            if overall_height > 0
            else 0.0
        )

        # ----------------------------------------------------
        # No seat cushion
        # ----------------------------------------------------

        if seat_count == 0:

            back_count = len([
                component
                for component in components
                if component.cls_name
                == "back_cushion"
            ])

            if 1 <= back_count <= 4:

                return (
                    back_count,
                    0.55,
                    "back cushion count",
                )

            return (
                0,
                0.0,
                "insufficient seat geometry",
            )

        # ----------------------------------------------------
        # Multiple separate seat cushions
        # ----------------------------------------------------

        if seat_count >= 2:

            final_count = min(
                seat_count,
                4,
            )

            confidence = 0.90

            return (
                final_count,
                confidence,
                "multiple seat cushions",
            )

        # ----------------------------------------------------
        # ONE CONTINUOUS CUSHION
        # ----------------------------------------------------
        #
        # One continuous cushion can represent
        # multiple seats.
        #
        # We therefore use:
        #
        #   seat width / overall sofa width
        #   sofa aspect ratio
        #   seat aspect ratio
        #   arm-to-arm span
        #
        # together.
        # ----------------------------------------------------

        seat_to_sofa_ratio = (
            seat_width
            / overall_width
            if overall_width > 0
            else 0.0
        )

        inner_span, left_arm, right_arm = (
            self.calculate_inner_seating_span(
                components,
                overall_width,
            )
        )

        # ----------------------------------------------------
        # If both arms exist, calculate the seat coverage
        # inside the arms.
        # ----------------------------------------------------

        inner_ratio = 0.0

        if (
            inner_span is not None
            and inner_span > 0
        ):

            inner_ratio = (
                seat_width
                / inner_span
            )

        # ----------------------------------------------------
        # Main classification
        # ----------------------------------------------------

        if seat_to_sofa_ratio < 0.42:

            count = 1

        elif seat_to_sofa_ratio < 0.62:

            count = 2

        elif seat_to_sofa_ratio < 0.80:

            # This is the important range for
            # normal 2/3 seat sofas.
            #
            # Use sofa proportions as secondary
            # evidence.

            if (
                sofa_aspect >= 2.75
                and seat_aspect >= 4.0
            ):

                count = 3

            else:

                count = 2

        else:

            # Very wide continuous cushion.

            if (
                sofa_aspect >= 2.90
                and seat_aspect >= 4.0
            ):

                count = 3

            else:

                count = 4

        # ----------------------------------------------------
        # Inner span refinement
        # ----------------------------------------------------

        if inner_span is not None:

            # If cushion fills almost all of the
            # usable span, do not automatically call
            # it a 4-seater.
            if (
                1.05
                <= inner_ratio
                <= 1.55
            ):

                if count > 3:

                    count = 3

            # Narrow usable span strongly supports
            # 2 seats.
            if (
                inner_span
                < overall_width * 0.55
            ):

                count = min(
                    count,
                    2,
                )

        # ----------------------------------------------------
        # Seat aspect ratio refinement
        # ----------------------------------------------------

        if seat_aspect < 2.30:

            count = min(
                count,
                2,
            )

        # ----------------------------------------------------
        # Very long sofa
        # ----------------------------------------------------

        if (
            sofa_aspect >= 3.30
            and seat_aspect >= 4.0
        ):

            count = max(
                count,
                3,
            )

        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------

        confidence = 0.82

        if (
            0.55
            <= seat_to_sofa_ratio
            <= 0.78
        ):

            confidence += 0.06

        if (
            seat_aspect >= 3.5
        ):

            confidence += 0.03

        if (
            inner_span is not None
        ):

            confidence += 0.03

        confidence = min(
            confidence,
            0.95,
        )

        return (
            count,
            confidence,
            "continuous seat-cushion segmentation geometry",
        )

    # ========================================================
    # REPORT CREATION
    # ========================================================

    def create_report(
        self,
        components,
        visible_legs,
        overall_width,
        overall_height,
        overall_aspect,
        seat_geometry,
        sofa_type,
        seat_count,
        confidence,
        method,
    ):

        seat_width = (
            seat_geometry["width"]
        )

        seat_height = (
            seat_geometry["height"]
        )

        seat_aspect = (
            seat_geometry["aspect_ratio"]
        )

        seat_to_sofa_ratio = (
            seat_width / overall_width
            if overall_width > 0
            else 0.0
        )

        arms = self.get_arms(
            components
        )

        report_components = []

        for component in components:

            report_components.append(
                {
                    "class": component.cls_name,
                    "confidence": round(
                        component.confidence,
                        4,
                    ),
                    "area_px": round(
                        component.area_px,
                        2,
                    ),
                    "bbox": [
                        component.x1,
                        component.y1,
                        component.x2,
                        component.y2,
                    ],
                }
            )

        return {
            "sofa_type": sofa_type,
            "seat_count": seat_count,
            "confidence": round(
                confidence,
                4,
            ),
            "classification_method": method,

            "seat_cushion_count": (
                seat_geometry["count"]
            ),

            "visible_leg_count": len(
                visible_legs
            ),

            "overall_width_px": round(
                overall_width,
                2,
            ),

            "overall_height_px": round(
                overall_height,
                2,
            ),

            "overall_aspect_ratio": round(
                overall_aspect,
                4,
            ),

            "seat_width_px": round(
                seat_width,
                2,
            ),

            "seat_height_px": round(
                seat_height,
                2,
            ),

            "seat_aspect_ratio": round(
                seat_aspect,
                4,
            ),

            "seat_to_sofa_width_ratio": round(
                seat_to_sofa_ratio,
                4,
            ),

            "arm_count": len(
                arms
            ),

            "components": report_components,
        }


    # ========================================================
    # FINAL COMBINED VISUALIZATION
    # ========================================================

    def create_final_visualization(
        self,
        original_image,
        all_components,
        geometry_components,
        report,
        output_path,
    ):
        """
        Create ONE final PNG containing:

        - Transparent background outside the sofa
        - Sofa segmentation
        - Geometry-valid component outlines
        - Original YOLO class names
        - Component confidence labels
        - Sofa type
        - Sofa confidence

        No bounding boxes.
        No physical left/right reassignment.
        """

        height, width = original_image.shape[:2]

        # ----------------------------------------------------
        # 1. Build foreground mask for background removal.
        # ----------------------------------------------------
        foreground = np.zeros(
            (height, width),
            dtype=np.uint8,
        )

        for component in all_components:
            if component.confidence < self.prediction_confidence:
                continue

            if component.mask is None:
                continue

            foreground = cv2.bitwise_or(
                foreground,
                component.mask,
            )

        # Clean small holes/noise without destroying thin sofa parts.
        kernel = np.ones((5, 5), np.uint8)

        foreground = cv2.morphologyEx(
            foreground,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=1,
        )

        foreground = cv2.morphologyEx(
            foreground,
            cv2.MORPH_OPEN,
            kernel,
            iterations=1,
        )

        # ----------------------------------------------------
        # 2. Create transparent RGBA image.
        # ----------------------------------------------------
        image = cv2.cvtColor(
            original_image,
            cv2.COLOR_BGR2BGRA,
        )

        # Everything outside detected sofa masks becomes
        # transparent.
        image[:, :, 3] = foreground

        # ----------------------------------------------------
        # 3. Draw geometry-valid segmentation outlines.
        # ----------------------------------------------------
        for component in geometry_components:
            if component.confidence < self.geometry_confidence:
                continue

            if component.mask is None:
                continue

            contours, _ = cv2.findContours(
                component.mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            # White contour.
            cv2.drawContours(
                image,
                contours,
                -1,
                (255, 255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # ----------------------------------------------------
        # 4. Top information panel.
        #
        # The panel is intentionally opaque so the sofa type
        # and confidence remain visible even after background
        # removal.
        # ----------------------------------------------------
        panel_height = 85

        panel = image.copy()

        cv2.rectangle(
            panel,
            (0, 0),
            (width, panel_height),
            (0, 0, 0, 220),
            -1,
        )

        image = cv2.addWeighted(
            panel,
            0.72,
            image,
            0.28,
            0,
        )

        # Make the top panel fully visible.
        image[:panel_height, :, 3] = 255

        font = cv2.FONT_HERSHEY_SIMPLEX

        sofa_text = (
            f"SOFA TYPE: {report['sofa_type']}"
        )

        confidence_text = (
            f"CONFIDENCE: "
            f"{report['confidence'] * 100:.1f}%"
        )

        cv2.putText(
            image,
            sofa_text,
            (12, 32),
            font,
            0.75,
            (255, 255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            image,
            confidence_text,
            (12, 66),
            font,
            0.62,
            (255, 255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # ----------------------------------------------------
        # 5. Component labels.
        #
        # IMPORTANT:
        # Original YOLO names are preserved exactly.
        #
        # No:
        #   PHYSICAL LEFT ARM
        #   PHYSICAL RIGHT ARM
        #
        # No bounding boxes are drawn.
        # ----------------------------------------------------
        used_positions = []

        for component in sorted(
            geometry_components,
            key=lambda item: (item.y1, item.x1),
        ):
            if component.confidence < self.geometry_confidence:
                continue

            label = (
                f"{component.cls_name} "
                f"{component.confidence:.2f}"
            )

            text_size, baseline = cv2.getTextSize(
                label,
                font,
                0.45,
                1,
            )

            text_width = text_size[0]
            text_height = text_size[1]

            x = max(
                5,
                min(
                    component.x1,
                    width - text_width - 12,
                ),
            )

            y = max(
                panel_height + text_height + 8,
                component.y1,
            )

            # Try to avoid stacking labels on exactly the same
            # position.
            original_y = y

            while any(
                abs(y - previous_y) < 22
                and abs(x - previous_x) < 160
                for previous_x, previous_y in used_positions
            ):
                y += 22

                if y >= height - 5:
                    y = original_y
                    break

            used_positions.append((x, y))

            # Black label background.
            cv2.rectangle(
                image,
                (
                    int(x),
                    int(y - text_height - 5),
                ),
                (
                    int(
                        min(
                            width - 1,
                            x + text_width + 8,
                        )
                    ),
                    int(
                        min(
                            height - 1,
                            y + baseline + 2,
                        )
                    ),
                ),
                (0, 0, 0, 230),
                -1,
            )

            # White text.
            cv2.putText(
                image,
                label,
                (
                    int(x + 4),
                    int(y),
                ),
                font,
                0.45,
                (255, 255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # ----------------------------------------------------
        # 6. Keep labels/panel visible.
        #
        # Labels are placed on/near detected sofa regions.
        # The label background is made opaque so the text does
        # not disappear when the image is viewed over a white
        # or checkerboard transparency background.
        # ----------------------------------------------------

        # ----------------------------------------------------
        # 7. Save ONE final PNG.
        # ----------------------------------------------------
        os.makedirs(
            os.path.dirname(output_path),
            exist_ok=True,
        )

        success = cv2.imwrite(
            output_path,
            image,
        )

        if not success:
            raise IOError(
                f"Could not save final visualization:\n"
                f"{output_path}"
            )

        print(
            f"\nCombined final image saved to:\n"
            f"{os.path.abspath(output_path)}"
        )

        return output_path


    # ========================================================
    # TEXT REPORT
    # ========================================================

    def create_text_report(
        self,
        report,
        components,
        visible_legs,
    ):

        lines = []

        lines.append(
            "=" * 70
        )

        lines.append(
            "SOFA GEOMETRY ANALYSIS"
        )

        lines.append(
            "=" * 70
        )

        lines.append("")

        lines.append(
            f"SOFA TYPE: {report['sofa_type']}"
        )

        lines.append(
            f"CONFIDENCE: "
            f"{report['confidence'] * 100:.1f}%"
        )

        lines.append("")

        lines.append(
            "COMPONENT DETECTIONS"
        )

        lines.append(
            "-" * 70
        )

        for component in components:

            lines.append(
                f"{component.cls_name:15s} "
                f"confidence="
                f"{component.confidence:.3f} "
                f"area="
                f"{component.area_px:.1f}px2"
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
            f"{report['overall_width_px']:.2f}px"
        )

        lines.append(
            f"Overall height: "
            f"{report['overall_height_px']:.2f}px"
        )

        lines.append(
            f"Overall aspect ratio: "
            f"{report['overall_aspect_ratio']:.4f}"
        )

        lines.append("")

        lines.append(
            f"Seat cushions: "
            f"{report['seat_cushion_count']}"
        )

        lines.append(
            f"Visible legs: "
            f"{report['visible_leg_count']}"
        )

        lines.append(
            f"Seat width: "
            f"{report['seat_width_px']:.2f}px"
        )

        lines.append(
            f"Seat height: "
            f"{report['seat_height_px']:.2f}px"
        )

        lines.append(
            f"Seat aspect ratio: "
            f"{report['seat_aspect_ratio']:.4f}"
        )

        lines.append(
            f"Seat / sofa width ratio: "
            f"{report['seat_to_sofa_width_ratio']:.4f}"
        )

        lines.append("")

        lines.append(
            f"Classification method: "
            f"{report['classification_method']}"
        )

        lines.append("")

        lines.append(
            "=" * 70
        )

        return "\n".join(
            lines
        )

    # ========================================================
    # COMPLETE ANALYSIS
    # ========================================================

    def analyze(
        self,
        image_path,
    ):

        print()
        print("=" * 70)
        print("SOFA GEOMETRY ANALYSIS")
        print("=" * 70)

        print(
            f"Input image:\n{image_path}"
        )

        # ----------------------------------------------------
        # Create output directory
        # ----------------------------------------------------

        os.makedirs(
            OUTPUT_DIR,
            exist_ok=True,
        )

        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        image = cv2.imread(
            image_path
        )

        if image is None:

            raise ValueError(
                f"Could not read image:\n"
                f"{image_path}"
            )

        image_height, image_width = (
            image.shape[:2]
        )

        print(
            f"\nImage width: "
            f"{image_width} px"
        )

        print(
            f"Image height: "
            f"{image_height} px"
        )

        # ----------------------------------------------------
        # YOLO
        # ----------------------------------------------------

        print()
        print(
            "[1/4] Running YOLO segmentation..."
        )

        original_image, raw_components = (
            self.run_inference(
                image_path
            )
        )

        if original_image is None:

            raise RuntimeError(
                "YOLO did not return an image."
            )

        print(
            f"Total detections: "
            f"{len(raw_components)}"
        )

        print(
            f"Prediction confidence threshold: "
            f"{self.prediction_confidence}"
        )

        print(
            f"Geometry confidence threshold: "
            f"{self.geometry_confidence}"
        )

        print()

        print(
            "ALL YOLO PREDICTIONS:"
        )

        for index, component in enumerate(
            raw_components
        ):

            print(
                f"Prediction {index}: "
                f"{component.cls_name} | "
                f"confidence="
                f"{component.confidence:.3f}"
            )

        # ----------------------------------------------------
        # Deduplicate
        # ----------------------------------------------------

        print()
        print(
            "[2/4] De-duplicating masks..."
        )

        components = (
            self.deduplicate_components(
                raw_components
            )
        )

        # ----------------------------------------------------
        # Geometry-valid components
        # ----------------------------------------------------

        geometry_components = (
            self.get_geometry_components(
                components
            )
        )

        print(
            f"Geometry-valid components: "
            f"{len(geometry_components)}"
        )

        for component in (
            geometry_components
        ):

            print(
                f"  {component.cls_name} "
                f"{component.confidence:.3f}"
            )

        # ----------------------------------------------------
        # Visible legs
        # ----------------------------------------------------

        visible_legs = (
            self.filter_visible_legs(
                geometry_components,
                image_height,
            )
        )

        # Remove raw legs from geometry list
        # and replace them with cleaned legs.
        non_leg_components = [
            component
            for component
            in geometry_components
            if component.cls_name
            != "legs"
        ]

        cleaned_components = (
            non_leg_components
            + visible_legs
        )

        # ----------------------------------------------------
        # Overall geometry
        # ----------------------------------------------------

        (
            overall_width,
            overall_height,
            overall_aspect,
        ) = self.calculate_overall_geometry(
            cleaned_components
        )

        # ----------------------------------------------------
        # Seat geometry
        # ----------------------------------------------------

        seat_geometry = (
            self.calculate_seat_geometry(
                cleaned_components
            )
        )

        # ----------------------------------------------------
        # Seater classification
        # ----------------------------------------------------

        print()
        print(
            "[3/4] Determining seater type..."
        )

        (
            seat_count,
            base_confidence,
            classification_method,
        ) = self.classify_seater(
            cleaned_components,
            overall_width,
            overall_height,
        )

        if seat_count == 1:
            sofa_type = "1-seater"

        elif seat_count == 2:
            sofa_type = "2-seater"

        elif seat_count == 3:
            sofa_type = "3-seater"

        elif seat_count == 4:
            sofa_type = "4-seater"

        else:
            sofa_type = "unknown"

        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------

        confidence = (
            base_confidence
        )

        arms = self.get_arms(
            cleaned_components
        )

        backs = [
            component
            for component
            in cleaned_components
            if component.cls_name
            == "back_cushion"
        ]

        if len(arms) >= 2:

            confidence += 0.03

        if len(backs) >= 1:

            confidence += 0.02

        if (
            seat_geometry["count"]
            >= 1
        ):

            confidence += 0.03

        confidence = min(
            confidence,
            0.99,
        )

        # ----------------------------------------------------
        # Create report
        # ----------------------------------------------------

        report = self.create_report(
            cleaned_components,
            visible_legs,
            overall_width,
            overall_height,
            overall_aspect,
            seat_geometry,
            sofa_type,
            seat_count,
            confidence,
            classification_method,
        )

        # ----------------------------------------------------
        # Save JSON
        # ----------------------------------------------------

        json_path = os.path.join(
            OUTPUT_DIR,
            "geometry_report.json",
        )

        with open(
            json_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                report,
                file,
                indent=4,
            )

        # ----------------------------------------------------
        # Save TXT
        # ----------------------------------------------------

        text_report = (
            self.create_text_report(
                report,
                cleaned_components,
                visible_legs,
            )
        )

        text_path = os.path.join(
            OUTPUT_DIR,
            "geometry_result.txt",
        )

        with open(
            text_path,
            "w",
            encoding="utf-8",
        ) as file:

            file.write(
                text_report
            )

        # ----------------------------------------------------
        # ONE COMBINED FINAL VISUALIZATION
        # ----------------------------------------------------

        final_visualization_path = os.path.join(
            OUTPUT_DIR,
            "sofa_geometry_result.png",
        )

        print()
        print(
            "[4/4] Creating combined background-removal "
            "and geometry result..."
        )

        self.create_final_visualization(
            original_image=original_image,
            all_components=components,
            geometry_components=cleaned_components,
            report=report,
            output_path=final_visualization_path,
        )

        # ----------------------------------------------------
        # Console result
        # ----------------------------------------------------

        print()
        print("=" * 70)
        print("FINAL RESULT")
        print("=" * 70)

        print(
            f"SOFA TYPE: "
            f"{sofa_type}"
        )

        print(
            f"CONFIDENCE: "
            f"{confidence * 100:.1f}%"
        )

        print()
        print(
            f"Seat cushions: "
            f"{seat_geometry['count']}"
        )

        print(
            f"Visible legs: "
            f"{len(visible_legs)}"
        )

        print()
        print(
            "OUTPUT FILES:"
        )

        print(
            f"Combined visualization:\n"
            f"{os.path.abspath(final_visualization_path)}"
        )

        print(
            f"\nText report:\n"
            f"{os.path.abspath(text_path)}"
        )

        print(
            f"\nJSON report:\n"
            f"{os.path.abspath(json_path)}"
        )

        print()
        print("=" * 70)

        return report


# ============================================================
# IMAGE PATH PROMPT
# ============================================================

def prompt_for_image_path():

    while True:

        print()

        image_path = input(
            "Enter path to sofa image: "
        ).strip()

        image_path = (
            image_path
            .strip('"')
            .strip("'")
        )

        if not image_path:

            print(
                "Please enter an image path."
            )

            continue

        if not os.path.isfile(
            image_path
        ):

            print(
                f"\nFile not found:\n"
                f"{image_path}"
            )

            continue

        return image_path


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Sofa YOLO segmentation "
            "and geometry analysis"
        )
    )

    parser.add_argument(
        "--image",
        default=None,
        help=(
            "Path to sofa image. "
            "If omitted, the program "
            "asks interactively."
        ),
    )

    parser.add_argument(
        "--weights",
        default=MODEL_PATH,
        help=(
            "Path to YOLO segmentation "
            "weights."
        ),
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=PREDICTION_CONFIDENCE,
        help=(
            "YOLO prediction confidence."
        ),
    )

    parser.add_argument(
        "--geometry-conf",
        type=float,
        default=GEOMETRY_CONFIDENCE,
        help=(
            "Minimum confidence for "
            "geometry analysis."
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Get image
    # --------------------------------------------------------

    if args.image:

        image_path = (
            args.image
            .strip('"')
            .strip("'")
        )

    else:

        image_path = (
            prompt_for_image_path()
        )

    if not os.path.isfile(
        image_path
    ):

        print(
            f"\nERROR: Image not found:\n"
            f"{image_path}",
            file=sys.stderr,
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Run analyzer
    # --------------------------------------------------------

    try:

        analyzer = (
            SofaGeometryAnalyzer(
                model_path=args.weights,
                prediction_confidence=args.conf,
                geometry_confidence=(
                    args.geometry_conf
                ),
            )
        )

        analyzer.analyze(
            image_path
        )

    except Exception as error:

        print()
        print(
            "=" * 70
        )

        print(
            "ERROR"
        )

        print(
            "=" * 70
        )

        print(
            str(error)
        )

        print()

        raise


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()