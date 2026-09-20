from ultralytics import YOLO
import cv2
import numpy as np
import os
import math


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = r".\runs\segment\models\sofa_yolo11n_seg_v1-2\weights\best.pt"

# ============================================================
# ASK USER FOR IMAGE
# ============================================================

IMAGE_PATH = input(
    "\nEnter the full path of the sofa image: "
).strip().strip('"')

if not IMAGE_PATH:
    raise ValueError(
        "No image path was provided."
    )

if not os.path.isfile(IMAGE_PATH):
    raise FileNotFoundError(
        f"\nImage file not found:\n{IMAGE_PATH}"
    )

print(
    f"\nInput image:\n{IMAGE_PATH}"
)
OUTPUT_DIR = r".\runs\geometry"

OUTPUT_IMAGE = os.path.join(
    OUTPUT_DIR,
    "geometry_result.jpg"
)

OUTPUT_REPORT = os.path.join(
    OUTPUT_DIR,
    "geometry_result.txt"
)

PREDICTION_CONFIDENCE = 0.25

GEOMETRY_CONFIDENCE = 0.50

MIN_CONTOUR_AREA = 100


# ============================================================
# SUPPORTED CLASSES
# ============================================================

EXPECTED_CLASSES = {
    0: "back_cushion",
    1: "base",
    2: "left_arm",
    3: "legs",
    4: "right_arm",
    5: "seat_cushion",
}


# ============================================================
# SEATER TYPES
# ============================================================

SEATER_TYPES = [
    "1-seater",
    "2-seater",
    "3-seater",
    "4-seater",
    "5+ seater",
    "sectional",
    "unknown",
]


# ============================================================
# REPORT LOGGER
# ============================================================

report_lines = []


def log(text=""):
    print(text)
    report_lines.append(str(text))


# ============================================================
# BASIC HELPERS
# ============================================================

def safe_ratio(a, b):
    if b is None or abs(b) < 1e-9:
        return 0.0

    return float(a) / float(b)


def clamp(value, minimum=0.0, maximum=1.0):
    return max(minimum, min(maximum, value))


def format_value(value, decimals=4):
    if value is None:
        return "N/A"

    if isinstance(value, float):
        return f"{value:.{decimals}f}"

    return str(value)


def center_distance(c1, c2):
    if c1 is None or c2 is None:
        return None

    return math.sqrt(
        (c1[0] - c2[0]) ** 2 +
        (c1[1] - c2[1]) ** 2
    )


def get_largest_component(components):
    if not components:
        return None

    return max(
        components,
        key=lambda x: x["area"]
    )


# ============================================================
# DRAWING
# ============================================================

CLASS_COLORS = {
    "back_cushion": (255, 0, 0),
    "base": (0, 255, 0),
    "left_arm": (0, 0, 255),
    "right_arm": (255, 0, 255),
    "seat_cushion": (0, 255, 255),
    "legs": (255, 255, 0),
}


def draw_text_with_background(
    image,
    text,
    position,
    font_scale=0.5,
    thickness=1,
    color=(255, 255, 255),
):
    font = cv2.FONT_HERSHEY_SIMPLEX

    (tw, th), baseline = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness
    )

    x, y = position

    cv2.rectangle(
        image,
        (x - 3, y - th - baseline - 3),
        (x + tw + 3, y + 3),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        image,
        text,
        (x, y),
        font,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA
    )


def draw_component_label(
    image,
    component,
    color
):
    x = component["bbox"][0]
    y = component["bbox"][1]

    label = (
        f'{component["class_name"]} '
        f'{component["confidence"]:.2f}'
    )

    draw_text_with_background(
        image,
        label,
        (x, max(20, y - 5)),
        font_scale=0.45,
        thickness=1,
        color=color
    )


# ============================================================
# LOAD MODEL
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"YOLO model not found:\n{MODEL_PATH}"
    )


if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(
        f"Input image not found:\n{IMAGE_PATH}"
    )


model = YOLO(MODEL_PATH)

log("Model loaded successfully.")
log(f"Model path: {MODEL_PATH}")
log(f"Classes: {EXPECTED_CLASSES}")


# ============================================================
# LOAD IMAGE
# ============================================================

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise RuntimeError(
        f"Could not read image:\n{IMAGE_PATH}"
    )


image_height, image_width = image.shape[:2]

log()
log(f"Image width: {image_width} px")
log(f"Image height: {image_height} px")


# ============================================================
# YOLO PREDICTION
# ============================================================

results = model.predict(
    source=image,
    conf=PREDICTION_CONFIDENCE,
    verbose=False
)

if not results:
    raise RuntimeError(
        "YOLO returned no results."
    )


result = results[0]

names = result.names


# ============================================================
# COLLECT ALL PREDICTIONS
# ============================================================

all_predictions = []

if result.boxes is not None:

    for i in range(len(result.boxes)):

        cls_id = int(
            result.boxes.cls[i].item()
        )

        confidence = float(
            result.boxes.conf[i].item()
        )

        class_name = names.get(
            cls_id,
            EXPECTED_CLASSES.get(
                cls_id,
                f"class_{cls_id}"
            )
        )

        all_predictions.append({
            "index": i,
            "class_id": cls_id,
            "class_name": class_name,
            "confidence": confidence,
        })


log()
log(f"Total detections: {len(all_predictions)}")

log(
    f"Prediction confidence threshold: "
    f"{PREDICTION_CONFIDENCE:.2f}"
)

log(
    f"Geometry confidence threshold: "
    f"{GEOMETRY_CONFIDENCE:.2f}"
)

log(
    f"Minimum component area: "
    f"{MIN_CONTOUR_AREA} px2"
)


# ============================================================
# PRINT ALL YOLO PREDICTIONS
# ============================================================

log()
log("ALL YOLO PREDICTIONS:")

for prediction in all_predictions:

    log(
        f'Prediction {prediction["index"]}: '
        f'{prediction["class_name"]} | '
        f'confidence={prediction["confidence"]:.3f}'
    )


# ============================================================
# EXTRACT MASK GEOMETRY
# ============================================================
# ============================================================
# EXTRACT YOLO SEGMENTATION CONTOURS
# ============================================================

components = []

if result.masks is None:
    log("No segmentation masks detected.")

else:

    # IMPORTANT:
    # result.masks.xy contains polygon points already mapped
    # to the ORIGINAL IMAGE coordinates.
    polygons = result.masks.xy

    for i, polygon in enumerate(polygons):

        if i >= len(all_predictions):
            break

        prediction = all_predictions[i]

        confidence = prediction["confidence"]
        class_name = prediction["class_name"]

        # ----------------------------------------------------
        # Convert YOLO polygon to OpenCV contour
        # ----------------------------------------------------

        if polygon is None or len(polygon) < 3:
            continue

        contour = np.asarray(
            polygon,
            dtype=np.float32
        )

        contour = contour.reshape(
            (-1, 1, 2)
        )

        # OpenCV contour for drawing
        contour_int = np.round(
            contour
        ).astype(np.int32)

        # ----------------------------------------------------
        # Calculate geometry directly from ORIGINAL
        # YOLO polygon coordinates
        # ----------------------------------------------------

        area = cv2.contourArea(
            contour_int
        )

        if area < MIN_CONTOUR_AREA:
            continue

        perimeter = cv2.arcLength(
            contour_int,
            True
        )

        x, y, w, h = cv2.boundingRect(
            contour_int
        )

        # ----------------------------------------------------
        # Center
        # ----------------------------------------------------

        moments = cv2.moments(
            contour_int
        )

        if moments["m00"] != 0:

            center_x = (
                moments["m10"] /
                moments["m00"]
            )

            center_y = (
                moments["m01"] /
                moments["m00"]
            )

        else:

            center_x = x + w / 2
            center_y = y + h / 2

        # ----------------------------------------------------
        # Aspect ratio
        # ----------------------------------------------------

        aspect_ratio = (
            float(w) / float(h)
            if h > 0
            else 0.0
        )

        # ----------------------------------------------------
        # Store component
        # ----------------------------------------------------

        components.append({

            "index": i,

            "class_id":
                prediction["class_id"],

            "class_name":
                class_name,

            "confidence":
                confidence,

            "contour":
                contour_int,

            "area":
                float(area),

            "perimeter":
                float(perimeter),

            "bbox": (
                int(x),
                int(y),
                int(w),
                int(h)
            ),

            "center": (
                float(center_x),
                float(center_y)
            ),

            "width":
                int(w),

            "height":
                int(h),

            "aspect_ratio":
                float(aspect_ratio),
        })


# ============================================================
# GEOMETRY FILTER
# ============================================================

geometry_components = [
    c
    for c in components
    if c["confidence"] >= GEOMETRY_CONFIDENCE
]


log()
log("GEOMETRY-VALID COMPONENTS:")

for component in geometry_components:

    log(
        f'{component["class_name"]} '
        f'{component["confidence"]:.3f} '
        f'area={component["area"]:.1f}'
    )

# ============================================================
# GEOMETRY FILTER
# ============================================================

geometry_components = [
    c
    for c in components
    if c["confidence"] >= GEOMETRY_CONFIDENCE
]


log()
log("GEOMETRY-VALID COMPONENTS:")

for component in geometry_components:

    log(
        f'{component["class_name"]} '
        f'{component["confidence"]:.3f}'
    )


# ============================================================
# COMPONENT SUMMARY
# ============================================================

log()
log("COMPONENT GEOMETRY:")

for component in components:

    x, y, w, h = component["bbox"]

    cx, cy = component["center"]

    log(
        f'{component["class_name"]} | '
        f'confidence={component["confidence"]:.3f} | '
        f'area={component["area"]:.1f} px2 | '
        f'bbox=({x},{y},{w},{h}) | '
        f'AR={component["aspect_ratio"]:.2f} | '
        f'perimeter={component["perimeter"]:.1f} px | '
        f'center=({cx:.1f},{cy:.1f})'
    )


# ============================================================
# CLASS GROUPS
# ============================================================

def class_components(class_name):

    return [
        c
        for c in geometry_components
        if c["class_name"] == class_name
    ]


back_components = class_components(
    "back_cushion"
)

base_components = class_components(
    "base"
)

arm_components = (
    class_components("left_arm") +
    class_components("right_arm")
)

seat_components = class_components(
    "seat_cushion"
)

leg_components = class_components(
    "legs"
)


# ============================================================
# PHYSICAL ARM ASSIGNMENT
# ============================================================

physical_left_arm = None
physical_right_arm = None

if len(arm_components) >= 2:

    sorted_arms = sorted(
        arm_components,
        key=lambda c: c["center"][0]
    )

    physical_left_arm = sorted_arms[0]

    physical_right_arm = sorted_arms[-1]


elif len(arm_components) == 1:

    only_arm = arm_components[0]

    if only_arm["center"][0] < image_width / 2:

        physical_left_arm = only_arm

    else:

        physical_right_arm = only_arm


log()
log("PHYSICAL ARM ASSIGNMENT:")

if physical_left_arm:

    log("Physical LEFT arm:")

    log(
        f'  YOLO class = '
        f'{physical_left_arm["class_name"]}'
    )

    log(
        f'  confidence = '
        f'{physical_left_arm["confidence"]:.3f}'
    )

    log(
        f'  center = '
        f'({physical_left_arm["center"][0]:.1f}, '
        f'{physical_left_arm["center"][1]:.1f})'
    )

else:

    log("Physical LEFT arm: NOT DETECTED")


if physical_right_arm:

    log("Physical RIGHT arm:")

    log(
        f'  YOLO class = '
        f'{physical_right_arm["class_name"]}'
    )

    log(
        f'  confidence = '
        f'{physical_right_arm["confidence"]:.3f}'
    )

    log(
        f'  center = '
        f'({physical_right_arm["center"][0]:.1f}, '
        f'{physical_right_arm["center"][1]:.1f})'
    )

else:

    log("Physical RIGHT arm: NOT DETECTED")


# ============================================================
# OVERALL SOFA BOUNDING BOX
# ============================================================

overall_x1 = None
overall_y1 = None
overall_x2 = None
overall_y2 = None

for component in geometry_components:

    x, y, w, h = component["bbox"]

    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h

    if overall_x1 is None:

        overall_x1 = x1
        overall_y1 = y1
        overall_x2 = x2
        overall_y2 = y2

    else:

        overall_x1 = min(
            overall_x1,
            x1
        )

        overall_y1 = min(
            overall_y1,
            y1
        )

        overall_x2 = max(
            overall_x2,
            x2
        )

        overall_y2 = max(
            overall_y2,
            y2
        )


overall_width = 0
overall_height = 0
overall_aspect_ratio = 0


if overall_x1 is not None:

    overall_width = (
        overall_x2 -
        overall_x1
    )

    overall_height = (
        overall_y2 -
        overall_y1
    )

    overall_aspect_ratio = safe_ratio(
        overall_width,
        overall_height
    )


log()
log("OVERALL SOFA GEOMETRY:")

log(
    f"Overall width: "
    f"{overall_width:.4f} px"
)

log(
    f"Overall height: "
    f"{overall_height:.4f} px"
)

log(
    f"Overall aspect ratio: "
    f"{overall_aspect_ratio:.4f}"
)


# ============================================================
# COMPONENT COUNTS
# ============================================================

component_counts = {}

for class_name in EXPECTED_CLASSES.values():

    component_counts[class_name] = len(
        class_components(class_name)
    )


log()
log("COMPONENT STRUCTURE:")

for class_name, count in component_counts.items():

    log(
        f"{class_name}: {count}"
    )


# ============================================================
# SEAT GEOMETRY
# ============================================================

seat = get_largest_component(
    seat_components
)

seat_confidence = 0.0
seat_width = 0
seat_height = 0
seat_area = 0
seat_aspect_ratio = 0
seat_bbox_width_ratio = 0


if seat is not None:

    seat_confidence = seat["confidence"]

    seat_width = seat["width"]

    seat_height = seat["height"]

    seat_area = seat["area"]

    seat_aspect_ratio = seat["aspect_ratio"]

    seat_bbox_width_ratio = safe_ratio(
        seat_width,
        overall_width
    )


log()
log("SEAT MASK QUALITY:")

if seat is not None:

    log(
        f"Seat confidence: "
        f"{seat_confidence:.3f}"
    )

    log(
        f"Seat width: "
        f"{seat_width} px"
    )

    log(
        f"Seat height: "
        f"{seat_height} px"
    )

    log(
        f"Seat area: "
        f"{seat_area:.1f} px2"
    )

    log(
        f"Seat aspect ratio: "
        f"{seat_aspect_ratio:.2f}"
    )

else:

    log("Seat cushion not detected.")


# ============================================================
# ARM SPAN
# ============================================================

arm_center_span = 0.0

arm_inner_span = 0.0

left_inner_edge = None

right_inner_edge = None


if physical_left_arm is not None:

    lx, ly, lw, lh = (
        physical_left_arm["bbox"]
    )

    left_inner_edge = lx + lw


if physical_right_arm is not None:

    rx, ry, rw, rh = (
        physical_right_arm["bbox"]
    )

    right_inner_edge = rx


if (
    physical_left_arm is not None
    and
    physical_right_arm is not None
):

    arm_center_span = center_distance(
        physical_left_arm["center"],
        physical_right_arm["center"]
    )

    if (
        right_inner_edge is not None
        and
        left_inner_edge is not None
        and
        right_inner_edge > left_inner_edge
    ):

        arm_inner_span = (
            right_inner_edge -
            left_inner_edge
        )


log()
log("SOFA SPAN GEOMETRY:")

log(
    f"Arm center-to-center span: "
    f"{arm_center_span:.2f}px"
)

log(
    f"Physical left arm inner edge: "
    f"{left_inner_edge}"
)

log(
    f"Physical right arm inner edge: "
    f"{right_inner_edge}"
)

log(
    f"Usable arm-to-arm inner span: "
    f"{arm_inner_span:.2f} px"
)


# ============================================================
# USABLE SEAT SPAN
# ============================================================

usable_seat_span = 0.0

seat_bbox_to_arm_inner_span_ratio = 0.0


if seat is not None:

    sx, sy, sw, sh = seat["bbox"]

    seat_left = sx

    seat_right = sx + sw

    if (
        left_inner_edge is not None
        and
        right_inner_edge is not None
    ):

        usable_left = max(
            seat_left,
            left_inner_edge
        )

        usable_right = min(
            seat_right,
            right_inner_edge
        )

        if usable_right > usable_left:

            usable_seat_span = (
                usable_right -
                usable_left
            )

        seat_bbox_to_arm_inner_span_ratio = safe_ratio(
            seat_width,
            arm_inner_span
        )


log()
log("USABLE SEAT SPAN:")

log(
    f"Detected seat bounding-box span: "
    f"{seat_width:.2f} px"
)

log(
    f"Usable seat span between arm inner edges: "
    f"{usable_seat_span:.2f} px"
)

log(
    f"Seat bbox / arm inner span ratio: "
    f"{seat_bbox_to_arm_inner_span_ratio:.4f}"
)


# ============================================================
# SEAT/BACK RELATIONSHIP
# ============================================================

back = get_largest_component(
    back_components
)

seat_back_area_ratio = 0.0

if (
    seat is not None
    and
    back is not None
):

    seat_back_area_ratio = safe_ratio(
        seat["area"],
        back["area"]
    )


# ============================================================
# ARM AREA RELATIONSHIP
# ============================================================

left_right_arm_area_ratio = 0.0


if (
    physical_left_arm is not None
    and
    physical_right_arm is not None
):

    left_right_arm_area_ratio = safe_ratio(
        min(
            physical_left_arm["area"],
            physical_right_arm["area"]
        ),
        max(
            physical_left_arm["area"],
            physical_right_arm["area"]
        )
    )


# ============================================================
# SEAT CONFIGURATION
# ============================================================

seat_count = len(
    seat_components
)

log()
log("SEAT CONFIGURATION:")

log(
    f"{seat_count} seat cushion "
    f"component(s) detected."
)


# ============================================================
# STRUCTURAL QUALITY
# ============================================================

def calculate_structural_quality():

    scores = []

    # Seat
    if seat is not None:

        scores.append(
            clamp(
                seat_confidence
            )
        )

    # Arms
    if (
        physical_left_arm is not None
        and
        physical_right_arm is not None
    ):

        arm_conf = (
            physical_left_arm["confidence"] +
            physical_right_arm["confidence"]
        ) / 2

        scores.append(
            clamp(arm_conf)
        )

    # Back
    if back is not None:

        scores.append(
            clamp(
                back["confidence"]
            )
        )

    # Overall geometry
    if overall_width > 0:

        scores.append(
            1.0
        )

    if not scores:

        return 0.0

    return float(
        np.mean(scores)
    )


structural_quality = (
    calculate_structural_quality()
)


# ============================================================
# SECTIONAL DETECTION
# ============================================================

def detect_sectional():

    score = 0.0

    reasons = []

    # Multiple seat components
    if seat_count >= 2:

        score += 0.30

        reasons.append(
            "multiple seat components"
        )

    # Multiple arm-like structures
    if len(arm_components) >= 3:

        score += 0.30

        reasons.append(
            "multiple arm structures"
        )

    # Very wide sofa
    if overall_aspect_ratio >= 3.8:

        score += 0.15

        reasons.append(
            "very wide overall geometry"
        )

    # Large vertical footprint
    if overall_height > 0:

        vertical_ratio = safe_ratio(
            overall_height,
            overall_width
        )

        if vertical_ratio > 0.42:

            score += 0.15

            reasons.append(
                "large vertical footprint"
            )

    # Seat/back structural complexity
    if (
        seat_count >= 2
        and
        len(back_components) >= 2
    ):

        score += 0.20

        reasons.append(
            "multiple seat/back sections"
        )

    return clamp(score), reasons


sectional_score, sectional_reasons = (
    detect_sectional()
)


# ============================================================
# PERSON CAPACITY ESTIMATION
# ============================================================

def estimate_person_capacity():

    """
    Estimate seating capacity from normalized geometry.

    IMPORTANT:
    This is a heuristic fallback.

    It should eventually be calibrated using
    your labeled sofa dataset.
    """

    if arm_inner_span <= 0:

        return None, 0.0

    if seat is None:

        return None, 0.0

    # --------------------------------------------------------
    # Normalize the seating width
    # --------------------------------------------------------

    span = float(
        arm_inner_span
    )

    # Empirical image-space baseline.
    #
    # This is deliberately broad and should be calibrated
    # from your own 1/2/3/4/5+ sofa dataset.
    #
    # Typical approximate seating widths:
    #
    # 1 seat  ~ 120-190 px
    # 2 seats ~ 190-300 px
    # 3 seats ~ 280-410 px
    # 4 seats ~ 380-520 px
    # 5+     ~ >500 px
    #
    # These are NOT universal pixel thresholds.
    # Aspect-normalized geometry below reduces dependence
    # on image resolution.

    candidates = {
        "1-seater": 150.0,
        "2-seater": 250.0,
        "3-seater": 350.0,
        "4-seater": 450.0,
        "5+ seater": 600.0,
    }

    # --------------------------------------------------------
    # Resolution normalization
    # --------------------------------------------------------

    image_reference_width = 640.0

    normalized_span = (
        span *
        image_reference_width /
        max(float(image_width), 1.0)
    )

    # --------------------------------------------------------
    # Width-based score
    # --------------------------------------------------------

    width_scores = {}

    for sofa_type, expected in candidates.items():

        difference = abs(
            normalized_span -
            expected
        )

        tolerance = 95.0

        score = max(
            0.0,
            1.0 -
            difference / tolerance
        )

        width_scores[sofa_type] = score

    # --------------------------------------------------------
    # Cushion evidence
    # --------------------------------------------------------

    cushion_scores = {
        "1-seater": 0.0,
        "2-seater": 0.0,
        "3-seater": 0.0,
        "4-seater": 0.0,
        "5+ seater": 0.0,
    }

    if seat_count == 1:

        # One continuous cushion is compatible with
        # all sofa sizes.
        cushion_scores["1-seater"] = 0.10
        cushion_scores["2-seater"] = 0.08
        cushion_scores["3-seater"] = 0.08
        cushion_scores["4-seater"] = 0.06
        cushion_scores["5+ seater"] = 0.04

    elif seat_count == 2:

        cushion_scores["2-seater"] = 0.18
        cushion_scores["3-seater"] = 0.12
        cushion_scores["4-seater"] = 0.10

    elif seat_count == 3:

        cushion_scores["3-seater"] = 0.20
        cushion_scores["4-seater"] = 0.12

    elif seat_count >= 4:

        cushion_scores["4-seater"] = 0.18
        cushion_scores["5+ seater"] = 0.22

    # --------------------------------------------------------
    # Aspect ratio evidence
    # --------------------------------------------------------

    aspect_scores = {
        "1-seater": 0.0,
        "2-seater": 0.0,
        "3-seater": 0.0,
        "4-seater": 0.0,
        "5+ seater": 0.0,
    }

    # Sofa aspect ratio should generally increase with
    # seating capacity, but there is overlap.

    if overall_aspect_ratio > 0:

        aspect_targets = {
            "1-seater": 1.25,
            "2-seater": 1.70,
            "3-seater": 2.30,
            "4-seater": 2.90,
            "5+ seater": 3.50,
        }

        for sofa_type, target in aspect_targets.items():

            difference = abs(
                overall_aspect_ratio -
                target
            )

            aspect_scores[sofa_type] = max(
                0.0,
                1.0 -
                difference / 1.6
            )

    # --------------------------------------------------------
    # Seat width ratio
    # --------------------------------------------------------

    seat_ratio_scores = {
        "1-seater": 0.0,
        "2-seater": 0.0,
        "3-seater": 0.0,
        "4-seater": 0.0,
        "5+ seater": 0.0,
    }

    # The seat should occupy a reasonable percentage of
    # the arm-to-arm span.

    if arm_inner_span > 0:

        ratio = safe_ratio(
            seat_width,
            arm_inner_span
        )

        # Values around 1 are expected.
        # Very high values may mean the seat bounding box
        # includes areas outside the actual usable region.

        ratio_quality = max(
            0.0,
            1.0 -
            abs(ratio - 1.0) / 0.6
        )

        for sofa_type in seat_ratio_scores:

            seat_ratio_scores[sofa_type] = (
                ratio_quality
            )

    # --------------------------------------------------------
    # Combine evidence
    # --------------------------------------------------------

    final_scores = {}

    for sofa_type in candidates:

        score = (

            width_scores[sofa_type] * 0.50 +

            cushion_scores[sofa_type] * 1.0 +

            aspect_scores[sofa_type] * 0.25 +

            seat_ratio_scores[sofa_type] * 0.10

        )

        final_scores[sofa_type] = score

    best_type = max(
        final_scores,
        key=final_scores.get
    )

    best_score = final_scores[
        best_type
    ]

    # --------------------------------------------------------
    # Normalize to 0-100
    # --------------------------------------------------------

    confidence = clamp(
        best_score
    ) * 100.0

    return (
        best_type,
        confidence,
        final_scores,
        normalized_span
    )


# ============================================================
# SEATER ANALYSIS
# ============================================================

(
    estimated_type,
    raw_seater_confidence,
    seater_scores,
    normalized_span
) = estimate_person_capacity()


# ============================================================
# FINAL CONFIDENCE
# ============================================================

if estimated_type is None:

    final_seater_type = "unknown"

    final_confidence = 0.0

    confidence_reason = (
        "Insufficient geometry."
    )

else:

    # Combine:
    #
    # 60% seater geometry
    # 25% YOLO structural quality
    # 15% sectional evidence

    sectional_penalty = (
        sectional_score * 100.0
    )

    final_confidence = (

        raw_seater_confidence * 0.60 +

        structural_quality * 100.0 * 0.25 +

        (100.0 - sectional_penalty) * 0.15

    )

    final_confidence = clamp(
        final_confidence / 100.0
    ) * 100.0

    final_seater_type = (
        estimated_type
    )

    confidence_reason = (
        "Combined seating geometry, "
        "component confidence and sofa structure."
    )


# ============================================================
# SECTIONAL OVERRIDE
# ============================================================

if sectional_score >= 0.65:

    final_seater_type = "sectional"

    final_confidence = max(
        final_confidence,
        sectional_score * 100.0
    )

    confidence_reason = (
        "Multiple structural indicators "
        "suggest sectional geometry."
    )


# ============================================================
# UNKNOWN / LOW CONFIDENCE RULE
# ============================================================

if final_confidence < 55.0:

    final_seater_type = "unknown"

    confidence_reason = (
        "Geometry does not provide enough "
        "evidence for a reliable seater classification."
    )


# ============================================================
# FINAL SEATER REPORT
# ============================================================

log()
log("SEATER ANALYSIS:")

log(
    f"Estimated sofa type: "
    f"{final_seater_type}"
)

log(
    f"Final confidence: "
    f"{final_confidence:.2f}%"
)

log(
    f"Raw geometry confidence: "
    f"{raw_seater_confidence:.2f}%"
)

log(
    f"Structural quality: "
    f"{structural_quality * 100.0:.2f}%"
)

log(
    f"Sectional score: "
    f"{sectional_score * 100.0:.2f}%"
)

log(
    f"Normalized arm-inner span: "
    f"{normalized_span:.2f} px"
)

log(
    f"Reason: "
    f"{confidence_reason}"
)


# ============================================================
# SCORE BREAKDOWN
# ============================================================

log()
log("SEATER CANDIDATE SCORES:")

if seater_scores:

    sorted_scores = sorted(
        seater_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    for sofa_type, score in sorted_scores:

        log(
            f"{sofa_type}: "
            f"{score * 100.0:.2f}%"
        )


# ============================================================
# SECTIONAL REASONS
# ============================================================

if sectional_reasons:

    log()
    log("SECTIONAL EVIDENCE:")

    for reason in sectional_reasons:

        log(
            f"- {reason}"
        )


# ============================================================
# GEOMETRY FEATURE VECTOR
# ============================================================

log()
log("GEOMETRY FEATURE VECTOR:")

log(
    f"overall_width_px = "
    f"{overall_width}"
)

log(
    f"overall_height_px = "
    f"{overall_height}"
)

log(
    f"overall_aspect_ratio = "
    f"{overall_aspect_ratio:.4f}"
)

log(
    f"seat_bbox_width_px = "
    f"{seat_width}"
)

log(
    f"seat_height_px = "
    f"{seat_height}"
)

log(
    f"seat_area_px2 = "
    f"{seat_area:.4f}"
)

log(
    f"seat_aspect_ratio = "
    f"{seat_aspect_ratio:.4f}"
)

log(
    f"seat_bbox_width_ratio = "
    f"{seat_bbox_width_ratio:.4f}"
)

log(
    f"arm_center_span_px = "
    f"{arm_center_span:.4f}"
)

log(
    f"arm_inner_span_px = "
    f"{arm_inner_span:.4f}"
)

log(
    f"usable_seat_span_px = "
    f"{usable_seat_span:.4f}"
)

log(
    f"seat_bbox_to_arm_inner_span_ratio = "
    f"{seat_bbox_to_arm_inner_span_ratio:.4f}"
)

log(
    f"seat_cushion_count = "
    f"{seat_count}"
)

log(
    f"seat_confidence = "
    f"{seat_confidence:.4f}"
)

log(
    f"seat_back_area_ratio = "
    f"{seat_back_area_ratio:.4f}"
)

log(
    f"left_right_arm_area_ratio = "
    f"{left_right_arm_area_ratio:.4f}"
)

log(
    f"structural_quality = "
    f"{structural_quality:.4f}"
)

log(
    f"sectional_score = "
    f"{sectional_score:.4f}"
)

log(
    f"seater_type = "
    f"{final_seater_type}"
)

log(
    f"seater_confidence = "
    f"{final_confidence:.2f}%"
)


# ============================================================
# DRAW VISUALIZATION
# ============================================================

visualization = image.copy()


for component in geometry_components:

    contour = component["contour"]

    class_name = component["class_name"]

    color = CLASS_COLORS.get(
        class_name,
        (255, 255, 255)
    )

    cv2.drawContours(
        visualization,
        [contour],
        -1,
        color,
        2
    )

    draw_component_label(
        visualization,
        component,
        color
    )


# ------------------------------------------------------------
# Draw arm inner edges
# ------------------------------------------------------------

if left_inner_edge is not None:

    cv2.line(
        visualization,
        (
            int(left_inner_edge),
            0
        ),
        (
            int(left_inner_edge),
            image_height
        ),
        (255, 255, 255),
        1
    )


if right_inner_edge is not None:

    cv2.line(
        visualization,
        (
            int(right_inner_edge),
            0
        ),
        (
            int(right_inner_edge),
            image_height
        ),
        (255, 255, 255),
        1
    )


# ------------------------------------------------------------
# Draw sofa type
# ------------------------------------------------------------

draw_text_with_background(
    visualization,
    f"SOFA TYPE: {final_seater_type}",
    (15, 30),
    font_scale=0.75,
    thickness=2,
    color=(255, 255, 255)
)


draw_text_with_background(
    visualization,
    f"CONFIDENCE: {final_confidence:.1f}%",
    (15, 60),
    font_scale=0.65,
    thickness=2,
    color=(255, 255, 255)
)


draw_text_with_background(
    visualization,
    f"SEATS: {seat_count}",
    (15, 90),
    font_scale=0.60,
    thickness=2,
    color=(255, 255, 255)
)


# ============================================================
# SAVE OUTPUT
# ============================================================

success = cv2.imwrite(
    OUTPUT_IMAGE,
    visualization
)

if not success:

    raise RuntimeError(
        f"Could not save visualization:\n"
        f"{OUTPUT_IMAGE}"
    )


with open(
    OUTPUT_REPORT,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "\n".join(report_lines)
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

log()
log("OUTPUT FILES:")

log(
    f"Visualization: "
    f"{OUTPUT_IMAGE}"
)

log(
    f"Text report: "
    f"{OUTPUT_REPORT}"
)

log()
log("Visualization saved successfully.")
log("Analysis complete.")
log(
    f"Text report saved to: "
    f"{OUTPUT_REPORT}"
)

log(
    f"Visualization saved to: "
    f"{OUTPUT_IMAGE}"
)