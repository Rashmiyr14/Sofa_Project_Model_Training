from ultralytics import YOLO
import cv2
import numpy as np
import os


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = r".\runs\segment\models\sofa_yolo11n_seg_v1-2\weights\best.pt"

OUTPUT_DIR = r".\runs\geometry"

IMAGE_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "geometry_result.jpg"
)

TEXT_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "geometry_result.txt"
)

# Minimum confidence for YOLO prediction
PREDICTION_CONFIDENCE = 0.25

# Minimum confidence for geometry calculations
GEOMETRY_CONFIDENCE = 0.50

# Minimum contour area
MIN_COMPONENT_AREA = 100


# ============================================================
# EXPECTED CLASSES
# ============================================================

EXPECTED_CLASSES = [
    "back_cushion",
    "base",
    "left_arm",
    "legs",
    "right_arm",
    "seat_cushion",
]


# ============================================================
# REPORT STORAGE
# ============================================================

report_lines = []


def log(text=""):
    """
    Print text to terminal and store it for the text report.
    """

    print(text)

    report_lines.append(
        str(text)
    )


def save_text_report():
    """
    Save complete analysis report.
    """

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    with open(
        TEXT_OUTPUT,
        "w",
        encoding="utf-8"
    ) as f:

        for line in report_lines:

            f.write(
                line + "\n"
            )


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_class_name(model, class_id):
    """
    Safely obtain YOLO class name.
    """

    try:

        return str(
            model.names[class_id]
        )

    except Exception:

        return f"class_{class_id}"


def get_components(components, class_name):
    """
    Return all components belonging to a class.
    """

    return [
        component
        for component in components
        if component["class_name"] == class_name
    ]


def get_largest_component(components, class_name):
    """
    Return the largest component of a given class.
    """

    matches = get_components(
        components,
        class_name
    )

    if not matches:

        return None

    return max(
        matches,
        key=lambda component: component["area"]
    )


def center_distance(component_a, component_b):
    """
    Calculate Euclidean distance between component centers.
    """

    if (
        component_a is None
        or
        component_b is None
    ):

        return None

    x1, y1 = component_a["center"]

    x2, y2 = component_b["center"]

    return float(
        np.sqrt(
            (x2 - x1) ** 2
            +
            (y2 - y1) ** 2
        )
    )


def safe_ratio(a, b):
    """
    Safe division.
    """

    if a is None or b is None:

        return None

    if b == 0:

        return None

    return float(
        a / b
    )


def format_value(value):
    """
    Format values for the report.
    """

    if value is None:

        return "N/A"

    if isinstance(value, float):

        return f"{value:.4f}"

    return str(value)


def get_bbox_edges(component):
    """
    Return:
        left, top, right, bottom
    """

    x, y, w, h = component["bbox"]

    return (
        x,
        y,
        x + w,
        y + h
    )


def calculate_horizontal_overlap(
    component_a,
    component_b
):
    """
    Calculate horizontal overlap between two bounding boxes.
    """

    if (
        component_a is None
        or
        component_b is None
    ):

        return None

    a_left, _, a_right, _ = get_bbox_edges(
        component_a
    )

    b_left, _, b_right, _ = get_bbox_edges(
        component_b
    )

    overlap_left = max(
        a_left,
        b_left
    )

    overlap_right = min(
        a_right,
        b_right
    )

    overlap = max(
        0,
        overlap_right - overlap_left
    )

    return float(
        overlap
    )


def calculate_usable_seat_span(
    seat,
    physical_left_arm,
    physical_right_arm
):
    """
    Calculate the portion of the seat bounding box
    that lies between the physical inner edges of
    the two arms.

    This avoids using seat pixels that extend underneath
    the arm regions.
    """

    if (
        seat is None
        or
        physical_left_arm is None
        or
        physical_right_arm is None
    ):

        return None

    seat_left, _, seat_right, _ = get_bbox_edges(
        seat
    )

    left_arm_left, _, left_arm_right, _ = get_bbox_edges(
        physical_left_arm
    )

    right_arm_left, _, right_arm_right, _ = get_bbox_edges(
        physical_right_arm
    )

    # Physical left arm's inner edge
    inner_left = left_arm_right

    # Physical right arm's inner edge
    inner_right = right_arm_left

    if inner_right <= inner_left:

        return 0.0

    # Seat portion inside the arm-to-arm region
    usable_left = max(
        seat_left,
        inner_left
    )

    usable_right = min(
        seat_right,
        inner_right
    )

    usable_span = max(
        0,
        usable_right - usable_left
    )

    return float(
        usable_span
    )


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model = YOLO(
        MODEL_PATH
    )

    log(
        "Model loaded successfully."
    )

    log(
        f"Model path: {MODEL_PATH}"
    )

    log(
        f"Classes: {model.names}"
    )

except Exception as e:

    log(
        "ERROR loading model:"
    )

    log(
        str(e)
    )

    save_text_report()

    raise


# ============================================================
# SELECT IMAGE
# ============================================================

image_path = input(
    "\nEnter image path: "
).strip().strip('"')


if not image_path:

    log(
        "ERROR: No image path entered."
    )

    save_text_report()

    raise ValueError(
        "No image path was provided."
    )


if not os.path.exists(
    image_path
):

    log("")
    log(
        "ERROR: Image does not exist."
    )

    log(
        image_path
    )

    save_text_report()

    raise FileNotFoundError(
        image_path
    )


# ============================================================
# READ IMAGE
# ============================================================

image = cv2.imread(
    image_path
)


if image is None:

    log("")
    log(
        "ERROR: Could not read image."
    )

    save_text_report()

    raise ValueError(
        "OpenCV could not read the image."
    )


height, width = image.shape[:2]


log("")
log("==============================================")
log("IMAGE INFORMATION")
log("==============================================")


log(
    f"Image path: {image_path}"
)

log(
    f"Image width: {width} px"
)

log(
    f"Image height: {height} px"
)


# ============================================================
# YOLO SEGMENTATION
# ============================================================

try:

    results = model.predict(
        source=image_path,
        conf=PREDICTION_CONFIDENCE,
        verbose=False
    )

except Exception as e:

    log("")
    log(
        "ERROR during YOLO prediction:"
    )

    log(
        str(e)
    )

    save_text_report()

    raise


if not results:

    log("")
    log(
        "ERROR: YOLO returned no results."
    )

    save_text_report()

    raise RuntimeError(
        "YOLO returned no results."
    )


result = results[0]


# ============================================================
# CHECK DETECTIONS
# ============================================================

if (
    result.boxes is None
    or
    len(result.boxes) == 0
):

    log("")
    log(
        "No objects detected by YOLO."
    )

    cv2.imwrite(
        IMAGE_OUTPUT,
        image
    )

    save_text_report()

    raise RuntimeError(
        "No objects detected."
    )


if result.masks is None:

    log("")
    log(
        "No segmentation masks detected."
    )

    cv2.imwrite(
        IMAGE_OUTPUT,
        image
    )

    save_text_report()

    raise RuntimeError(
        "No segmentation masks detected."
    )


masks = result.masks.data.cpu().numpy()

boxes = result.boxes


detection_count = min(
    len(masks),
    len(boxes)
)


log("")
log("==============================================")
log("YOLO DETECTION SUMMARY")
log("==============================================")


log(
    f"Total detections: {detection_count}"
)

log(
    f"Prediction confidence threshold: "
    f"{PREDICTION_CONFIDENCE:.2f}"
)

log(
    f"Geometry confidence threshold: "
    f"{GEOMETRY_CONFIDENCE:.2f}"
)


# ============================================================
# ALL YOLO PREDICTIONS
# ============================================================

log("")
log("==============================================")
log("ALL YOLO PREDICTIONS")
log("==============================================")


for i in range(
    detection_count
):

    class_id = int(
        boxes.cls[i].item()
    )

    confidence = float(
        boxes.conf[i].item()
    )

    class_name = get_class_name(
        model,
        class_id
    )

    log(
        f"Prediction {i}: "
        f"{class_name} | "
        f"confidence={confidence:.3f}"
    )


# ============================================================
# CLASS COLORS
# ============================================================

class_colors = {

    0: (255, 0, 0),        # back_cushion

    1: (0, 255, 0),        # base

    2: (255, 0, 255),      # left_arm

    3: (0, 255, 255),      # legs

    4: (0, 0, 255),        # right_arm

    5: (255, 255, 0),      # seat_cushion
}


# ============================================================
# COMPONENT GEOMETRY
# ============================================================

log("")
log("==============================================")
log("COMPONENT GEOMETRY")
log("==============================================")


components = []


# Start visualization from original image.
visualization = image.copy()


for i in range(
    detection_count
):

    class_id = int(
        boxes.cls[i].item()
    )

    confidence = float(
        boxes.conf[i].item()
    )

    class_name = get_class_name(
        model,
        class_id
    )


    # --------------------------------------------------------
    # Prediction threshold
    # --------------------------------------------------------

    if confidence < PREDICTION_CONFIDENCE:

        continue


    # --------------------------------------------------------
    # Resize mask
    # --------------------------------------------------------

    mask_resized = cv2.resize(
        masks[i],
        (width, height),
        interpolation=cv2.INTER_NEAREST
    )


    binary_mask = (
        mask_resized > 0.5
    ).astype(
        np.uint8
    ) * 255


    # --------------------------------------------------------
    # Find contours
    # --------------------------------------------------------

    contours, _ = cv2.findContours(
        binary_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    if not contours:

        continue


    # Largest contour
    contour = max(
        contours,
        key=cv2.contourArea
    )


    area = cv2.contourArea(
        contour
    )


    if area < MIN_COMPONENT_AREA:

        continue


    # --------------------------------------------------------
    # Bounding box
    # --------------------------------------------------------

    x, y, w, h = cv2.boundingRect(
        contour
    )


    aspect_ratio = safe_ratio(
        w,
        h
    )


    # --------------------------------------------------------
    # Perimeter
    # --------------------------------------------------------

    perimeter = cv2.arcLength(
        contour,
        True
    )


    # --------------------------------------------------------
    # Centroid
    # --------------------------------------------------------

    moments = cv2.moments(
        contour
    )


    if moments["m00"] != 0:

        cx = (
            moments["m10"]
            /
            moments["m00"]
        )

        cy = (
            moments["m01"]
            /
            moments["m00"]
        )

    else:

        cx = x + w / 2

        cy = y + h / 2


    center = (
        float(cx),
        float(cy)
    )


    # --------------------------------------------------------
    # Store component
    # --------------------------------------------------------

    component = {

        "index":
            i,

        "class_id":
            class_id,

        "class_name":
            class_name,

        "confidence":
            confidence,

        "area":
            float(area),

        "bbox":
            (
                int(x),
                int(y),
                int(w),
                int(h)
            ),

        "aspect_ratio":
            aspect_ratio,

        "perimeter":
            float(perimeter),

        "center":
            center,

        "contour":
            contour,
    }


    components.append(
        component
    )


    # ========================================================
    # FINAL VISUALIZATION RULE
    # ========================================================
    #
    # Only geometry-valid detections are drawn.
    #
    # Therefore:
    #
    # confidence >= 0.50 -> DRAW
    # confidence <  0.50 -> DO NOT DRAW
    #
    # Low-confidence detections are still shown in the
    # textual YOLO prediction report above.
    # ========================================================

    if confidence < GEOMETRY_CONFIDENCE:

        continue


    # --------------------------------------------------------
    # Visualization color
    # --------------------------------------------------------

    color = class_colors.get(
        class_id,
        (255, 255, 255)
    )


    # --------------------------------------------------------
    # Mask overlay
    # --------------------------------------------------------

    mask_pixels = (
        binary_mask > 0
    )


    overlay = visualization.copy()


    overlay[
        mask_pixels
    ] = color


    visualization = cv2.addWeighted(
        visualization,
        0.70,
        overlay,
        0.30,
        0
    )


    # --------------------------------------------------------
    # Contour
    # --------------------------------------------------------

    cv2.drawContours(
        visualization,
        [contour],
        -1,
        color,
        2
    )


    # --------------------------------------------------------
    # Bounding box
    # --------------------------------------------------------

    cv2.rectangle(
        visualization,
        (x, y),
        (x + w, y + h),
        color,
        2
    )


    # --------------------------------------------------------
    # Center
    # --------------------------------------------------------

    cv2.circle(
        visualization,
        (
            int(cx),
            int(cy)
        ),
        5,
        color,
        -1
    )


    # --------------------------------------------------------
    # Label
    # --------------------------------------------------------

    label = (
        f"{class_name} "
        f"{confidence:.2f}"
    )


    label_y = max(
        y - 7,
        18
    )


    cv2.putText(
        visualization,
        label,
        (
            x,
            label_y
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA
    )


# ============================================================
# PRINT COMPONENT GEOMETRY
# ============================================================

for component in components:

    x, y, w, h = component["bbox"]

    cx, cy = component["center"]


    log("")


    log(
        f"{component['class_name']} | "
        f"confidence={component['confidence']:.3f} | "
        f"area={component['area']:.1f} px2 | "
        f"bbox=({x},{y},{w},{h}) | "
        f"AR={component['aspect_ratio']:.2f} | "
        f"perimeter={component['perimeter']:.1f} px | "
        f"center=({cx:.1f},{cy:.1f})"
    )


# ============================================================
# GEOMETRY-VALID COMPONENTS
# ============================================================

geometry_components = [

    component

    for component in components

    if component["confidence"]
    >= GEOMETRY_CONFIDENCE
]


log("")
log("==============================================")
log("GEOMETRY-VALID COMPONENTS")
log("==============================================")


log(
    f"Geometry confidence threshold: "
    f"{GEOMETRY_CONFIDENCE:.2f}"
)


log(
    f"Geometry-valid components: "
    f"{len(geometry_components)}"
)


for component in geometry_components:

    log(
        f"{component['class_name']} | "
        f"confidence={component['confidence']:.3f}"
    )


# ============================================================
# SELECT MAIN COMPONENTS
# ============================================================

back = get_largest_component(
    geometry_components,
    "back_cushion"
)


base = get_largest_component(
    geometry_components,
    "base"
)


seat = get_largest_component(
    geometry_components,
    "seat_cushion"
)


left_arm_candidates = get_components(
    geometry_components,
    "left_arm"
)


right_arm_candidates = get_components(
    geometry_components,
    "right_arm"
)


legs = get_components(
    geometry_components,
    "legs"
)


# ============================================================
# PHYSICAL LEFT / RIGHT ARM DETECTION
# ============================================================
#
# Important:
#
# The YOLO class names are:
#
#     left_arm
#     right_arm
#
# But the model can occasionally assign the labels
# opposite to the actual image position.
#
# Therefore, geometry uses horizontal image position:
#
# smallest X center = physical LEFT
# largest X center  = physical RIGHT
#
# The original YOLO class name is retained and reported.
# ============================================================

all_arms = (
    left_arm_candidates
    +
    right_arm_candidates
)


all_arms = sorted(
    all_arms,
    key=lambda component:
        component["center"][0]
)


physical_left_arm = None
physical_right_arm = None


if len(all_arms) >= 2:

    physical_left_arm = all_arms[0]

    physical_right_arm = all_arms[-1]


elif len(all_arms) == 1:

    if (
        all_arms[0]["center"][0]
        <
        width / 2
    ):

        physical_left_arm = all_arms[0]

    else:

        physical_right_arm = all_arms[0]


# ============================================================
# REPORT PHYSICAL ARM ASSIGNMENT
# ============================================================

log("")
log("==============================================")
log("PHYSICAL ARM ASSIGNMENT")
log("==============================================")


if physical_left_arm is not None:

    log(
        "Physical LEFT arm:"
    )

    log(
        f"  YOLO class = "
        f"{physical_left_arm['class_name']}"
    )

    log(
        f"  confidence = "
        f"{physical_left_arm['confidence']:.3f}"
    )

    log(
        f"  center = "
        f"({physical_left_arm['center'][0]:.1f}, "
        f"{physical_left_arm['center'][1]:.1f})"
    )

else:

    log(
        "Physical LEFT arm: N/A"
    )


if physical_right_arm is not None:

    log(
        "Physical RIGHT arm:"
    )

    log(
        f"  YOLO class = "
        f"{physical_right_arm['class_name']}"
    )

    log(
        f"  confidence = "
        f"{physical_right_arm['confidence']:.3f}"
    )

    log(
        f"  center = "
        f"({physical_right_arm['center'][0]:.1f}, "
        f"{physical_right_arm['center'][1]:.1f})"
    )

else:

    log(
        "Physical RIGHT arm: N/A"
    )


# ============================================================
# COMPONENT RELATIONSHIPS
# ============================================================

log("")
log("==============================================")
log("COMPONENT RELATIONSHIPS")
log("==============================================")


# ------------------------------------------------------------
# Arm center distance
# ------------------------------------------------------------

if (
    physical_left_arm is not None
    and
    physical_right_arm is not None
):

    arm_center_distance = center_distance(
        physical_left_arm,
        physical_right_arm
    )

    log(
        f"Physical left arm <-> "
        f"physical right arm center distance: "
        f"{arm_center_distance:.2f} px"
    )

else:

    arm_center_distance = None

    log(
        "Physical left arm <-> "
        "physical right arm center distance: N/A"
    )


# ------------------------------------------------------------
# Seat/back distance
# ------------------------------------------------------------

if (
    seat is not None
    and
    back is not None
):

    seat_back_distance = center_distance(
        seat,
        back
    )

    log(
        f"Seat <-> Back cushion center distance: "
        f"{seat_back_distance:.2f} px"
    )

else:

    seat_back_distance = None

    log(
        "Seat <-> Back cushion center distance: N/A"
    )


# ------------------------------------------------------------
# Left arm/seat distance
# ------------------------------------------------------------

if (
    physical_left_arm is not None
    and
    seat is not None
):

    left_arm_seat_distance = center_distance(
        physical_left_arm,
        seat
    )

    log(
        f"Physical left arm <-> Seat "
        f"center distance: "
        f"{left_arm_seat_distance:.2f} px"
    )

else:

    left_arm_seat_distance = None

    log(
        "Physical left arm <-> Seat "
        "center distance: N/A"
    )


# ------------------------------------------------------------
# Right arm/seat distance
# ------------------------------------------------------------

if (
    physical_right_arm is not None
    and
    seat is not None
):

    right_arm_seat_distance = center_distance(
        physical_right_arm,
        seat
    )

    log(
        f"Physical right arm <-> Seat "
        f"center distance: "
        f"{right_arm_seat_distance:.2f} px"
    )

else:

    right_arm_seat_distance = None

    log(
        "Physical right arm <-> Seat "
        "center distance: N/A"
    )


# ============================================================
# OVERALL SOFA GEOMETRY
# ============================================================

log("")
log("==============================================")
log("OVERALL SOFA GEOMETRY")
log("==============================================")


geometry_boxes = []


for component in geometry_components:

    x, y, w, h = component["bbox"]

    geometry_boxes.append(
        (
            x,
            y,
            x + w,
            y + h
        )
    )


if geometry_boxes:

    min_x = min(
        box[0]
        for box in geometry_boxes
    )

    min_y = min(
        box[1]
        for box in geometry_boxes
    )

    max_x = max(
        box[2]
        for box in geometry_boxes
    )

    max_y = max(
        box[3]
        for box in geometry_boxes
    )

    overall_width = float(
        max_x - min_x
    )

    overall_height = float(
        max_y - min_y
    )

    overall_aspect_ratio = safe_ratio(
        overall_width,
        overall_height
    )

else:

    overall_width = None

    overall_height = None

    overall_aspect_ratio = None


log(
    f"Overall width: "
    f"{format_value(overall_width)} px"
)


log(
    f"Overall height: "
    f"{format_value(overall_height)} px"
)


log(
    f"Overall aspect ratio: "
    f"{format_value(overall_aspect_ratio)}"
)


# ============================================================
# COMPONENT STRUCTURE
# ============================================================

log("")
log("==============================================")
log("COMPONENT STRUCTURE")
log("==============================================")


component_counts = {}


for class_name in EXPECTED_CLASSES:

    count = len(
        get_components(
            geometry_components,
            class_name
        )
    )

    component_counts[
        class_name
    ] = count

    log(
        f"{class_name}: {count}"
    )


# ============================================================
# SEAT MASK QUALITY
# ============================================================

log("")
log("==============================================")
log("SEAT MASK QUALITY")
log("==============================================")


seat_quality = "UNDETERMINED"

seat_confidence = None

seat_width = None

seat_height = None

seat_area = None

seat_aspect_ratio = None


if seat is None:

    log(
        "No geometry-valid seat cushion detected."
    )

else:

    seat_confidence = seat[
        "confidence"
    ]

    x, y, w, h = seat[
        "bbox"
    ]

    seat_width = w

    seat_height = h

    seat_area = seat[
        "area"
    ]

    seat_aspect_ratio = seat[
        "aspect_ratio"
    ]


    concerns = []


    if seat_confidence < GEOMETRY_CONFIDENCE:

        concerns.append(
            "low confidence"
        )


    if seat_height < 30:

        concerns.append(
            "very small seat height"
        )


    if seat_aspect_ratio > 10:

        concerns.append(
            "extreme aspect ratio"
        )


    if seat_area < 500:

        concerns.append(
            "small mask area"
        )


    if len(concerns) == 0:

        seat_quality = "GOOD"

    elif len(concerns) == 1:

        seat_quality = "WEAK"

    else:

        seat_quality = "POOR"


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


    log(
        f"Seat mask quality: "
        f"{seat_quality}"
    )


    if concerns:

        log(
            "Seat mask concerns: "
            +
            ", ".join(concerns)
        )


# ============================================================
# SEAT GEOMETRY
# ============================================================

log("")
log("==============================================")
log("SEAT GEOMETRY")
log("==============================================")


seat_width_ratio = None

seat_back_area_ratio = None

left_right_arm_area_ratio = None


if seat is not None:

    log(
        f"Detected seat bounding-box width: "
        f"{seat_width} px"
    )


    log(
        f"Seat height: "
        f"{seat_height} px"
    )


    log(
        f"Seat aspect ratio: "
        f"{seat_aspect_ratio:.2f}"
    )


    # --------------------------------------------------------
    # Seat / overall width
    # --------------------------------------------------------

    if overall_width is not None:

        seat_width_ratio = safe_ratio(
            seat_width,
            overall_width
        )

        log(
            f"Seat bounding-box / overall width: "
            f"{seat_width_ratio:.4f}"
        )

    else:

        log(
            "Seat bounding-box / overall width: N/A"
        )


    # --------------------------------------------------------
    # Seat / back area
    # --------------------------------------------------------

    if back is not None:

        seat_back_area_ratio = safe_ratio(
            seat["area"],
            back["area"]
        )

        log(
            f"Seat / Back area ratio: "
            f"{seat_back_area_ratio:.4f}"
        )

    else:

        log(
            "Seat / Back area ratio: N/A"
        )


    # --------------------------------------------------------
    # Arm area symmetry
    # --------------------------------------------------------

    if (
        physical_left_arm is not None
        and
        physical_right_arm is not None
    ):

        left_right_arm_area_ratio = safe_ratio(
            physical_left_arm["area"],
            physical_right_arm["area"]
        )

        log(
            f"Physical left / physical right "
            f"arm area ratio: "
            f"{left_right_arm_area_ratio:.4f}"
        )

    else:

        log(
            "Physical left / physical right "
            "arm area ratio: N/A"
        )

else:

    log(
        "Seat geometry unavailable."
    )


# ============================================================
# SOFA SPAN GEOMETRY
# ============================================================

log("")
log("==============================================")
log("SOFA SPAN GEOMETRY")
log("==============================================")


arm_center_span = None

arm_inner_span = None


if (
    physical_left_arm is not None
    and
    physical_right_arm is not None
):

    # --------------------------------------------------------
    # Arm center-to-center span
    # --------------------------------------------------------

    arm_center_span = center_distance(
        physical_left_arm,
        physical_right_arm
    )

    log(
        f"Arm center-to-center span: "
        f"{arm_center_span:.2f} px"
    )


    # --------------------------------------------------------
    # Physical inner edges
    # --------------------------------------------------------

    left_arm_x = (
        physical_left_arm["bbox"][0]
    )

    left_arm_width = (
        physical_left_arm["bbox"][2]
    )

    right_arm_x = (
        physical_right_arm["bbox"][0]
    )


    left_inner_edge = (
        left_arm_x
        +
        left_arm_width
    )


    right_inner_edge = (
        right_arm_x
    )


    arm_inner_span = max(
        0.0,
        float(
            right_inner_edge
            -
            left_inner_edge
        )
    )


    log(
        f"Physical left arm inner edge: "
        f"{left_inner_edge} px"
    )


    log(
        f"Physical right arm inner edge: "
        f"{right_inner_edge} px"
    )


    log(
        f"Usable arm-to-arm inner span: "
        f"{arm_inner_span:.2f} px"
    )


else:

    log(
        "Arm span unavailable."
    )


# ============================================================
# CORRECTED USABLE SEAT SPAN
# ============================================================

log("")
log("==============================================")
log("USABLE SEAT SPAN")
log("==============================================")


usable_seat_span = None

seat_span_ratio = None


if (
    seat is not None
    and
    physical_left_arm is not None
    and
    physical_right_arm is not None
):

    usable_seat_span = calculate_usable_seat_span(
        seat,
        physical_left_arm,
        physical_right_arm
    )


    log(
        f"Detected seat bounding-box span: "
        f"{seat['bbox'][2]} px"
    )


    log(
        f"Usable seat span between arm inner edges: "
        f"{usable_seat_span:.2f} px"
    )


    if arm_inner_span is not None:

        seat_span_ratio = safe_ratio(
            usable_seat_span,
            arm_inner_span
        )


        log(
            f"Usable seat span / "
            f"arm inner span: "
            f"{seat_span_ratio:.4f}"
        )


    else:

        log(
            "Usable seat span / "
            "arm inner span: N/A"
        )


else:

    log(
        "Usable seat span: N/A"
    )

    log(
        "Reason: seat and both physical arms "
        "are required."
    )


# ============================================================
# SEAT CONFIGURATION
# ============================================================

log("")
log("==============================================")
log("SEAT CONFIGURATION")
log("==============================================")


seat_cushion_count = component_counts[
    "seat_cushion"
]


if seat_cushion_count == 0:

    log(
        "No continuous seat cushion detected."
    )

elif seat_cushion_count == 1:

    log(
        "1 seat cushion component detected."
    )

else:

    log(
        f"{seat_cushion_count} seat cushion "
        f"components detected."
    )


# ============================================================
# SEATER ANALYSIS
# ============================================================

log("")
log("==============================================")
log("SEATER ANALYSIS")
log("==============================================")


if seat is None:

    log(
        "Seater analysis: UNDETERMINED"
    )

    log(
        "Reason: no geometry-valid seat mask."
    )


elif seat_quality in [
    "WEAK",
    "POOR"
]:

    log(
        "Seater analysis: UNDETERMINED"
    )

    log(
        "Reason: seat mask quality is "
        f"{seat_quality}."
    )


elif seat_span_ratio is None:

    log(
        "Seater analysis: UNDETERMINED"
    )

    log(
        "Reason: usable seat span could not "
        "be calculated."
    )


else:

    log(
        "Seat geometry is strong enough "
        "for preliminary analysis."
    )


    log(
        f"Usable seat span ratio: "
        f"{seat_span_ratio:.4f}"
    )


    log(
        f"Seat aspect ratio: "
        f"{seat_aspect_ratio:.2f}"
    )


    log(
        f"Seat cushion count: "
        f"{seat_cushion_count}"
    )


    # --------------------------------------------------------
    # Geometry interpretation
    # --------------------------------------------------------

    if seat_span_ratio >= 0.80:

        geometry_interpretation = (
            "long continuous usable seating span"
        )

    elif seat_span_ratio >= 0.55:

        geometry_interpretation = (
            "moderate usable seating span"
        )

    else:

        geometry_interpretation = (
            "short usable seating span"
        )


    log(
        "Geometry interpretation: "
        f"{geometry_interpretation}"
    )


    log("")


    log(
        "Preliminary seater interpretation: "
        "requires calibration."
    )


    log(
        "A single continuous cushion is NOT "
        "automatically treated as one seater."
    )


    log(
        "Seater reasoning should combine seat "
        "span, sofa proportions, component geometry "
        "and structural divisions."
    )


# ============================================================
# GEOMETRY FEATURE VECTOR
# ============================================================

log("")
log("==============================================")
log("GEOMETRY FEATURE VECTOR")
log("==============================================")


feature_vector = {

    "overall_width_px":
        overall_width,

    "overall_height_px":
        overall_height,

    "overall_aspect_ratio":
        overall_aspect_ratio,

    "seat_bbox_width_px":
        seat_width,

    "seat_height_px":
        seat_height,

    "seat_area_px2":
        seat_area,

    "seat_aspect_ratio":
        seat_aspect_ratio,

    "seat_bbox_width_ratio":
        seat_width_ratio,

    "arm_center_span_px":
        arm_center_span,

    "arm_inner_span_px":
        arm_inner_span,

    "usable_seat_span_px":
        usable_seat_span,

    "usable_seat_span_ratio":
        seat_span_ratio,

    "seat_cushion_count":
        seat_cushion_count,

    "seat_confidence":
        seat_confidence,

    "seat_mask_quality":
        seat_quality,

    "seat_back_area_ratio":
        seat_back_area_ratio,

    "left_right_arm_area_ratio":
        left_right_arm_area_ratio,
}


for key, value in feature_vector.items():

    log(
        f"{key} = "
        f"{format_value(value)}"
    )


# ============================================================
# ANALYSIS LIMITATIONS
# ============================================================

log("")
log("==============================================")
log("ANALYSIS LIMITATIONS")
log("==============================================")


log(
    "Measurements are image-space pixels."
)


log(
    "Pixel measurements do not represent real "
    "physical dimensions without scale or calibration."
)


log(
    "Perspective and camera angle can change "
    "measured geometry."
)


log(
    "Sofa design can change the visible geometry."
)


log(
    "A single continuous cushion does not by itself "
    "determine seating capacity."
)


log(
    "Multiple seat cushions do not automatically "
    "guarantee a specific seating capacity."
)


log(
    "The physical left/right arm assignment is based "
    "on horizontal image position, not only YOLO class name."
)


log(
    "Low-confidence detections are included in the "
    "prediction report but are excluded from geometry "
    "and final visualization when confidence is below "
    f"{GEOMETRY_CONFIDENCE:.2f}."
)


log(
    "Seater classification is an image-based inference."
)


log(
    "The current seater interpretation is deliberately "
    "preliminary and requires calibration."
)


# ============================================================
# OUTPUT FILES
# ============================================================

log("")
log("==============================================")
log("OUTPUT FILES")
log("==============================================")


log(
    f"Visualization: {IMAGE_OUTPUT}"
)


log(
    f"Text report: {TEXT_OUTPUT}"
)


# ============================================================
# SAVE VISUALIZATION
# ============================================================

success = cv2.imwrite(
    IMAGE_OUTPUT,
    visualization
)


if success:

    log(
        "Visualization saved successfully."
    )

else:

    log(
        "WARNING: Could not save visualization."
    )


# ============================================================
# SAVE TEXT REPORT
# ============================================================

save_text_report()


# ============================================================
# FINAL MESSAGE
# ============================================================

print("")
print("==============================================")
print("ANALYSIS COMPLETE")
print("==============================================")


print(
    f"Text report: {TEXT_OUTPUT}"
)


print(
    f"Visualization: {IMAGE_OUTPUT}"
)


print(
    f"Valid geometry components: "
    f"{len(geometry_components)}"
)


print(
    f"Seat cushion components: "
    f"{seat_cushion_count}"
)
