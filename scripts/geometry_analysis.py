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

# YOLO prediction threshold.
# All predictions >= this threshold are returned by YOLO.
PREDICTION_CONFIDENCE = 0.25

# Only detections >= this threshold participate in geometry analysis.
GEOMETRY_CONFIDENCE = 0.50

# Ignore extremely small contours.
MIN_COMPONENT_AREA = 100


# ============================================================
# REPORT STORAGE
# ============================================================

report_lines = []


def log(text=""):
    """Print text and store it for the report."""
    print(text)
    report_lines.append(str(text))


def save_text_report():
    """Save the collected report to a text file."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(
        TEXT_OUTPUT,
        "w",
        encoding="utf-8"
    ) as f:

        for line in report_lines:
            f.write(line + "\n")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_components(components, class_name):
    """Return all components matching a class name."""

    return [
        component
        for component in components
        if component["class_name"] == class_name
    ]


def get_largest_component(components, class_name):
    """Return the largest component for a class."""

    matches = get_components(
        components,
        class_name
    )

    if not matches:
        return None

    return max(
        matches,
        key=lambda x: x["area"]
    )


def center_distance(component_a, component_b):
    """Calculate Euclidean distance between two component centers."""

    if component_a is None or component_b is None:
        return None

    x1, y1 = component_a["center"]
    x2, y2 = component_b["center"]

    return float(
        np.sqrt(
            (x2 - x1) ** 2 +
            (y2 - y1) ** 2
        )
    )


def safe_ratio(a, b):
    """Return a / b safely."""

    if a is None or b is None:
        return None

    if b == 0:
        return None

    return float(a / b)


def format_value(value):
    """Format report values consistently."""

    if value is None:
        return "N/A"

    if isinstance(value, float):
        return f"{value:.4f}"

    return str(value)


# ============================================================
# VISUALIZATION COLORS
# ============================================================

CLASS_COLORS = {

    "back_cushion": (255, 0, 0),

    "base": (0, 255, 0),

    "seat_cushion": (255, 255, 0),

    "legs": (0, 255, 255),

    "left_arm": (255, 0, 255),

    "right_arm": (255, 128, 0),
}


# ============================================================
# DRAW TEXT WITH BACKGROUND
# ============================================================

def draw_text_with_background(
    image,
    text,
    position,
    color,
    font_scale=0.55,
    thickness=2
):

    x, y = position

    font = cv2.FONT_HERSHEY_SIMPLEX

    (
        text_width,
        text_height
    ), baseline = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness
    )

    x = max(
        5,
        min(
            x,
            image.shape[1] - text_width - 5
        )
    )

    y = max(
        text_height + 5,
        min(
            y,
            image.shape[0] - 5
        )
    )

    cv2.rectangle(
        image,
        (
            x - 3,
            y - text_height - baseline - 3
        ),
        (
            x + text_width + 3,
            y + baseline + 3
        ),
        (255, 255, 255),
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


# ============================================================
# DRAW COMPONENT LABEL
# ============================================================

def draw_component_label(
    image,
    component,
    display_name,
    color
):

    x, y, w, h = component["bbox"]

    label = (
        f"{display_name} "
        f"{component['confidence']:.2f}"
    )

    label_x = x
    label_y = y - 8

    if label_y < 20:
        label_y = y + 22

    draw_text_with_background(
        image,
        label,
        (
            label_x,
            label_y
        ),
        color,
        font_scale=0.48,
        thickness=1
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

    model = YOLO(MODEL_PATH)

    log("Model loaded successfully.")
    log(
        f"Model path: {MODEL_PATH}"
    )

    log(
        f"Classes: {model.names}"
    )

except Exception as e:

    log("ERROR loading model:")
    log(str(e))

    save_text_report()

    raise


# ============================================================
# SELECT IMAGE
# ============================================================

image_path = input(
    "\nEnter image path: "
).strip().strip('"')


if not os.path.exists(image_path):

    log("")
    log("ERROR: Image does not exist.")
    log(image_path)

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
    log("ERROR: Could not read image.")

    save_text_report()

    raise ValueError(
        "OpenCV could not read the image."
    )


height, width = image.shape[:2]


# ============================================================
# IMAGE INFORMATION
# ============================================================

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

results = model.predict(
    source=image_path,
    conf=PREDICTION_CONFIDENCE,
    verbose=False
)


if not results:

    log("")
    log("ERROR: YOLO returned no results.")

    cv2.imwrite(
        IMAGE_OUTPUT,
        image
    )

    save_text_report()

    raise RuntimeError(
        "YOLO returned no results."
    )


result = results[0]

boxes = result.boxes


# ============================================================
# CHECK DETECTIONS
# ============================================================

if boxes is None or len(boxes) == 0:

    log("")
    log("No detections found.")

    cv2.imwrite(
        IMAGE_OUTPUT,
        image
    )

    save_text_report()

    raise RuntimeError(
        "No YOLO detections found."
    )


# ============================================================
# CHECK SEGMENTATION MASKS
# ============================================================

if result.masks is None:

    log("")
    log("No segmentation masks detected.")

    cv2.imwrite(
        IMAGE_OUTPUT,
        image
    )

    save_text_report()

    raise RuntimeError(
        "No segmentation masks detected."
    )


masks = result.masks.data.cpu().numpy()


# ============================================================
# YOLO DETECTION SUMMARY
# ============================================================

log("")
log("==============================================")
log("YOLO DETECTION SUMMARY")
log("==============================================")


log(
    f"Total detections: {len(boxes)}"
)

log(
    "Prediction confidence threshold: "
    f"{PREDICTION_CONFIDENCE:.2f}"
)

log(
    "Geometry confidence threshold: "
    f"{GEOMETRY_CONFIDENCE:.2f}"
)

log(
    f"Minimum component area: "
    f"{MIN_COMPONENT_AREA} px2"
)


# ============================================================
# ALL YOLO PREDICTIONS
# ============================================================

log("")
log("==============================================")
log("ALL YOLO PREDICTIONS")
log("==============================================")


for i in range(len(boxes)):

    class_id = int(
        boxes.cls[i].item()
    )

    confidence = float(
        boxes.conf[i].item()
    )

    class_name = model.names[class_id]

    log(
        f"Prediction {i}: "
        f"{class_name} | "
        f"confidence={confidence:.3f}"
    )


# ============================================================
# COMPONENT GEOMETRY
# ============================================================

log("")
log("==============================================")
log("COMPONENT GEOMETRY")
log("==============================================")


components = []


for i in range(len(masks)):

    class_id = int(
        boxes.cls[i].item()
    )

    confidence = float(
        boxes.conf[i].item()
    )

    class_name = model.names[class_id]


    # --------------------------------------------------------
    # RESIZE MASK
    # --------------------------------------------------------

    mask_resized = cv2.resize(
        masks[i],
        (width, height),
        interpolation=cv2.INTER_NEAREST
    )


    binary_mask = (
        mask_resized > 0.5
    ).astype(np.uint8) * 255


    # --------------------------------------------------------
    # FIND CONTOURS
    # --------------------------------------------------------

    contours, _ = cv2.findContours(
        binary_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    if not contours:
        continue


    # --------------------------------------------------------
    # LARGEST CONTOUR
    # --------------------------------------------------------

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
    # BOUNDING BOX
    # --------------------------------------------------------

    x, y, w, h = cv2.boundingRect(
        contour
    )


    aspect_ratio = safe_ratio(
        w,
        h
    )


    # --------------------------------------------------------
    # PERIMETER
    # --------------------------------------------------------

    perimeter = cv2.arcLength(
        contour,
        True
    )


    # --------------------------------------------------------
    # CENTROID
    # --------------------------------------------------------

    moments = cv2.moments(
        contour
    )


    if moments["m00"] != 0:

        cx = (
            moments["m10"] /
            moments["m00"]
        )

        cy = (
            moments["m01"] /
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
    # STORE COMPONENT
    # --------------------------------------------------------

    component = {

        "index": i,

        "class_id": class_id,

        "class_name": class_name,

        "confidence": confidence,

        "area": float(area),

        "bbox": (
            x,
            y,
            w,
            h
        ),

        "aspect_ratio": aspect_ratio,

        "perimeter": float(perimeter),

        "center": center,

        "contour": contour,
    }


    components.append(
        component
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
    "Geometry confidence threshold: "
    f"{GEOMETRY_CONFIDENCE:.2f}"
)


log(
    "Geometry-valid components: "
    f"{len(geometry_components)}"
)


for component in geometry_components:

    log(
        f"{component['class_name']} | "
        f"confidence="
        f"{component['confidence']:.3f}"
    )


# ============================================================
# INITIAL VISUALIZATION
# ONLY GEOMETRY-VALID COMPONENTS
# ============================================================

visualization = image.copy()


for component in geometry_components:

    class_name = component["class_name"]

    contour = component["contour"]

    x, y, w, h = component["bbox"]

    cx, cy = component["center"]


    # --------------------------------------------------------
    # COLOR
    # --------------------------------------------------------

    color = CLASS_COLORS.get(
        class_name,
        (200, 200, 200)
    )


    # --------------------------------------------------------
    # MASK
    # --------------------------------------------------------

    mask_resized = cv2.resize(
        masks[component["index"]],
        (width, height),
        interpolation=cv2.INTER_NEAREST
    )


    binary_mask = (
        mask_resized > 0.5
    ).astype(np.uint8) * 255


    mask_pixels = (
        binary_mask > 0
    )


    overlay = visualization.copy()

    overlay[mask_pixels] = color


    visualization = cv2.addWeighted(
        visualization,
        0.75,
        overlay,
        0.25,
        0
    )


    # --------------------------------------------------------
    # CONTOUR
    # --------------------------------------------------------

    cv2.drawContours(
        visualization,
        [contour],
        -1,
        color,
        2
    )


    # --------------------------------------------------------
    # BOUNDING BOX
    # --------------------------------------------------------

    cv2.rectangle(
        visualization,
        (x, y),
        (x + w, y + h),
        color,
        2
    )


    # --------------------------------------------------------
    # CENTER
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
# PHYSICAL ARM ASSIGNMENT
# BASED ON IMAGE POSITION
# ============================================================

all_arms = (
    left_arm_candidates +
    right_arm_candidates
)


all_arms = sorted(
    all_arms,
    key=lambda component: component["center"][0]
)


physical_left_arm = None
physical_right_arm = None


if len(all_arms) >= 2:

    # The arm with the smaller x-center
    # is physically on the left side.

    physical_left_arm = all_arms[0]

    # The arm with the larger x-center
    # is physically on the right side.

    physical_right_arm = all_arms[-1]


elif len(all_arms) == 1:

    if (
        all_arms[0]["center"][0]
        < width / 2
    ):

        physical_left_arm = all_arms[0]

    else:

        physical_right_arm = all_arms[0]


# ============================================================
# VISUAL LABELS
# ============================================================

if physical_left_arm:

    draw_component_label(
        visualization,
        physical_left_arm,
        "PHYSICAL LEFT ARM",
        (0, 0, 255)
    )


if physical_right_arm:

    draw_component_label(
        visualization,
        physical_right_arm,
        "PHYSICAL RIGHT ARM",
        (0, 0, 255)
    )


if back:

    draw_component_label(
        visualization,
        back,
        "BACK CUSHION",
        (255, 0, 0)
    )


if seat:

    draw_component_label(
        visualization,
        seat,
        "SEAT CUSHION",
        (0, 150, 150)
    )


if base:

    draw_component_label(
        visualization,
        base,
        "BASE",
        (0, 180, 0)
    )


# ============================================================
# COMPONENT GEOMETRY REPORT
# ============================================================

for component in components:

    x, y, w, h = component["bbox"]

    cx, cy = component["center"]

    log("")

    log(
        f"{component['class_name']} | "
        f"confidence="
        f"{component['confidence']:.3f} | "
        f"area="
        f"{component['area']:.1f} px2 | "
        f"bbox="
        f"({x},{y},{w},{h}) | "
        f"AR="
        f"{component['aspect_ratio']:.2f} | "
        f"perimeter="
        f"{component['perimeter']:.1f} px | "
        f"center="
        f"({cx:.1f},{cy:.1f})"
    )


# ============================================================
# PHYSICAL ARM ASSIGNMENT REPORT
# ============================================================

log("")
log("==============================================")
log("PHYSICAL ARM ASSIGNMENT")
log("==============================================")


if physical_left_arm:

    log("Physical LEFT arm:")

    log(
        "  YOLO class = "
        f"{physical_left_arm['class_name']}"
    )

    log(
        "  confidence = "
        f"{physical_left_arm['confidence']:.3f}"
    )

    log(
        "  center = "
        f"({physical_left_arm['center'][0]:.1f}, "
        f"{physical_left_arm['center'][1]:.1f})"
    )

else:

    log(
        "Physical LEFT arm: N/A"
    )


if physical_right_arm:

    log("Physical RIGHT arm:")

    log(
        "  YOLO class = "
        f"{physical_right_arm['class_name']}"
    )

    log(
        "  confidence = "
        f"{physical_right_arm['confidence']:.3f}"
    )

    log(
        "  center = "
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
# ARM TO ARM
# ------------------------------------------------------------

if (
    physical_left_arm
    and physical_right_arm
):

    arm_distance = center_distance(
        physical_left_arm,
        physical_right_arm
    )

    log(
        "Physical left arm <-> "
        "physical right arm center distance: "
        f"{arm_distance:.2f} px"
    )

else:

    arm_distance = None

    log(
        "Physical left arm <-> "
        "physical right arm center distance: N/A"
    )


# ------------------------------------------------------------
# SEAT TO BACK
# ------------------------------------------------------------

if seat and back:

    seat_back_distance = center_distance(
        seat,
        back
    )

    log(
        "Seat <-> Back cushion center distance: "
        f"{seat_back_distance:.2f} px"
    )

else:

    seat_back_distance = None

    log(
        "Seat <-> Back cushion center distance: N/A"
    )


# ------------------------------------------------------------
# LEFT ARM TO SEAT
# ------------------------------------------------------------

if physical_left_arm and seat:

    left_arm_seat_distance = center_distance(
        physical_left_arm,
        seat
    )

    log(
        "Physical left arm <-> Seat "
        "center distance: "
        f"{left_arm_seat_distance:.2f} px"
    )

else:

    left_arm_seat_distance = None

    log(
        "Physical left arm <-> Seat "
        "center distance: N/A"
    )


# ------------------------------------------------------------
# RIGHT ARM TO SEAT
# ------------------------------------------------------------

if physical_right_arm and seat:

    right_arm_seat_distance = center_distance(
        physical_right_arm,
        seat
    )

    log(
        "Physical right arm <-> Seat "
        "center distance: "
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


    overall_width = (
        max_x - min_x
    )

    overall_height = (
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


# ------------------------------------------------------------
# SAFE VALUE VARIABLES
# ------------------------------------------------------------

overall_width_value = (
    float(overall_width)
    if overall_width is not None
    else None
)


overall_height_value = (
    float(overall_height)
    if overall_height is not None
    else None
)


# ------------------------------------------------------------
# REPORT
# ------------------------------------------------------------

log(
    f"Overall width: "
    f"{format_value(overall_width_value)} px"
)


log(
    f"Overall height: "
    f"{format_value(overall_height_value)} px"
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


component_class_names = [

    "back_cushion",

    "base",

    "left_arm",

    "legs",

    "right_arm",

    "seat_cushion",
]


component_counts = {}


for class_name in component_class_names:

    count = len(
        get_components(
            geometry_components,
            class_name
        )
    )

    component_counts[class_name] = count

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

    seat_confidence = seat["confidence"]

    x, y, w, h = seat["bbox"]

    seat_width = w

    seat_height = h

    seat_area = seat["area"]

    seat_aspect_ratio = seat["aspect_ratio"]


    concerns = []


    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    if seat_confidence < GEOMETRY_CONFIDENCE:

        concerns.append(
            "low confidence"
        )


    # --------------------------------------------------------
    # HEIGHT
    # --------------------------------------------------------

    if seat_height < 30:

        concerns.append(
            "very small seat height"
        )


    # --------------------------------------------------------
    # ASPECT RATIO
    # --------------------------------------------------------

    if seat_aspect_ratio > 10:

        concerns.append(
            "extreme aspect ratio"
        )


    # --------------------------------------------------------
    # AREA
    # --------------------------------------------------------

    if seat_area < 500:

        concerns.append(
            "small mask area"
        )


    # --------------------------------------------------------
    # QUALITY
    # --------------------------------------------------------

    if len(concerns) == 0:

        seat_quality = "GOOD"

    elif len(concerns) == 1:

        seat_quality = "WEAK"

    else:

        seat_quality = "POOR"


    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

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
            + ", ".join(concerns)
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


if seat:

    log(
        "Detected seat bounding-box width: "
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
    # SEAT WIDTH / OVERALL WIDTH
    # --------------------------------------------------------

    if overall_width is not None and overall_width != 0:

        seat_width_ratio = (
            seat_width /
            overall_width
        )


        log(
            "Seat bounding-box / "
            "overall width: "
            f"{seat_width_ratio:.4f}"
        )

    else:

        log(
            "Seat bounding-box / "
            "overall width: N/A"
        )


    # --------------------------------------------------------
    # SEAT AREA / BACK AREA
    # --------------------------------------------------------

    if back:

        seat_back_area_ratio = safe_ratio(
            seat["area"],
            back["area"]
        )


        if seat_back_area_ratio is not None:

            log(
                "Seat / Back area ratio: "
                f"{seat_back_area_ratio:.4f}"
            )

        else:

            log(
                "Seat / Back area ratio: N/A"
            )

    else:

        log(
            "Seat / Back area ratio: N/A"
        )


    # --------------------------------------------------------
    # LEFT / RIGHT ARM AREA
    # --------------------------------------------------------

    if (
        physical_left_arm
        and physical_right_arm
    ):

        left_right_arm_area_ratio = safe_ratio(
            physical_left_arm["area"],
            physical_right_arm["area"]
        )


        if left_right_arm_area_ratio is not None:

            log(
                "Physical left / physical right "
                "arm area ratio: "
                f"{left_right_arm_area_ratio:.4f}"
            )

        else:

            log(
                "Physical left / physical right "
                "arm area ratio: N/A"
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

physical_left_inner_edge = None

physical_right_inner_edge = None


if (
    physical_left_arm
    and physical_right_arm
):

    # --------------------------------------------------------
    # ARM CENTER-TO-CENTER SPAN
    # --------------------------------------------------------

    arm_center_span = center_distance(
        physical_left_arm,
        physical_right_arm
    )


    # --------------------------------------------------------
    # ARM INNER EDGES
    # --------------------------------------------------------

    left_x = physical_left_arm["bbox"][0]

    left_w = physical_left_arm["bbox"][2]

    right_x = physical_right_arm["bbox"][0]


    physical_left_inner_edge = (
        left_x +
        left_w
    )


    physical_right_inner_edge = right_x


    # --------------------------------------------------------
    # INNER SPAN
    # --------------------------------------------------------

    arm_inner_span = max(
        0,
        physical_right_inner_edge -
        physical_left_inner_edge
    )


    log(
        "Arm center-to-center span: "
        f"{arm_center_span:.2f} px"
    )


    log(
        "Physical left arm inner edge: "
        f"{physical_left_inner_edge} px"
    )


    log(
        "Physical right arm inner edge: "
        f"{physical_right_inner_edge} px"
    )


    log(
        "Usable arm-to-arm inner span: "
        f"{arm_inner_span:.2f} px"
    )


else:

    log(
        "Arm span unavailable."
    )


# ============================================================
# USABLE SEAT SPAN
# ============================================================

log("")
log("==============================================")
log("USABLE SEAT SPAN")
log("==============================================")


seat_bbox_span = None

usable_seat_span = None

usable_seat_span_ratio = None

seat_bbox_to_arm_inner_span_ratio = None


if seat:

    seat_x = seat["bbox"][0]

    seat_w = seat["bbox"][2]


    seat_bbox_left = seat_x

    seat_bbox_right = (
        seat_x +
        seat_w
    )


    seat_bbox_span = seat_w


    log(
        "Detected seat bounding-box span: "
        f"{seat_bbox_span:.2f} px"
    )


    if (
        physical_left_inner_edge is not None
        and physical_right_inner_edge is not None
    ):

        # ----------------------------------------------------
        # Calculate overlap between the detected seat bbox
        # and the physical arm-to-arm inner region.
        # ----------------------------------------------------

        usable_left = max(
            seat_bbox_left,
            physical_left_inner_edge
        )


        usable_right = min(
            seat_bbox_right,
            physical_right_inner_edge
        )


        usable_seat_span = max(
            0,
            usable_right -
            usable_left
        )


        # ----------------------------------------------------
        # Ratios
        # ----------------------------------------------------

        if arm_inner_span is not None:

            usable_seat_span_ratio = safe_ratio(
                usable_seat_span,
                arm_inner_span
            )


            # Important feature:
            # detected seat width compared with
            # arm-to-arm inner span.

            seat_bbox_to_arm_inner_span_ratio = safe_ratio(
                seat_bbox_span,
                arm_inner_span
            )


        log(
            "Usable seat span between arm "
            "inner edges: "
            f"{usable_seat_span:.2f} px"
        )


        log(
            "Seat bbox / arm inner span ratio: "
            f"{format_value(seat_bbox_to_arm_inner_span_ratio)}"
        )


        # ----------------------------------------------------
        # DRAW ARM INNER SPAN
        # ----------------------------------------------------

        line_y = min(
            height - 30,
            max(
                30,
                seat["bbox"][1] +
                seat["bbox"][3] +
                20
            )
        )


        # ----------------------------------------------------
        # MAIN SPAN LINE
        # ----------------------------------------------------

        cv2.line(
            visualization,
            (
                int(physical_left_inner_edge),
                int(line_y)
            ),
            (
                int(physical_right_inner_edge),
                int(line_y)
            ),
            (0, 0, 255),
            3
        )


        # ----------------------------------------------------
        # LEFT END MARKER
        # ----------------------------------------------------

        cv2.line(
            visualization,
            (
                int(physical_left_inner_edge),
                int(line_y - 10)
            ),
            (
                int(physical_left_inner_edge),
                int(line_y + 10)
            ),
            (0, 0, 255),
            2
        )


        # ----------------------------------------------------
        # RIGHT END MARKER
        # ----------------------------------------------------

        cv2.line(
            visualization,
            (
                int(physical_right_inner_edge),
                int(line_y - 10)
            ),
            (
                int(physical_right_inner_edge),
                int(line_y + 10)
            ),
            (0, 0, 255),
            2
        )


        # ----------------------------------------------------
        # SPAN LABEL
        # ----------------------------------------------------

        span_label = (
            "ARM INNER SPAN: "
            f"{arm_inner_span:.0f} px"
        )


        (
            label_width,
            label_height
        ), _ = cv2.getTextSize(
            span_label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            2
        )


        label_x = int(
            (
                physical_left_inner_edge +
                physical_right_inner_edge
            ) / 2
            -
            label_width / 2
        )


        label_x = max(
            5,
            min(
                label_x,
                width - label_width - 5
            )
        )


        label_y = max(
            20,
            int(line_y - 8)
        )


        draw_text_with_background(
            visualization,
            span_label,
            (
                label_x,
                label_y
            ),
            (0, 0, 255),
            font_scale=0.50,
            thickness=2
        )


    else:

        log(
            "Usable seat span unavailable "
            "because both arm inner edges "
            "are not available."
        )


else:

    log(
        "Detected seat bounding-box span: N/A"
    )


    log(
        "Usable seat span: N/A"
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
        "No seat cushion component detected."
    )


elif seat_cushion_count == 1:

    log(
        "1 seat cushion component detected."
    )


else:

    log(
        f"{seat_cushion_count} seat cushion "
        "components detected."
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


elif (
    seat_bbox_to_arm_inner_span_ratio is None
    or arm_inner_span is None
):

    log(
        "Seater analysis: UNDETERMINED"
    )

    log(
        "Reason: insufficient arm/seat geometry."
    )


else:

    log(
        "Seat geometry is strong enough "
        "for preliminary analysis."
    )


    log(
        "Seat bbox / arm inner span ratio: "
        f"{seat_bbox_to_arm_inner_span_ratio:.4f}"
    )


    log(
        "Seat aspect ratio: "
        f"{seat_aspect_ratio:.2f}"
    )


    log(
        "Seat cushion count: "
        f"{seat_cushion_count}"
    )


    # --------------------------------------------------------
    # GEOMETRY INTERPRETATION
    # --------------------------------------------------------

    if (
        seat_bbox_to_arm_inner_span_ratio
        >= 1.30
    ):

        geometry_interpretation = (
            "seat bbox extends substantially "
            "across the arm-to-arm region"
        )


    elif (
        seat_bbox_to_arm_inner_span_ratio
        >= 1.00
    ):

        geometry_interpretation = (
            "seat bbox spans approximately "
            "the arm-to-arm seating region"
        )


    else:

        geometry_interpretation = (
            "seat bbox is narrower than "
            "the arm-to-arm seating region"
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
        usable_seat_span_ratio,

    "seat_bbox_to_arm_inner_span_ratio":
        seat_bbox_to_arm_inner_span_ratio,

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
    "on horizontal image position, not only YOLO "
    "class name."
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


if not success:

    log(
        "ERROR: Could not save visualization."
    )

else:

    log(
        "Visualization saved successfully."
    )


# ============================================================
# SAVE TEXT REPORT
# ============================================================

save_text_report()


# ============================================================
# FINAL MESSAGE
# ============================================================

print("")

print(
    "Analysis complete."
)

print(
    f"Text report saved to: {TEXT_OUTPUT}"
)

print(
    f"Visualization saved to: {IMAGE_OUTPUT}"
)