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

# Minimum confidence for displaying YOLO predictions
PREDICTION_CONFIDENCE = 0.25

# Minimum confidence for geometry calculations
GEOMETRY_CONFIDENCE = 0.50

# Minimum contour area
MIN_COMPONENT_AREA = 100


# ============================================================
# REPORT STORAGE
# ============================================================

report_lines = []


def log(text=""):
    print(text)
    report_lines.append(str(text))


def save_text_report():
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
    return [
        c for c in components
        if c["class_name"] == class_name
    ]


def get_largest_component(components, class_name):
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

    if b is None or b == 0:
        return None

    return float(a / b)


def format_value(value):

    if value is None:
        return "N/A"

    if isinstance(value, float):
        return f"{value:.4f}"

    return str(value)


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

    log("Model loaded successfully")
    log(f"Classes: {model.names}")

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

    raise FileNotFoundError(image_path)


# ============================================================
# READ IMAGE
# ============================================================

image = cv2.imread(image_path)

if image is None:

    log("")
    log("ERROR: Could not read image.")

    save_text_report()

    raise ValueError(
        "OpenCV could not read the image."
    )


height, width = image.shape[:2]

log("")
log(f"Image path: {image_path}")
log(f"Image size: {width} x {height}")


# ============================================================
# YOLO SEGMENTATION
# ============================================================

results = model.predict(
    source=image_path,
    conf=PREDICTION_CONFIDENCE,
    verbose=False
)

result = results[0]


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
boxes = result.boxes


log("")
log(f"Total masks detected: {len(masks)}")


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
        f"{class_name} "
        f"confidence={confidence:.3f}"
    )


# ============================================================
# SEGMENTATION + CONTOUR GEOMETRY
# ============================================================

log("")
log("==============================================")
log("COMPONENT GEOMETRY")
log("==============================================")


components = []

visualization = image.copy()


# Visualization colors by class
class_colors = {
    0: (255, 0, 0),      # back_cushion
    1: (0, 255, 0),      # base
    2: (255, 0, 255),    # left_arm
    3: (0, 255, 255),    # legs
    4: (0, 0, 255),      # right_arm
    5: (255, 255, 0),    # seat_cushion
}


for i in range(len(masks)):

    class_id = int(
        boxes.cls[i].item()
    )

    confidence = float(
        boxes.conf[i].item()
    )

    class_name = model.names[class_id]

    # --------------------------------------------------------
    # Display threshold
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
    ).astype(np.uint8) * 255

    # --------------------------------------------------------
    # Contours
    # --------------------------------------------------------

    contours, _ = cv2.findContours(
        binary_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        continue

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

        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]

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
        "index": i,
        "class_id": class_id,
        "class_name": class_name,
        "confidence": confidence,
        "area": float(area),
        "bbox": (x, y, w, h),
        "aspect_ratio": aspect_ratio,
        "perimeter": float(perimeter),
        "center": center,
        "contour": contour,
    }

    components.append(
        component
    )

    # --------------------------------------------------------
    # Visualization
    # --------------------------------------------------------

    color = class_colors.get(
        class_id,
        (255, 255, 255)
    )

    mask_pixels = (
        binary_mask > 0
    )

    overlay = visualization.copy()

    overlay[
        mask_pixels
    ] = color

    visualization = cv2.addWeighted(
        visualization,
        0.65,
        overlay,
        0.35,
        0
    )

    # Contour
    cv2.drawContours(
        visualization,
        [contour],
        -1,
        color,
        2
    )

    # Bounding box
    cv2.rectangle(
        visualization,
        (x, y),
        (x + w, y + h),
        color,
        2
    )

    # Center
    cv2.circle(
        visualization,
        (int(cx), int(cy)),
        5,
        color,
        -1
    )

    # Label
    label = (
        f"{class_name} "
        f"{confidence:.2f}"
    )

    label_y = max(
        y - 5,
        15
    )

    cv2.putText(
        visualization,
        label,
        (x, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA
    )


# ============================================================
# PRINT ALL COMPONENT GEOMETRY
# ============================================================

for component in components:

    x, y, w, h = component["bbox"]
    cx, cy = component["center"]

    log("")

    log(
        f"{component['class_name']} | "
        f"confidence={component['confidence']:.3f} | "
        f"area={component['area']:.1f} | "
        f"bbox=({x},{y},{w},{h}) | "
        f"AR={component['aspect_ratio']:.2f} | "
        f"perimeter={component['perimeter']:.1f} | "
        f"center=({cx:.1f},{cy:.1f})"
    )


# ============================================================
# SAVE VISUALIZATION
# ============================================================

cv2.imwrite(
    IMAGE_OUTPUT,
    visualization
)


log("")
log("==============================================")
log("GEOMETRY VISUALIZATION COMPLETE")
log("==============================================")

log(
    f"Components visualized: {len(components)}"
)

log("")
log("Saved result to:")
log(IMAGE_OUTPUT)


# ============================================================
# GEOMETRY-VALID COMPONENTS
# ============================================================

geometry_components = [
    component
    for component in components
    if component["confidence"] >= GEOMETRY_CONFIDENCE
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
# PHYSICAL ARM ORDER
# ============================================================
#
# We determine left/right using image position.
# We do not blindly trust the class name for geometry.
# ============================================================

all_arms = (
    left_arm_candidates +
    right_arm_candidates
)

all_arms = sorted(
    all_arms,
    key=lambda c: c["center"][0]
)


physical_left_arm = None
physical_right_arm = None


if len(all_arms) >= 2:

    physical_left_arm = all_arms[0]
    physical_right_arm = all_arms[-1]

elif len(all_arms) == 1:

    if all_arms[0]["center"][0] < width / 2:

        physical_left_arm = all_arms[0]

    else:

        physical_right_arm = all_arms[0]


# ============================================================
# COMPONENT RELATIONSHIPS
# ============================================================

log("")
log("==============================================")
log("COMPONENT RELATIONSHIPS")
log("==============================================")


if (
    physical_left_arm and
    physical_right_arm
):

    arm_distance = center_distance(
        physical_left_arm,
        physical_right_arm
    )

    log(
        f"Left arm ↔ Right arm: "
        f"{arm_distance:.2f} px"
    )

else:

    arm_distance = None

    log(
        "Left arm ↔ Right arm: N/A"
    )


if seat and back:

    seat_back_distance = center_distance(
        seat,
        back
    )

    log(
        f"Seat ↔ Back cushion: "
        f"{seat_back_distance:.2f} px"
    )

else:

    seat_back_distance = None

    log(
        "Seat ↔ Back cushion: N/A"
    )


if physical_left_arm and seat:

    left_arm_seat_distance = center_distance(
        physical_left_arm,
        seat
    )

    log(
        f"Left arm ↔ Seat: "
        f"{left_arm_seat_distance:.2f} px"
    )

else:

    left_arm_seat_distance = None

    log(
        "Left arm ↔ Seat: N/A"
    )


if physical_right_arm and seat:

    right_arm_seat_distance = center_distance(
        physical_right_arm,
        seat
    )

    log(
        f"Right arm ↔ Seat: "
        f"{right_arm_seat_distance:.2f} px"
    )

else:

    right_arm_seat_distance = None

    log(
        "Right arm ↔ Seat: N/A"
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

    overall_width = max_x - min_x

    overall_height = max_y - min_y

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

    seat_confidence = seat["confidence"]

    x, y, w, h = seat["bbox"]

    seat_width = w
    seat_height = h
    seat_area = seat["area"]
    seat_aspect_ratio = seat["aspect_ratio"]

    concerns = []

    if seat_confidence < 0.50:
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
        f"{seat_area:.1f} px²"
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
        f"Seat width: "
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

    if overall_width:

        seat_width_ratio = (
            seat_width /
            overall_width
        )

        log(
            f"Seat / overall width: "
            f"{seat_width_ratio:.2f}"
        )

    else:

        log(
            "Seat / overall width: N/A"
        )

    if back:

        seat_back_area_ratio = safe_ratio(
            seat["area"],
            back["area"]
        )

        log(
            f"Seat / Back area ratio: "
            f"{seat_back_area_ratio:.2f}"
        )

    else:

        log(
            "Seat / Back area ratio: N/A"
        )

    if (
        physical_left_arm and
        physical_right_arm
    ):

        left_right_arm_area_ratio = safe_ratio(
            physical_left_arm["area"],
            physical_right_arm["area"]
        )

        log(
            f"Left / Right arm area ratio: "
            f"{left_right_arm_area_ratio:.2f}"
        )

    else:

        log(
            "Left / Right arm area ratio: N/A"
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
bbox_inner_arm_span = None


if (
    physical_left_arm and
    physical_right_arm
):

    arm_center_span = center_distance(
        physical_left_arm,
        physical_right_arm
    )

    log(
        f"Arm center-to-center span: "
        f"{arm_center_span:.2f} px"
    )

    left_x = physical_left_arm["bbox"][0]
    left_w = physical_left_arm["bbox"][2]

    right_x = physical_right_arm["bbox"][0]

    left_inner_edge = (
        left_x +
        left_w
    )

    right_inner_edge = right_x

    bbox_inner_arm_span = max(
        0,
        right_inner_edge -
        left_inner_edge
    )

    log(
        f"Arm bounding-box inner span: "
        f"{bbox_inner_arm_span:.2f} px"
    )

else:

    log(
        "Arm span unavailable."
    )


# ============================================================
# SEAT SPAN RELATIONSHIP
# ============================================================

log("")
log("==============================================")
log("SEAT SPAN RELATIONSHIP")
log("==============================================")


seat_span = None
seat_span_ratio = None


if seat:

    # Seat bounding-box width
    seat_span = seat["bbox"][2]

    log(
        f"Seat span: "
        f"{seat_span} px"
    )

else:

    log(
        "Seat span: N/A"
    )


if (
    seat_span is not None and
    arm_center_span is not None
):

    seat_span_ratio = safe_ratio(
        seat_span,
        arm_center_span
    )

    log(
        f"Arm center span: "
        f"{arm_center_span:.2f} px"
    )

    log(
        f"Seat span / arm center span: "
        f"{seat_span_ratio:.2f}"
    )

else:

    log(
        "Seat span / arm center span: N/A"
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
        "1 continuous seat cushion detected."
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

elif (
    seat_span_ratio is None or
    arm_center_span is None
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
        f"Seat span ratio: "
        f"{seat_span_ratio:.2f}"
    )

    log(
        f"Seat aspect ratio: "
        f"{seat_aspect_ratio:.2f}"
    )

    log(
        f"Seat cushion count: "
        f"{seat_cushion_count}"
    )

    if seat_span_ratio >= 0.80:

        geometry_interpretation = (
            "long continuous seating span"
        )

    elif seat_span_ratio >= 0.55:

        geometry_interpretation = (
            "moderate seating span"
        )

    else:

        geometry_interpretation = (
            "short seating span"
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

    "seat_width_px":
        seat_width,

    "seat_height_px":
        seat_height,

    "seat_area_px2":
        seat_area,

    "seat_aspect_ratio":
        seat_aspect_ratio,

    "seat_width_ratio":
        seat_width_ratio,

    "arm_span_px":
        arm_center_span,

    "bbox_inner_arm_span_px":
        bbox_inner_arm_span,

    "seat_span_ratio":
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
        f"{key} = {format_value(value)}"
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
    "A single continuous cushion does not determine "
    "seating capacity."
)

log(
    "Perspective and sofa design can change measured "
    "geometry."
)

log(
    "Seater classification is an image-based inference."
)

log(
    "The current seater interpretation is deliberately "
    "not a final calibrated classifier."
)

log(
    "Low-confidence YOLO detections are displayed "
    "for debugging but excluded from geometry when "
    f"confidence < {GEOMETRY_CONFIDENCE:.2f}."
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
# SAVE TEXT REPORT
# ============================================================

save_text_report()


print("")
print("Analysis complete.")
print(
    f"Text report saved to: {TEXT_OUTPUT}"
)
print(
    f"Visualization saved to: {IMAGE_OUTPUT}"
)