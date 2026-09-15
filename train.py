from ultralytics import YOLO
from pathlib import Path

# ==============================
# SETTINGS
# ==============================

DATA = "dataset/cleaned/data.yaml"

PROJECT = "models"
NAME = "sofa_yolo11n_seg_v1"

MAX_EPOCHS = 100

CHECKPOINT = Path(PROJECT) / NAME / "weights" / "last.pt"


# ==============================
# START / RESUME TRAINING
# ==============================

if CHECKPOINT.exists():

    print("\n========================================")
    print("RESUMING PREVIOUS SOFA MODEL TRAINING")
    print("========================================")
    print(f"Checkpoint: {CHECKPOINT}")
    print("Training will continue from the saved epoch.")
    print("Press CTRL+C whenever you want to stop.\n")

    model = YOLO(str(CHECKPOINT))

    model.train(
        resume=True
    )

else:

    print("\n========================================")
    print("STARTING NEW SOFA MODEL TRAINING")
    print("========================================")
    print(f"Maximum epochs: {MAX_EPOCHS}")
    print("Press CTRL+C whenever you want to stop.\n")

    model = YOLO("yolo11n-seg.pt")

    model.train(
        data=DATA,
        epochs=MAX_EPOCHS,
        imgsz=640,
        batch=4,
        device="cpu",
        workers=2,
        project=PROJECT,
        name=NAME
    )