from pathlib import Path

LABEL_DIR = Path("dataset/cleaned/train/labels")

FILES = [
    "one_seater_v1i_yolov8__1327_jpg.rf.677804ed1a32f615b5cd73b7afa8138b.txt",
    "two_seater_v1i_yolov8__1324_jpg.rf.af2af3806ecf764c64f52f507b486684.txt",
    "two_seater_v1i_yolov8__1429_jpg.rf.15b2495c3a50a3e2afa91f423bba2c35.txt",
    "two_seater_v1i_yolov8__1_jpg.rf.ef062d26994470bb8f3ae56c3b853680.txt",
]

for filename in FILES:
    path = LABEL_DIR / filename

    text = path.read_text(encoding="utf-8")

    # The corrupted files contain literal "\n" inside a line.
    text = text.replace("\\n", "\n")

    path.write_text(text, encoding="utf-8")

    print(f"Repaired: {filename}")

print()
print("4 label files repaired.")