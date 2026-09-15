from pathlib import Path
import shutil

# ============================================================
# SOFA DATASET CLEANER
# Original 9 classes -> Clean 6 component classes
# ============================================================

SOURCE = Path("dataset/original/extracted")
OUTPUT = Path("dataset/cleaned")

# Classes we want to keep
CLASS_NAMES = {
    0: "back_cushion",
    1: "base",
    2: "left_arm",
    3: "legs",
    4: "right_arm",
    5: "seat_cushion",
}

# Classes we want to remove
REMOVE_CLASSES = {6, 7, 8}


def process_split(split_name):

    source_images = SOURCE / split_name / "images"
    source_labels = SOURCE / split_name / "labels"

    output_images = OUTPUT / split_name / "images"
    output_labels = OUTPUT / split_name / "labels"

    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    image_count = 0
    modified_count = 0
    empty_count = 0

    for image_path in source_images.iterdir():

        if not image_path.is_file():
            continue

        label_path = source_labels / f"{image_path.stem}.txt"

        if not label_path.exists():
            print(f"WARNING: Missing label: {image_path.name}")
            continue

        original_lines = label_path.read_text().splitlines()

        new_lines = []

        for line in original_lines:

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            try:
                class_id = int(parts[0])
            except ValueError:
                print(f"WARNING: Invalid class ID: {label_path}")
                continue

            # Keep only classes 0-5
            if class_id in CLASS_NAMES:

                # Class IDs 0-5 already have the correct numbering
                new_lines.append(line)

            # Classes 6,7,8 are intentionally ignored

        # Copy image
        shutil.copy2(
            image_path,
            output_images / image_path.name
        )

        # Write cleaned annotation
        output_label = output_labels / label_path.name

        output_label.write_text(
            "\n".join(new_lines) +
            ("\n" if new_lines else "")
        )

        image_count += 1

        if len(new_lines) != len(original_lines):
            modified_count += 1

        if len(new_lines) == 0:
            empty_count += 1

    print()
    print("=" * 50)
    print(f"{split_name.upper()} DATASET")
    print("=" * 50)
    print(f"Images copied       : {image_count}")
    print(f"Labels modified     : {modified_count}")
    print(f"Empty labels        : {empty_count}")


def create_yaml():

    yaml_content = """path: C:/projects/Sofa_Project_Model_Training/dataset/cleaned

train: train/images
val: valid/images

nc: 6

names:
  0: back_cushion
  1: base
  2: left_arm
  3: legs
  4: right_arm
  5: seat_cushion
"""

    yaml_path = OUTPUT / "data.yaml"

    yaml_path.write_text(yaml_content)

    print()
    print("Created:")
    print(yaml_path)


def main():

    print("=" * 60)
    print("SOFA DATASET CLEANING")
    print("9 CLASSES -> 6 COMPONENT CLASSES")
    print("=" * 60)

    print()
    print("KEEPING:")
    for class_id, name in CLASS_NAMES.items():
        print(f"  {class_id} -> {name}")

    print()
    print("REMOVING:")
    print("  6 -> one-seater")
    print("  7 -> two-seater")
    print("  8 -> stallion-sofa")

    print()
    print("Original dataset:")
    print(SOURCE)

    print()
    print("Clean dataset:")
    print(OUTPUT)

    # Process train
    process_split("train")

    # Process validation
    process_split("valid")

    # Create YAML
    create_yaml()

    print()
    print("=" * 60)
    print("CLEANING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()