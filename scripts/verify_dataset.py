from pathlib import Path

DATASET = Path("dataset/cleaned")

VALID_CLASSES = set(range(6))

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def verify_split(split_name):

    image_dir = DATASET / split_name / "images"
    label_dir = DATASET / split_name / "labels"

    print()
    print("=" * 60)
    print(f"VERIFYING {split_name.upper()} DATASET")
    print("=" * 60)

    errors = 0
    checked_images = 0
    checked_labels = 0
    checked_polygons = 0

    images = {
        p.stem: p
        for p in image_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    }

    labels = {
        p.stem: p
        for p in label_dir.iterdir()
        if p.suffix.lower() == ".txt"
    }

    # Image -> Label
    for stem in images:
        if stem not in labels:
            print(f"ERROR: Missing label for image: {images[stem].name}")
            errors += 1

    # Label -> Image
    for stem in labels:
        if stem not in images:
            print(f"ERROR: Missing image for label: {labels[stem].name}")
            errors += 1

    # Validate labels
    for stem, label_path in labels.items():

        checked_labels += 1

        try:
            lines = label_path.read_text(encoding="utf-8").splitlines()
        except Exception as e:
            print(f"ERROR: Cannot read {label_path.name}: {e}")
            errors += 1
            continue

        for line_number, line in enumerate(lines, start=1):

            line = line.strip()

            # Empty line is ignored
            if not line:
                continue

            parts = line.split()

            # Class ID + at least 3 coordinate pairs
            if len(parts) < 7:
                print(
                    f"ERROR: Too few polygon values: "
                    f"{label_path.name}, line {line_number}"
                )
                errors += 1
                continue

            # Class ID
            try:
                class_id = int(parts[0])
            except ValueError:
                print(
                    f"ERROR: Invalid class ID '{parts[0]}': "
                    f"{label_path.name}, line {line_number}"
                )
                errors += 1
                continue

            if class_id not in VALID_CLASSES:
                print(
                    f"ERROR: Invalid class {class_id}: "
                    f"{label_path.name}, line {line_number}"
                )
                errors += 1

            # Coordinates
            coordinates = parts[1:]

            if len(coordinates) % 2 != 0:
                print(
                    f"ERROR: Odd number of coordinates "
                    f"({len(coordinates)}): "
                    f"{label_path.name}, line {line_number}"
                )
                errors += 1
                continue

            coordinate_error = False

            for index, value in enumerate(coordinates, start=1):

                try:
                    coordinate = float(value)
                except ValueError:
                    print(
                        f"ERROR: Non-numeric value '{value}' "
                        f"at coordinate {index}: "
                        f"{label_path.name}, line {line_number}"
                    )
                    errors += 1
                    coordinate_error = True
                    break

                if not 0.0 <= coordinate <= 1.0:
                    print(
                        f"ERROR: Coordinate {coordinate} outside 0-1: "
                        f"{label_path.name}, line {line_number}"
                    )
                    errors += 1
                    coordinate_error = True
                    break

            if not coordinate_error:
                checked_polygons += 1

    checked_images = len(images)

    print()
    print(f"Images checked    : {checked_images}")
    print(f"Labels checked    : {checked_labels}")
    print(f"Polygons checked  : {checked_polygons}")
    print(f"Errors found      : {errors}")

    return errors


def main():

    print("=" * 60)
    print("SOFA SEGMENTATION DATASET VALIDATION")
    print("=" * 60)

    train_errors = verify_split("train")
    valid_errors = verify_split("valid")

    total_errors = train_errors + valid_errors

    print()
    print("=" * 60)

    if total_errors == 0:
        print("DATASET VALIDATION PASSED")
        print("All annotations appear valid.")
    else:
        print("DATASET VALIDATION FAILED")
        print(f"Total errors: {total_errors}")

    print("=" * 60)


if __name__ == "__main__":
    main()