import argparse
import json
import re
import shutil
from pathlib import Path


CASE_RE = re.compile(r"^(CTV_\d+)(?:_0000)?\.nii\.gz$")


def case_id_from_name(path: Path) -> str:
    match = CASE_RE.match(path.name)
    if match is None:
        raise ValueError(f"Unexpected case filename: {path}")
    return match.group(1)


def collect_fold_validation_masks(results_dir: Path) -> dict:
    masks = {}
    duplicates = []
    for fold in range(5):
        validation_dir = results_dir / f"fold_{fold}" / "validation"
        if not validation_dir.is_dir():
            raise FileNotFoundError(f"Missing validation directory: {validation_dir}")
        for mask_file in sorted(validation_dir.glob("*.nii.gz")):
            case_id = case_id_from_name(mask_file)
            if case_id in masks:
                duplicates.append((case_id, masks[case_id], mask_file))
            masks[case_id] = mask_file
    if duplicates:
        details = "\n".join(f"{case}: {old} and {new}" for case, old, new in duplicates)
        raise RuntimeError(f"Duplicate validation predictions across folds:\n{details}")
    return masks


def collect_prediction_masks(pred_dir: Path) -> dict:
    if not pred_dir.is_dir():
        raise FileNotFoundError(f"Missing prediction directory: {pred_dir}")
    masks = {}
    for mask_file in sorted(pred_dir.glob("*.nii.gz")):
        case_id = case_id_from_name(mask_file)
        if case_id in masks:
            raise RuntimeError(f"Duplicate prediction for {case_id}: {masks[case_id]} and {mask_file}")
        masks[case_id] = mask_file
    return masks


def collect_images(image_dir: Path) -> dict:
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")
    images = {}
    for image_file in sorted(image_dir.glob("*_0000.nii.gz")):
        case_id = case_id_from_name(image_file)
        if case_id in images:
            raise RuntimeError(f"Duplicate image for {case_id}: {images[case_id]} and {image_file}")
        images[case_id] = image_file
    return images


def copy_initial_masks(images: dict, masks: dict, image_dir: Path, overwrite: bool) -> list:
    missing = sorted(set(images) - set(masks))
    extra = sorted(set(masks) - set(images))
    if missing:
        raise RuntimeError(f"Missing prediction masks for image cases: {missing[:20]} (total={len(missing)})")
    if extra:
        raise RuntimeError(f"Prediction masks without matching images: {extra[:20]} (total={len(extra)})")

    written = []
    for case_id in sorted(images):
        dst = image_dir / f"{case_id}_0001.nii.gz"
        if dst.exists() and not overwrite:
            raise FileExistsError(f"Destination already exists, use --overwrite to replace: {dst}")
        shutil.copy2(masks[case_id], dst)
        written.append(dst)
    return written


def update_dataset_json(dataset_json: Path) -> None:
    if not dataset_json.is_file():
        print(f"dataset.json not found, skip update: {dataset_json}")
        return
    with dataset_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    channel_names = data.get("channel_names", data.get("modality", {}))
    if not isinstance(channel_names, dict):
        raise RuntimeError(f"Unsupported channel_names/modality format in {dataset_json}")
    channel_names["0"] = channel_names.get("0", "CT")
    channel_names["1"] = channel_names.get("1", "initial_mask")
    if "channel_names" in data:
        data["channel_names"] = channel_names
    else:
        data["modality"] = channel_names

    with dataset_json.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy v2-1 validation/test predictions as Dataset005 _0001 initial-mask channels."
    )
    parser.add_argument(
        "--results-dir",
        default="/home/wusi/nnUNet_SAM2/nnUNetFrame/DATASET/nnUNet_results/"
        "Dataset003_Rectal146pAll_v2-1/nnUNetTrainerSAM2__nnUNetPlans__3d_fullres",
        type=Path,
    )
    parser.add_argument(
        "--dataset-dir",
        default="/home/wusi/nnUNet_SAM2/nnUNetFrame/DATASET/nnUNet_raw/Dataset005_RectalAllPrompt",
        type=Path,
    )
    parser.add_argument("--test-pred-dirname", default="testresults_fold3")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-dataset-json", action="store_true")
    args = parser.parse_args()

    images_tr = args.dataset_dir / "imagesTr"
    images_ts = args.dataset_dir / "imagesTs"

    train_images = collect_images(images_tr)
    train_masks = collect_fold_validation_masks(args.results_dir)
    train_written = copy_initial_masks(train_images, train_masks, images_tr, args.overwrite)
    print(f"imagesTr: copied {len(train_written)} initial-mask channels to *_0001.nii.gz")

    test_images = collect_images(images_ts)
    test_masks = collect_prediction_masks(args.results_dir / args.test_pred_dirname)
    test_written = copy_initial_masks(test_images, test_masks, images_ts, args.overwrite)
    print(f"imagesTs: copied {len(test_written)} initial-mask channels to *_0001.nii.gz")

    if not args.skip_dataset_json:
        update_dataset_json(args.dataset_dir / "dataset.json")
        print("dataset.json: ensured channel 1 is initial_mask")


if __name__ == "__main__":
    main()
