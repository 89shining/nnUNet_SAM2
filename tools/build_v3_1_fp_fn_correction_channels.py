import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import SimpleITK as sitk


def compute_2d_dice_per_slice(
    initial_mask: np.ndarray,
    gt_mask: np.ndarray,
    gt_range_only: bool = True,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    initial = initial_mask > 0
    gt = gt_mask > 0
    if initial.shape != gt.shape:
        raise ValueError(f"Shape mismatch: initial={initial.shape}, gt={gt.shape}")

    dice = np.ones(initial.shape[0], dtype=np.float32)
    gt_area_z = gt.reshape(gt.shape[0], -1).sum(axis=1)
    fg_slices = np.where(gt_area_z > 0)[0]
    if fg_slices.size == 0:
        return dice, (-1, -1)

    z_min = int(fg_slices[0])
    z_max = int(fg_slices[-1])
    z_iter = range(z_min, z_max + 1) if gt_range_only else range(initial.shape[0])
    for z in z_iter:
        pred_z = initial[z]
        gt_z = gt[z]
        denom = float(pred_z.sum() + gt_z.sum())
        if denom <= 0:
            dice[z] = 1.0
            continue
        inter = float(np.logical_and(pred_z, gt_z).sum())
        dice[z] = float((2.0 * inter) / (denom + eps))
    return dice, (z_min, z_max)


def select_worst_dice_slices(initial_mask: np.ndarray, gt_mask: np.ndarray, top_k: int = 3) -> Tuple[List[int], np.ndarray, Tuple[int, int]]:
    dice, gt_range = compute_2d_dice_per_slice(initial_mask, gt_mask, gt_range_only=True)
    z_min, z_max = gt_range
    if z_min < 0:
        return [], dice, gt_range

    candidates = np.arange(z_min, z_max + 1)
    if candidates.size <= top_k:
        selected = candidates
    else:
        order = np.lexsort((candidates, dice[candidates]))
        selected = candidates[order[:top_k]]
    return [int(i) for i in selected.tolist()], dice, gt_range


def build_fp_fn_correction_prompts(
    initial_mask: np.ndarray,
    gt_mask: np.ndarray,
    selected_slices: List[int],
) -> Tuple[np.ndarray, np.ndarray]:
    initial = initial_mask > 0
    gt = gt_mask > 0
    pos_corr_mask = np.zeros_like(gt, dtype=np.uint8)
    neg_corr_mask = np.zeros_like(gt, dtype=np.uint8)
    for z in selected_slices:
        # FN: GT says CTV but initial prediction missed it. Positive correction asks v3-1 to add/keep it.
        pos_corr_mask[z] = np.logical_and(gt[z], np.logical_not(initial[z])).astype(np.uint8)
        # FP: initial prediction says CTV but GT says background. Negative correction asks v3-1 to remove it.
        neg_corr_mask[z] = np.logical_and(initial[z], np.logical_not(gt[z])).astype(np.uint8)
    return pos_corr_mask, neg_corr_mask


def build_v3_1_correction_mask_channels(
    ct: np.ndarray,
    initial_mask: np.ndarray,
    gt_mask: np.ndarray,
    top_k: int = 3,
):
    if ct.shape != initial_mask.shape or ct.shape != gt_mask.shape:
        raise ValueError(f"Shape mismatch: ct={ct.shape}, initial={initial_mask.shape}, gt={gt_mask.shape}")
    selected_slices, dice_per_slice, gt_range = select_worst_dice_slices(initial_mask, gt_mask, top_k=top_k)
    pos_corr_mask, neg_corr_mask = build_fp_fn_correction_prompts(initial_mask, gt_mask, selected_slices)
    return pos_corr_mask, neg_corr_mask, selected_slices, dice_per_slice, gt_range


def case_id_from_image(path: Path) -> str:
    name = path.name
    if not name.endswith("_0000.nii.gz"):
        raise ValueError(f"Expected *_0000.nii.gz image file, got {path}")
    return name[: -len("_0000.nii.gz")]


def read_array(path: Path):
    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(image)
    return image, array


def write_like(reference_image, array: np.ndarray, path: Path) -> None:
    out = sitk.GetImageFromArray(array.astype(np.uint8, copy=False))
    out.CopyInformation(reference_image)
    sitk.WriteImage(out, str(path), True)


def process_split(dataset_dir: Path, split: str, top_k: int, overwrite: bool, require_labels: bool = True) -> List[dict]:
    image_dir = dataset_dir / f"images{split}"
    label_dir = dataset_dir / f"labels{split}"
    if not image_dir.is_dir():
        print(f"Skip {split}: missing {image_dir}")
        return []
    if not label_dir.is_dir():
        message = (
            f"Missing {label_dir}. v3-1 uses GT for oracle simulated correction prompts, "
            "so train and test construction must both have GT labels."
        )
        if require_labels:
            raise FileNotFoundError(message)
        print(f"Skip {split}: {message}")
        return []

    records = []
    for ct_path in sorted(image_dir.glob("*_0000.nii.gz")):
        case_id = case_id_from_image(ct_path)
        initial_path = image_dir / f"{case_id}_0001.nii.gz"
        gt_path = label_dir / f"{case_id}.nii.gz"
        pos_path = image_dir / f"{case_id}_0002.nii.gz"
        neg_path = image_dir / f"{case_id}_0003.nii.gz"
        if not initial_path.is_file():
            raise FileNotFoundError(f"Missing initial mask channel for {case_id}: {initial_path}")
        if not gt_path.is_file():
            raise FileNotFoundError(f"Missing GT label for {case_id}: {gt_path}")
        if (pos_path.exists() or neg_path.exists()) and not overwrite:
            raise FileExistsError(f"Correction channels already exist for {case_id}; use --overwrite.")

        ct_img, ct = read_array(ct_path)
        _, initial = read_array(initial_path)
        _, gt = read_array(gt_path)
        pos_corr, neg_corr, selected, dice, gt_range = build_v3_1_correction_mask_channels(ct, initial, gt, top_k=top_k)

        write_like(ct_img, pos_corr, pos_path)
        write_like(ct_img, neg_corr, neg_path)

        fp_area = [int(neg_corr[z].sum()) for z in selected]
        fn_area = [int(pos_corr[z].sum()) for z in selected]
        selected_dice = [float(dice[z]) for z in selected]
        record = {
            "split": split,
            "case_id": case_id,
            "gt_range": [int(gt_range[0]), int(gt_range[1])],
            "selected_slices": selected,
            "selected_dice": selected_dice,
            "fp_area": fp_area,
            "fn_area": fn_area,
            "positive_correction_mask_sum": int(pos_corr.sum()),
            "negative_correction_mask_sum": int(neg_corr.sum()),
        }
        records.append(record)
        print(
            f"{split} {case_id}: GT range={record['gt_range']}, selected={selected}, "
            f"dice={[round(i, 4) for i in selected_dice]}, FP={fp_area}, FN={fn_area}, "
            f"pos_sum={record['positive_correction_mask_sum']}, neg_sum={record['negative_correction_mask_sum']}"
        )
    return records


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
    channel_names["1"] = "initial_mask"
    channel_names["2"] = "positive_correction_mask"
    channel_names["3"] = "negative_correction_mask"
    if "channel_names" in data:
        data["channel_names"] = channel_names
    else:
        data["modality"] = channel_names
    with dataset_json.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build v3-1 oracle simulated FP/FN correction mask channels. "
            "This is an oracle simulated correction prompt setting."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        default="/home/wusi/nnUNet_SAM2/nnUNetFrame/DATASET/nnUNet_raw/Dataset005_RectalAllPrompt",
        type=Path,
    )
    parser.add_argument("--top-k", default=3, type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["Tr", "Ts"],
        choices=["Tr", "Ts"],
        help="Splits to process. Default processes both training and testing with the same oracle prompt logic.",
    )
    parser.add_argument("--debug-jsonl", default=None, type=Path)
    parser.add_argument("--skip-dataset-json", action="store_true")
    args = parser.parse_args()

    # This is an oracle simulated correction prompt setting for both train and test:
    # GT is used to select the worst Dice top-k slices and to construct FP/FN correction masks.
    records = []
    for split in args.splits:
        records.extend(process_split(args.dataset_dir, split, top_k=args.top_k, overwrite=args.overwrite))

    if not args.skip_dataset_json:
        update_dataset_json(args.dataset_dir / "dataset.json")
        print("dataset.json: ensured 4 input channels: CT, initial_mask, positive_correction_mask, negative_correction_mask")

    debug_jsonl = args.debug_jsonl or (args.dataset_dir / "v3_1_fp_fn_correction_debug.jsonl")
    with debug_jsonl.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote debug records: {debug_jsonl}")


if __name__ == "__main__":
    main()
