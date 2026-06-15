import argparse
from pathlib import Path

import numpy as np

from build_v3_1_fp_fn_correction_channels import build_v3_1_correction_mask_channels


def _load_case_data(case_file: Path):
    if case_file.suffix == ".npz":
        payload = np.load(case_file)
        data = payload["data"]
        seg = payload["seg"]
    elif case_file.suffix == ".npy":
        data = np.load(case_file, mmap_mode="r")
        seg = np.load(case_file.with_name(case_file.stem + "_seg.npy"), mmap_mode="r")
    else:
        raise ValueError(f"Unsupported case file: {case_file}")
    return data, seg


def build_sidecar_for_case(case_file: Path, overwrite: bool = False, top_k: int = 3) -> Path:
    case_id = case_file.stem
    sidecar_path = case_file.with_name(f"{case_id}_v3_prompt.npy")
    if sidecar_path.exists() and not overwrite:
        return sidecar_path

    data, seg = _load_case_data(case_file)
    if data.shape[0] < 2:
        raise RuntimeError(f"{case_id}: expected at least 2 preprocessed channels [CT, initial_mask], got {data.shape}")
    if seg.shape[0] < 1:
        raise RuntimeError(f"{case_id}: expected segmentation channels, got {seg.shape}")

    ct = np.asarray(data[0])
    initial = np.asarray(data[1])
    gt = np.asarray(seg[0])
    pos_corr, neg_corr, selected, dice, gt_range = build_v3_1_correction_mask_channels(ct, initial, gt, top_k=top_k)
    prompt = np.stack((pos_corr, neg_corr), axis=0).astype(np.uint8, copy=False)
    np.save(sidecar_path, prompt)
    print(
        f"{case_id}: wrote {sidecar_path.name}, gt_range={gt_range}, "
        f"selected={selected}, dice={[round(float(dice[z]), 4) for z in selected]}"
    )
    return sidecar_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build v3-1 case-level prompt sidecars in an nnUNet preprocessed folder. "
            "The main preprocessed data stays two-channel; prompts are saved as <case_id>_v3_prompt.npy."
        )
    )
    parser.add_argument("--preprocessed-folder", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    case_files = sorted(
        [
            p
            for p in args.preprocessed_folder.glob("*.npy")
            if not p.name.endswith("_seg.npy") and not p.name.endswith("_v3_prompt.npy")
        ]
    )
    if len(case_files) == 0:
        case_files = sorted(args.preprocessed_folder.glob("*.npz"))

    if len(case_files) == 0:
        raise FileNotFoundError(f"No preprocessed case files found in {args.preprocessed_folder}")

    for case_file in case_files:
        build_sidecar_for_case(case_file, overwrite=args.overwrite, top_k=args.top_k)


if __name__ == "__main__":
    main()
