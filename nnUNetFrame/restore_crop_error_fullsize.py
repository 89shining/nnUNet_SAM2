import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def parse_group_name(name: str) -> Tuple[int, str]:
    # Expected: K1_inward, K2_outward, ...
    if "_" not in name or not name.startswith("K"):
        raise ValueError(f"Invalid group folder name: {name}")
    k_str, mode = name.split("_", 1)
    k = int(k_str[1:])
    if k not in (1, 2, 3):
        raise ValueError(f"K must be 1/2/3, got {k} in {name}")
    if mode not in {"inward", "outward", "upshift", "downshift"}:
        raise ValueError(f"Invalid mode {mode} in {name}")
    return k, mode


def get_io_backend():
    try:
        import SimpleITK as sitk  # type: ignore

        return "sitk", sitk
    except Exception:
        pass
    try:
        import nibabel as nib  # type: ignore

        return "nib", nib
    except Exception:
        pass
    raise ImportError("Need SimpleITK or nibabel.")


def read_nii_zyx(path: str, backend_name: str, backend_mod):
    if backend_name == "sitk":
        img = backend_mod.ReadImage(path)
        arr = backend_mod.GetArrayFromImage(img)
        return arr, {"img": img}

    img = backend_mod.load(path)
    arr_xyz = np.asarray(img.dataobj)
    if arr_xyz.ndim != 3:
        raise ValueError(f"Expected 3D image: {path}, shape={arr_xyz.shape}")
    arr_zyx = np.transpose(arr_xyz, (2, 1, 0))
    return arr_zyx, {"affine": img.affine, "header": img.header.copy()}


def write_nii_zyx(arr_zyx: np.ndarray, out_path: str, meta: Dict, backend_name: str, backend_mod):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if backend_name == "sitk":
        out_img = backend_mod.GetImageFromArray(arr_zyx)
        out_img.CopyInformation(meta["img"])
        backend_mod.WriteImage(out_img, out_path)
        return

    arr_xyz = np.transpose(arr_zyx, (2, 1, 0))
    out_img = backend_mod.Nifti1Image(arr_xyz, meta["affine"], header=meta["header"])
    backend_mod.save(out_img, out_path)


def strip_nii_ext(filename: str) -> str:
    if filename.endswith(".nii.gz"):
        return filename[:-7]
    if filename.endswith(".nii"):
        return filename[:-4]
    return filename


def find_nii(folder: str, stem: str) -> str:
    cands = [os.path.join(folder, stem + ".nii.gz"), os.path.join(folder, stem + ".nii")]
    for c in cands:
        if os.path.exists(c):
            return c
    return ""


def find_label_path(gt_dir: str, case_id: str) -> str:
    return find_nii(gt_dir, case_id)


def find_image_path(image_dir: str, case_id: str, image_suffix: str) -> str:
    p = find_nii(image_dir, case_id + image_suffix)
    if p:
        return p
    return find_nii(image_dir, case_id)


def compute_bounds_from_gt(gt_zyx: np.ndarray, k: int, mode: str, gt_threshold: float) -> Tuple[int, int]:
    z_indices = np.where(np.any(gt_zyx > gt_threshold, axis=(1, 2)))[0]
    if len(z_indices) == 0:
        raise ValueError("GT has no foreground.")

    low = int(z_indices[0])   # 下界序号小
    high = int(z_indices[-1]) # 上界序号大

    if mode == "inward":
        low_new, high_new = low + k, high - k
    elif mode == "outward":
        low_new, high_new = low - k, high + k
    elif mode == "upshift":
        low_new, high_new = low + k, high + k
    elif mode == "downshift":
        low_new, high_new = low - k, high - k
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return low_new, high_new


def clip_bounds(low: int, high: int, z_size: int) -> Tuple[int, int]:
    low = max(0, low)
    high = min(z_size - 1, high)
    return low, high


def list_pred_files(folder: str) -> List[str]:
    p = Path(folder)
    nii_gz = sorted(str(x) for x in p.glob("*.nii.gz"))
    nii = sorted(str(x) for x in p.glob("*.nii"))
    # Remove duplicates if both globs overlap (they shouldn't, but keep safe)
    return sorted(set(nii_gz + nii))


def restore_one(
    pred_path: str,
    gt_path: str,
    image_path: str,
    out_path: str,
    k: int,
    mode: str,
    gt_threshold: float,
    backend_name: str,
    backend_mod,
):
    pred, _ = read_nii_zyx(pred_path, backend_name, backend_mod)
    gt, _ = read_nii_zyx(gt_path, backend_name, backend_mod)
    image, image_meta = read_nii_zyx(image_path, backend_name, backend_mod)

    if gt.shape != image.shape:
        raise ValueError(f"GT/Image shape mismatch: gt={gt.shape}, image={image.shape}")
    if pred.ndim != 3:
        raise ValueError(f"Prediction is not 3D: {pred.shape}")
    if pred.shape[1:] != image.shape[1:]:
        raise ValueError(f"Only Z-crop restore is supported: pred={pred.shape}, image={image.shape}")

    low, high = compute_bounds_from_gt(gt, k, mode, gt_threshold)
    low, high = clip_bounds(low, high, image.shape[0])
    if low > high:
        raise ValueError(f"Invalid cropped z-range after clipping: low={low}, high={high}")

    expected_depth = high - low + 1
    if pred.shape[0] != expected_depth:
        raise ValueError(
            f"Depth mismatch, pred_z={pred.shape[0]}, expected={expected_depth}, "
            f"case={os.path.basename(pred_path)}, mode={mode}, k={k}, range=[{low},{high}]"
        )

    restored = np.zeros_like(image, dtype=pred.dtype)
    restored[low : high + 1, :, :] = pred
    write_nii_zyx(restored, out_path, image_meta, backend_name, backend_mod)


def main():
    parser = argparse.ArgumentParser(
        description="Restore Crop_error predictions to original full size by GT-derived z bounds."
    )
    parser.add_argument(
        "--pred_root",
        default="/home/wusi/nnUNet/nnUNetFrame/DATASET/nnUNet_results/"
        "Dataset008_EsoCTV73p/nnUNetTrainer__nnUNetPlans__3d_fullres/TestResults_28p_fold1/Crop_error",
        help="Input root with subfolders like K1_inward, K1_outward, ...",
    )
    parser.add_argument(
        "--out_root",
        default="/home/wusi/nnUNet/nnUNetFrame/DATASET/nnUNet_results/"
        "Dataset008_EsoCTV73p/nnUNetTrainer__nnUNetPlans__3d_fullres/TestResults_28p_fold1/Crop_error_fullsize",
        help="Output root for restored full-size predictions.",
    )
    parser.add_argument(
        "--gt_dir",
        default="/home/wusi/nnUNet/nnUNetFrame/DATASET/nnUNet_raw/Dataset009_EsoCTV73pAll/labelsTs",
        help="Full-size GT directory.",
    )
    parser.add_argument(
        "--image_dir",
        default="/home/wusi/nnUNet/nnUNetFrame/DATASET/nnUNet_raw/Dataset009_EsoCTV73pAll/imagesTs",
        help="Full-size original image directory.",
    )
    parser.add_argument(
        "--image_suffix",
        default="_0000",
        help="Suffix used in imagesTs filename. default: _0000",
    )
    parser.add_argument(
        "--gt_threshold",
        type=float,
        default=0.0,
        help="GT foreground threshold, default > 0.",
    )
    args = parser.parse_args()

    backend_name, backend_mod = get_io_backend()
    print(f"IO backend: {backend_name}")

    os.makedirs(args.out_root, exist_ok=True)
    group_dirs = sorted([d for d in os.listdir(args.pred_root) if os.path.isdir(os.path.join(args.pred_root, d))])
    if not group_dirs:
        raise FileNotFoundError(f"No group folders found under: {args.pred_root}")

    total_ok = 0
    total_fail = 0
    for group in group_dirs:
        try:
            k, mode = parse_group_name(group)
        except Exception as e:
            print(f"[Skip Group] {group}: {e}")
            continue

        pred_dir = os.path.join(args.pred_root, group)
        out_dir = os.path.join(args.out_root, group)
        os.makedirs(out_dir, exist_ok=True)

        pred_files = list_pred_files(pred_dir)
        if not pred_files:
            print(f"[Skip Group] {group}: no prediction files.")
            continue

        print(f"\n[Group] {group} | files={len(pred_files)}")
        ok = 0
        fail = 0
        for pred_path in pred_files:
            pred_name = os.path.basename(pred_path)
            case_id = strip_nii_ext(pred_name)
            gt_path = find_label_path(args.gt_dir, case_id)
            image_path = find_image_path(args.image_dir, case_id, args.image_suffix)
            if not gt_path:
                print(f"[Skip] GT not found: {case_id}")
                fail += 1
                continue
            if not image_path:
                print(f"[Skip] Image not found: {case_id}")
                fail += 1
                continue

            out_path = os.path.join(out_dir, pred_name)
            try:
                restore_one(
                    pred_path=pred_path,
                    gt_path=gt_path,
                    image_path=image_path,
                    out_path=out_path,
                    k=k,
                    mode=mode,
                    gt_threshold=args.gt_threshold,
                    backend_name=backend_name,
                    backend_mod=backend_mod,
                )
                ok += 1
                print(f"[OK] {group}/{pred_name}")
            except Exception as e:
                fail += 1
                print(f"[Fail] {group}/{pred_name}: {e}")

        print(f"[Group Done] {group}: OK={ok}, Fail={fail}")
        total_ok += ok
        total_fail += fail

    print(f"\nAll done. Total OK={total_ok}, Total Fail={total_fail}")


if __name__ == "__main__":
    main()
