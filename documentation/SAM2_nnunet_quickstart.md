# nnUNet_SAM2 Quickstart

This project adds a custom trainer `nnUNetTrainerSAM2` with a dual-encoder design:

- Main path: nnUNet native `ResidualEncoder + UNetDecoder`
- Auxiliary path: SAM2 Hiera encoder with trainable Adapters
- Fusion: deep feature fusion into nnUNet bottleneck skip (nnSAM-style deep fusion)

## Supported Configurations

- `2d`
- `3d_fullres` (SAM2 branch runs slice-wise in 2D and is reshaped/fused into 3D bottleneck features)

## Training Logic

Training schedule and optimization stay in nnUNet default logic (plans/adaptive config + default trainer behavior).

## Required Environment Variable

Set the SAM2 pretrained checkpoint path before training (must match cfg family, e.g. SAM2.1 cfg + SAM2.1 checkpoint):

```bash
export NNUNET_SAM2_CHECKPOINT=/absolute/path/to/sam2_hiera_large.pt
```

PowerShell:

```powershell
$env:NNUNET_SAM2_CHECKPOINT="D:\path\to\sam2_hiera_large.pt"
```

## Optional Environment Variables

```powershell
$env:NNUNET_SAM2_CFG="configs/sam2.1/sam2.1_hiera_l.yaml" # or sam2.1_hiera_l.yaml
$env:NNUNET_SAM2_REPO="D:\path\to\SAM2-UNet" # if SAM2-UNet is not under nnUNet_SAM2
$env:NNUNET_SAM2_INPUT_SIZE="1024"           # resize for SAM2 branch input
$env:NNUNET_SAM2_SLICE_BATCH="32"            # slice chunk size for 3D mode
```

## Train

```bash
nnUNetv2_train DATASET_ID_OR_NAME 2d 0 -tr nnUNetTrainerSAM2
nnUNetv2_train DATASET_ID_OR_NAME 3d_fullres 0 -tr nnUNetTrainerSAM2
```

## Notes

- SAM2 trunk parameters are frozen; Adapter parameters are trainable.
- For `3d_fullres`, if memory is tight, reduce `NNUNET_SAM2_INPUT_SIZE` and/or `NNUNET_SAM2_SLICE_BATCH`.

