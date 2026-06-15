import os
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nnunetv2.training.nnUNetTrainer.variants.network_architecture.sam2_nnunet_arch import (
    SAM2DualEncoderResidualUNet,
    get_sam2_cfg_from_env,
    get_sam2_checkpoint_from_env,
)


def main():
    os.environ.setdefault("NNUNET_SAM2_INPUT_SIZE", "64")
    os.environ.setdefault("NNUNET_SAM2_SLICE_BATCH", "2")
    os.environ.setdefault("NNUNET_SAM2_DEBUG_SHAPES", "1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SAM2DualEncoderResidualUNet(
        input_channels=2,
        n_stages=4,
        features_per_stage=[16, 32, 64, 128],
        conv_op=nn.Conv3d,
        kernel_sizes=[[3, 3, 3]] * 4,
        strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
        n_blocks_per_stage=[1, 1, 1, 1],
        num_classes=2,
        n_conv_per_stage_decoder=[1, 1, 1],
        conv_bias=True,
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"negative_slope": 1e-2, "inplace": True},
        deep_supervision=False,
        sam2_checkpoint_path=get_sam2_checkpoint_from_env(),
        sam2_model_cfg=get_sam2_cfg_from_env(),
    ).to(device)

    cases = {}

    x2 = torch.randn(1, 2, 16, 64, 64, device=device)
    x2[:, 1:2] = torch.sigmoid(x2[:, 1:2])
    cases["ct_initial_only"] = x2

    x4_empty = torch.zeros(1, 4, 16, 64, 64, device=device)
    x4_empty[:, 0:2] = x2
    cases["empty_correction_prompt"] = x4_empty

    x4_neg = torch.zeros(1, 4, 16, 64, 64, device=device)
    x4_neg[:, 0:2] = x2
    x4_neg[:, 3:4, 5:8, 18:26, 18:26] = 1.0
    cases["negative_correction_mask"] = x4_neg

    x4_both = torch.zeros(1, 4, 16, 64, 64, device=device)
    x4_both[:, 0:2] = x2
    x4_both[:, 2:3, 4:6, 34:42, 34:42] = 1.0
    x4_both[:, 3:4, 8:10, 18:26, 18:26] = 1.0
    cases["positive_negative_correction_masks"] = x4_both

    with torch.no_grad():
        for name, x in cases.items():
            y = model(x)
            out_shape = tuple(y.shape) if torch.is_tensor(y) else [tuple(i.shape) for i in y]
            print(name, "input:", tuple(x.shape), "output:", out_shape)


if __name__ == "__main__":
    main()
