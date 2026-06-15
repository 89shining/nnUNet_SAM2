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


def count(module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return trainable, total


def main():
    os.environ.setdefault("NNUNET_SAM2_INPUT_SIZE", "64")
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
    )

    groups = {
        "nnUNet encoder": model.encoder,
        "nnUNet decoder": model.decoder,
        "SAM2 image encoder incl Adapter": model.sam_image_encoder,
        "SAM2 prompt encoder": model.sam_prompt_encoder,
        "SAM2 mask decoder": model.sam_mask_decoder,
        "CorrectionMaskEncoder": model.correction_mask_encoder,
        "SAM image fusion": nn.ModuleList([model.sam_unify, model.fuse_proj, model.fuse_gate or nn.ModuleList()]),
        "prompt logit fusion": nn.ModuleList([model.prompt_proj, model.prompt_gate]),
    }
    for name, module in groups.items():
        trainable, total = count(module)
        print(f"{name}: trainable={trainable} / total={total}")

    adapter_trainable = sum(
        p.numel()
        for name, p in model.sam_image_encoder.named_parameters()
        if "prompt_learn" in name and p.requires_grad
    )
    sam_original_trainable = sum(
        p.numel()
        for name, p in model.sam_image_encoder.named_parameters()
        if "prompt_learn" not in name and p.requires_grad
    )
    print(f"Adapter trainable params: {adapter_trainable}")
    print(f"SAM2 original image encoder trainable params: {sam_original_trainable}")


if __name__ == "__main__":
    main()
