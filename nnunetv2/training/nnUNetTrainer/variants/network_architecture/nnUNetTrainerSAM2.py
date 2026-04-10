import pydoc
from typing import List, Tuple, Union

from torch import nn

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.sam2_nnunet_arch import (
    SAM2DualEncoderResidualUNet,
    get_sam2_cfg_from_env,
    get_sam2_checkpoint_from_env,
)


class nnUNetTrainerSAM2(nnUNetTrainer):
    """
    nnUNet trainer using native nnUNet encoder/decoder plus SAM2 auxiliary encoder branch.
    """

    @staticmethod
    def build_network_architecture(
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        _ = architecture_class_name
        architecture_kwargs = dict(**arch_init_kwargs)
        for key in arch_init_kwargs_req_import:
            if architecture_kwargs.get(key, None) is not None:
                architecture_kwargs[key] = pydoc.locate(architecture_kwargs[key])

        # Be compatible with plans that provide plain UNet kwargs.
        if "n_blocks_per_stage" not in architecture_kwargs and "n_conv_per_stage" in architecture_kwargs:
            architecture_kwargs["n_blocks_per_stage"] = architecture_kwargs["n_conv_per_stage"]
        if "n_conv_per_stage" in architecture_kwargs:
            del architecture_kwargs["n_conv_per_stage"]

        architecture_kwargs["deep_supervision"] = enable_deep_supervision
        architecture_kwargs["sam2_checkpoint_path"] = get_sam2_checkpoint_from_env()
        architecture_kwargs["sam2_model_cfg"] = get_sam2_cfg_from_env()

        return SAM2DualEncoderResidualUNet(
            input_channels=num_input_channels,
            num_classes=num_output_channels,
            **architecture_kwargs,
        )
