import os
import sys
from pathlib import Path
from typing import List, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim
from dynamic_network_architectures.building_blocks.residual import BasicBlockD, BottleneckD
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd


def _add_sam2_unet_repo_to_path() -> Path:
    override = os.environ.get("NNUNET_SAM2_REPO", None)
    if override is not None:
        sam2_unet_root = Path(override)
        if not sam2_unet_root.exists():
            raise FileNotFoundError(
                f"NNUNET_SAM2_REPO points to non-existing path: {sam2_unet_root}"
            )
    else:
        candidates = [
            Path(__file__).resolve().parents[5] / "SAM2-UNet",
            Path(__file__).resolve().parents[6] / "SAM2-UNet",
        ]
        sam2_unet_root = None
        for c in candidates:
            if c.exists():
                sam2_unet_root = c
                break
        if sam2_unet_root is None:
            raise FileNotFoundError(
                "SAM2-UNet directory was not found. Tried: " + ", ".join(str(i) for i in candidates)
            )

    sam2_unet_root_str = str(sam2_unet_root)
    if sam2_unet_root_str not in sys.path:
        sys.path.insert(0, sam2_unet_root_str)
    return sam2_unet_root


_add_sam2_unet_repo_to_path()
from sam2.build_sam import build_sam2  # noqa: E402


class Adapter(nn.Module):
    def __init__(self, blk: nn.Module) -> None:
        super().__init__()
        self.block = blk
        dim = blk.attn.qkv.in_features
        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, 32),
            nn.GELU(),
            nn.Linear(32, dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prompt = self.prompt_learn(x)
        return self.block(x + prompt)


class SAM2DualEncoderResidualUNet(nn.Module):
    """
    nnUNet native ResidualEncoder + UNetDecoder with an auxiliary SAM2 encoder branch.
    2D input: standard deep fusion.
    3D input: slice-wise SAM2 encoding then reshaped fusion.
    """

    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: Type[_ConvNd],
        kernel_sizes: Union[int, List[int], Tuple[int, ...]],
        strides: Union[int, List[int], Tuple[int, ...]],
        n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]] = None,
        n_conv_per_stage: Union[int, List[int], Tuple[int, ...]] = None,
        num_classes: int = 2,
        n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]] = 2,
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[torch.nn.Module]] = None,
        nonlin_kwargs: dict = None,
        deep_supervision: bool = True,
        block: Union[Type[BasicBlockD], Type[BottleneckD]] = BasicBlockD,
        bottleneck_channels: Union[int, List[int], Tuple[int, ...]] = None,
        stem_channels: int = None,
        sam2_checkpoint_path: str = None,
        sam2_model_cfg: str = "configs/sam2/sam2_hiera_l.yaml",
    ):
        super().__init__()

        if n_blocks_per_stage is None and n_conv_per_stage is not None:
            n_blocks_per_stage = n_conv_per_stage
        if n_blocks_per_stage is None:
            raise ValueError("n_blocks_per_stage is required (or provide n_conv_per_stage).")

        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)

        self.encoder = ResidualEncoder(
            input_channels,
            n_stages,
            features_per_stage,
            conv_op,
            kernel_sizes,
            strides,
            n_blocks_per_stage,
            conv_bias,
            norm_op,
            norm_op_kwargs,
            dropout_op,
            dropout_op_kwargs,
            nonlin,
            nonlin_kwargs,
            block,
            bottleneck_channels,
            return_skips=True,
            disable_default_stem=False,
            stem_channels=stem_channels,
        )
        self.decoder = UNetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision)

        self.net_dim = convert_conv_op_to_dim(conv_op)
        if self.net_dim not in (2, 3):
            raise RuntimeError(f"Unsupported conv dim {self.net_dim}. Only 2D/3D are supported.")

        self.sam_input_adapter = (
            nn.Identity() if input_channels == 3 else nn.Conv2d(input_channels, 3, kernel_size=1, bias=False)
        )

        sam_model = build_sam2(
            sam2_model_cfg,
            ckpt_path=sam2_checkpoint_path,
            device="cpu",
            mode="eval",
        )
        for attr in (
            "sam_mask_decoder",
            "sam_prompt_encoder",
            "memory_encoder",
            "memory_attention",
            "mask_downsample",
            "obj_ptr_tpos_proj",
            "obj_ptr_proj",
        ):
            if hasattr(sam_model, attr):
                delattr(sam_model, attr)
        if hasattr(sam_model, "image_encoder") and hasattr(sam_model.image_encoder, "neck"):
            del sam_model.image_encoder.neck

        self.sam_encoder = sam_model.image_encoder.trunk
        for p in self.sam_encoder.parameters():
            p.requires_grad = False
        self.sam_encoder.blocks = nn.Sequential(*[Adapter(b) for b in self.sam_encoder.blocks])

        deepest_channels = self.encoder.output_channels[-1]
        self.sam_fuse_proj = nn.Conv2d(1152, deepest_channels, kernel_size=1, bias=False)
        self.fusion_scale = nn.Parameter(torch.tensor(1.0))

        self.sam_input_size = int(os.environ.get("NNUNET_SAM2_INPUT_SIZE", "1024"))
        self.slice_batch = int(os.environ.get("NNUNET_SAM2_SLICE_BATCH", "32"))

    def _sam_encode_in_chunks(self, x_2d: torch.Tensor) -> torch.Tensor:
        feats = []
        bs = x_2d.shape[0]
        for i in range(0, bs, self.slice_batch):
            part = x_2d[i : i + self.slice_batch]
            _, _, _, f = self.sam_encoder(part)
            feats.append(f)
        return torch.cat(feats, dim=0)

    def _build_sam_feat_2d(self, x: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        sam_x = self.sam_input_adapter(x)
        sam_x = F.interpolate(
            sam_x,
            size=(self.sam_input_size, self.sam_input_size),
            mode="bilinear",
            align_corners=True,
        )
        sam_feat = self._sam_encode_in_chunks(sam_x)
        sam_feat = self.sam_fuse_proj(sam_feat)
        sam_feat = F.interpolate(sam_feat, size=target_hw, mode="bilinear", align_corners=False)
        return sam_feat

    def _build_sam_feat_3d(self, x: torch.Tensor, target_dhw: Tuple[int, int, int]) -> torch.Tensor:
        b, c, d, h, w = x.shape
        slices = x.permute(0, 2, 1, 3, 4).reshape(b * d, c, h, w)
        sam_feat_2d = self._build_sam_feat_2d(slices, target_hw=(target_dhw[1], target_dhw[2]))
        c_out = sam_feat_2d.shape[1]
        sam_feat_3d = sam_feat_2d.reshape(b, d, c_out, target_dhw[1], target_dhw[2]).permute(0, 2, 1, 3, 4)
        if sam_feat_3d.shape[2] != target_dhw[0]:
            sam_feat_3d = F.interpolate(sam_feat_3d, size=target_dhw, mode="trilinear", align_corners=False)
        return sam_feat_3d

    def forward(self, x: torch.Tensor):
        skips = list(self.encoder(x))

        if self.net_dim == 2:
            sam_feat = self._build_sam_feat_2d(x, target_hw=skips[-1].shape[2:])
            skips[-1] = skips[-1] + self.fusion_scale * sam_feat
        else:
            sam_feat = self._build_sam_feat_3d(x, target_dhw=skips[-1].shape[2:])
            skips[-1] = skips[-1] + self.fusion_scale * sam_feat

        return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), (
            "Provide spatial size only, e.g. (x, y) or (z, y, x), without batch/channel dims."
        )
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(
            input_size
        )


def get_sam2_checkpoint_from_env() -> str:
    return os.environ.get("NNUNET_SAM2_CHECKPOINT", None)


def _normalize_sam2_cfg_name(cfg_name: str) -> str:
    mapping = {
        "sam2_hiera_t.yaml": "configs/sam2/sam2_hiera_t.yaml",
        "sam2_hiera_s.yaml": "configs/sam2/sam2_hiera_s.yaml",
        "sam2_hiera_b+.yaml": "configs/sam2/sam2_hiera_b+.yaml",
        "sam2_hiera_l.yaml": "configs/sam2/sam2_hiera_l.yaml",
        "sam2.1_hiera_t.yaml": "configs/sam2.1/sam2.1_hiera_t.yaml",
        "sam2.1_hiera_s.yaml": "configs/sam2.1/sam2.1_hiera_s.yaml",
        "sam2.1_hiera_b+.yaml": "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "sam2.1_hiera_l.yaml": "configs/sam2.1/sam2.1_hiera_l.yaml",
    }
    return mapping.get(cfg_name, cfg_name)


def get_sam2_cfg_from_env() -> str:
    raw = os.environ.get("NNUNET_SAM2_CFG", "configs/sam2/sam2_hiera_l.yaml")
    return _normalize_sam2_cfg_name(raw)
