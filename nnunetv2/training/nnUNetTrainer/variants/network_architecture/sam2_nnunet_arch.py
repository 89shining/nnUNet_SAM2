import os
import sys
import warnings
from pathlib import Path
from typing import List, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim, get_matching_convtransp
from dynamic_network_architectures.building_blocks.residual import BasicBlockD, BottleneckD
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
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


class SAM2FusionDecoder(nn.Module):
    """
    nnSAM-style decoder: one selected decoder stage receives additional SAM feature channels
    via concatenated skip features.
    """

    def __init__(
        self,
        encoder: ResidualEncoder,
        num_classes: int,
        n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
        deep_supervision: bool,
        fusion_stage_index: int,
        fusion_channels: int,
    ):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        self.fusion_stage_index = fusion_stage_index
        self.fusion_channels = fusion_channels

        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1, (
            "n_conv_per_stage must have one entry per decoder stage."
        )

        transpconv_op = get_matching_convtransp(conv_op=encoder.conv_op)

        stages = []
        transpconvs = []
        seg_layers = []
        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_transpconv = encoder.strides[-s]
            transpconvs.append(
                transpconv_op(
                    input_features_below,
                    input_features_skip,
                    stride_for_transpconv,
                    stride_for_transpconv,
                    bias=encoder.conv_bias,
                )
            )

            stage_idx = s - 1
            extra_channels = self.fusion_channels if stage_idx == self.fusion_stage_index else 0
            stages.append(
                StackedConvBlocks(
                    n_conv_per_stage[stage_idx],
                    encoder.conv_op,
                    (2 * input_features_skip) + extra_channels,
                    input_features_skip,
                    encoder.kernel_sizes[-(s + 1)],
                    1,
                    encoder.conv_bias,
                    encoder.norm_op,
                    encoder.norm_op_kwargs,
                    encoder.dropout_op,
                    encoder.dropout_op_kwargs,
                    encoder.nonlin,
                    encoder.nonlin_kwargs,
                )
            )

            seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))

        self.stages = nn.ModuleList(stages)
        self.transpconvs = nn.ModuleList(transpconvs)
        self.seg_layers = nn.ModuleList(seg_layers)

    def forward(self, skips):
        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.transpconvs[s](lres_input)
            x = torch.cat((x, skips[-(s + 2)]), 1)
            x = self.stages[s](x)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x

        seg_outputs = seg_outputs[::-1]
        return seg_outputs if self.deep_supervision else seg_outputs[0]

    def compute_conv_feature_map_size(self, input_size):
        skip_sizes = []
        for s in range(len(self.encoder.strides) - 1):
            skip_sizes.append([i // j for i, j in zip(input_size, self.encoder.strides[s])])
            input_size = skip_sizes[-1]

        output = np.int64(0)
        for s in range(len(self.stages)):
            output += self.stages[s].compute_conv_feature_map_size(skip_sizes[-(s + 1)])
            output += np.prod([self.encoder.output_channels[-(s + 2)], *skip_sizes[-(s + 1)]], dtype=np.int64)
            if self.deep_supervision or (s == (len(self.stages) - 1)):
                output += np.prod([self.num_classes, *skip_sizes[-(s + 1)]], dtype=np.int64)
        return output


class SAM2DualEncoderResidualUNet(nn.Module):
    """
    nnUNet native ResidualEncoder + decoder with an auxiliary SAM2 encoder branch.
    Uses nnSAM-style concatenation fusion at a selected deep skip stage.
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

        self.fusion_skip_index = int(os.environ.get("NNUNET_SAM2_FUSION_SKIP_INDEX", "3"))
        max_skip_index = len(self.encoder.output_channels) - 2
        if max_skip_index < 0:
            raise RuntimeError("Invalid encoder output stages for SAM2 fusion.")
        self.fusion_skip_index = max(0, min(self.fusion_skip_index, max_skip_index))

        self.sam_fusion_channels = int(os.environ.get("NNUNET_SAM2_FUSION_CHANNELS", "256"))
        if self.sam_fusion_channels <= 0:
            raise ValueError(f"NNUNET_SAM2_FUSION_CHANNELS must be > 0, got {self.sam_fusion_channels}")

        self.fusion_stage_index = (len(self.encoder.output_channels) - 2) - self.fusion_skip_index
        self.decoder = SAM2FusionDecoder(
            self.encoder,
            num_classes,
            n_conv_per_stage_decoder,
            deep_supervision,
            fusion_stage_index=self.fusion_stage_index,
            fusion_channels=self.sam_fusion_channels,
        )

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

        sam_out_channels = self._infer_sam_out_channels()
        self.sam_fuse_proj = nn.Conv2d(sam_out_channels, self.sam_fusion_channels, kernel_size=1, bias=False)
        self.fusion_scale = nn.Parameter(torch.tensor(1.0))

        self.sam_input_size = int(os.environ.get("NNUNET_SAM2_INPUT_SIZE", "1024"))
        if self.sam_input_size <= 0:
            raise ValueError(f"NNUNET_SAM2_INPUT_SIZE must be > 0, got {self.sam_input_size}")

        self.slice_batch = int(os.environ.get("NNUNET_SAM2_SLICE_BATCH", "4"))
        if self.slice_batch <= 0:
            raise ValueError(f"NNUNET_SAM2_SLICE_BATCH must be > 0, got {self.slice_batch}")

    def _infer_sam_out_channels(self) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 64, 64, dtype=torch.float32)
            _, _, _, f = self.sam_encoder(dummy)
        return int(f.shape[1])

    def _sam_encode_in_chunks(self, x_2d: torch.Tensor) -> torch.Tensor:
        feats = []
        bs = x_2d.shape[0]
        for i in range(0, bs, self.slice_batch):
            part = x_2d[i: i + self.slice_batch]
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
        target_spatial_shape = skips[self.fusion_skip_index].shape[2:]

        if self.net_dim == 2:
            sam_feat = self._build_sam_feat_2d(x, target_hw=target_spatial_shape)
            skips[self.fusion_skip_index] = torch.cat(
                (skips[self.fusion_skip_index], self.fusion_scale * sam_feat), dim=1
            )
        else:
            sam_feat = self._build_sam_feat_3d(x, target_dhw=target_spatial_shape)
            skips[self.fusion_skip_index] = torch.cat(
                (skips[self.fusion_skip_index], self.fusion_scale * sam_feat), dim=1
            )

        return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), (
            "Provide spatial size only, e.g. (x, y) or (z, y, x), without batch/channel dims."
        )
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(
            input_size
        )


def get_sam2_checkpoint_from_env() -> str:
    ckpt_path = os.environ.get("NNUNET_SAM2_CHECKPOINT", None)
    if ckpt_path is None:
        warnings.warn(
            "NNUNET_SAM2_CHECKPOINT is not set. SAM2 trunk may start without pretrained weights, "
            "which is usually not intended for training.",
            UserWarning,
        )
        return None

    if not Path(ckpt_path).is_file():
        raise FileNotFoundError(f"NNUNET_SAM2_CHECKPOINT points to a non-existing file: {ckpt_path}")

    return ckpt_path


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
