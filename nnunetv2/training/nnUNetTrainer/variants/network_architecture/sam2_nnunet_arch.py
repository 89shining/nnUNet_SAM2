import math
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
    Default nnUNet-style decoder that expects unchanged skip channel sizes.
    """

    def __init__(
        self,
        encoder: ResidualEncoder,
        num_classes: int,
        n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
        deep_supervision: bool,
    ):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes

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

            stages.append(
                StackedConvBlocks(
                    n_conv_per_stage[s - 1],
                    encoder.conv_op,
                    2 * input_features_skip,
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
    Uses per-skip nearest-scale SAM2 fusion with gated residual addition.
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

        self.decoder = SAM2FusionDecoder(
            self.encoder,
            num_classes,
            n_conv_per_stage_decoder,
            deep_supervision,
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

        self.sam_input_size = int(os.environ.get("NNUNET_SAM2_INPUT_SIZE", "1024"))
        if self.sam_input_size <= 0:
            raise ValueError(f"NNUNET_SAM2_INPUT_SIZE must be > 0, got {self.sam_input_size}")

        self.slice_batch = int(os.environ.get("NNUNET_SAM2_SLICE_BATCH", "4"))
        if self.slice_batch <= 0:
            raise ValueError(f"NNUNET_SAM2_SLICE_BATCH must be > 0, got {self.slice_batch}")

        self.fuse_mode = os.environ.get("NNUNET_SAM2_FUSE_MODE", "gated_add").lower()
        if self.fuse_mode not in ("gated_add", "add"):
            raise ValueError(
                f"NNUNET_SAM2_FUSE_MODE must be one of ['gated_add', 'add'], got {self.fuse_mode}"
            )

        self.sam_out_channels = self._infer_sam_out_channels()
        conv_cls = nn.Conv2d if self.net_dim == 2 else nn.Conv3d
        self.sam_common_channels = int(
            os.environ.get("NNUNET_SAM2_COMMON_CHANNELS", os.environ.get("NNUNET_SAM2_FUSION_CHANNELS", "256"))
        )
        if self.sam_common_channels <= 0:
            raise ValueError(f"NNUNET_SAM2_COMMON_CHANNELS must be > 0, got {self.sam_common_channels}")

        # O(M): one unify layer per SAM feature level.
        self.sam_unify = nn.ModuleList(
            [conv_cls(s_ch, self.sam_common_channels, kernel_size=1, bias=False) for s_ch in self.sam_out_channels]
        )
        # O(N): one projection/gate/scale per nnUNet skip level.
        self.fuse_proj = nn.ModuleList(
            [
                conv_cls(self.sam_common_channels, x_ch, kernel_size=1, bias=False)
                for x_ch in self.encoder.output_channels
            ]
        )
        self.fuse_gate = (
            nn.ModuleList(
                [conv_cls(2 * x_ch, x_ch, kernel_size=1, bias=True) for x_ch in self.encoder.output_channels]
            )
            if self.fuse_mode == "gated_add"
            else None
        )
        self.fuse_scale = nn.ParameterList(
            [nn.Parameter(torch.tensor(0.0)) for _ in self.encoder.output_channels]
        )

    @staticmethod
    def _normalize_sam_outputs(sam_outputs) -> List[torch.Tensor]:
        if isinstance(sam_outputs, (list, tuple)):
            feats = [i for i in sam_outputs if torch.is_tensor(i)]
            if len(feats) == 0:
                raise RuntimeError("SAM2 trunk returned no tensor features.")
            return feats
        raise RuntimeError(f"Unsupported SAM2 trunk output type: {type(sam_outputs)}")

    def _infer_sam_out_channels(self) -> List[int]:
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 64, 64, dtype=torch.float32)
            feats = self._normalize_sam_outputs(self.sam_encoder(dummy))
        return [int(i.shape[1]) for i in feats]

    def _sam_encode_in_chunks(self, x_2d: torch.Tensor) -> List[torch.Tensor]:
        feats = None
        bs = x_2d.shape[0]
        for i in range(0, bs, self.slice_batch):
            part = x_2d[i: i + self.slice_batch]
            part_feats = self._normalize_sam_outputs(self.sam_encoder(part))
            if feats is None:
                feats = [[] for _ in range(len(part_feats))]
            if len(part_feats) != len(feats):
                raise RuntimeError("Inconsistent number of SAM2 feature levels across chunks.")
            for j, f in enumerate(part_feats):
                feats[j].append(f)
        return [torch.cat(i, dim=0) for i in feats]

    def _build_sam_feats_2d(self, x: torch.Tensor) -> List[torch.Tensor]:
        sam_x = self.sam_input_adapter(x)
        sam_x = F.interpolate(
            sam_x,
            size=(self.sam_input_size, self.sam_input_size),
            mode="bilinear",
            align_corners=True,
        )
        return self._sam_encode_in_chunks(sam_x)

    def _build_sam_feats_3d(self, x: torch.Tensor) -> List[torch.Tensor]:
        b, c, d, h, w = x.shape
        slices = x.permute(0, 2, 1, 3, 4).reshape(b * d, c, h, w)
        sam_feats_2d = self._build_sam_feats_2d(slices)
        sam_feats_3d = []
        for feat_2d in sam_feats_2d:
            c_out = feat_2d.shape[1]
            sam_feats_3d.append(feat_2d.reshape(b, d, c_out, feat_2d.shape[-2], feat_2d.shape[-1]).permute(0, 2, 1, 3, 4))
        return sam_feats_3d

    @staticmethod
    def _nearest_scale_idx(skip_spatial, sam_spatials) -> int:
        def _norm_dims(shape):
            if len(shape) == 3:
                return shape[1], shape[2]
            return tuple(shape)

        skip_dims = _norm_dims(skip_spatial)
        scores = []
        for j, shape in enumerate(sam_spatials):
            sam_dims = _norm_dims(shape)
            score = 0.0
            for a, b in zip(skip_dims, sam_dims):
                score += abs(math.log2(max(a, 1)) - math.log2(max(b, 1)))
            scores.append((score, j))
        return min(scores)[1]

    def _resize_to_skip(self, feat: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if feat.shape[2:] == skip.shape[2:]:
            return feat
        mode = "bilinear" if self.net_dim == 2 else "trilinear"
        return F.interpolate(feat, size=skip.shape[2:], mode=mode, align_corners=False)

    def forward(self, x: torch.Tensor):
        skips = list(self.encoder(x))
        sam_feats = self._build_sam_feats_2d(x) if self.net_dim == 2 else self._build_sam_feats_3d(x)
        sam_spatials = [i.shape[2:] for i in sam_feats]

        fused_skips = []
        for i, skip in enumerate(skips):
            j = self._nearest_scale_idx(skip.shape[2:], sam_spatials)
            sam_feat = self.sam_unify[j](sam_feats[j])
            sam_feat = self._resize_to_skip(sam_feat, skip)
            sam_proj = self.fuse_proj[i](sam_feat)
            if self.fuse_mode == "gated_add":
                gate = torch.sigmoid(self.fuse_gate[i](torch.cat((skip, sam_proj), dim=1)))
                fused_skips.append(skip + self.fuse_scale[i] * gate * sam_proj)
            else:
                fused_skips.append(skip + self.fuse_scale[i] * sam_proj)

        return self.decoder(fused_skips)

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
