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


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name, None)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "y", "on")


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


class CorrectionMaskEncoder(nn.Module):
    """
    Lightweight dense prompt encoder for positive/negative correction masks.
    Positive and negative correction masks carry add/delete semantics and are
    intentionally kept separate from SAM2's original mask prompt.
    """

    def __init__(self, prompt_embed_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(2, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, prompt_embed_dim, kernel_size=1),
        )

    def forward(self, corr_masks: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        if corr_masks.shape[2:] != target_hw:
            corr_masks = F.interpolate(corr_masks.float(), size=target_hw, mode="bilinear", align_corners=False)
        dense_corr = self.encoder(corr_masks.float())
        has_corr = (corr_masks.abs().flatten(1).sum(dim=1) > 0).to(dtype=dense_corr.dtype)
        return dense_corr * has_corr.view(-1, 1, 1, 1)


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
    v3-1 uses SAM2 as a prompt-conditioned refinement branch and injects the
    resulting SAM2 output into the later nnUNet skip features via gated residual
    addition.
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

        self.main_input_channels = max(2, int(input_channels))
        self.encoder = ResidualEncoder(
            self.main_input_channels,
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

        self.sam_input_adapter = nn.Conv2d(1, 3, kernel_size=1, bias=False)

        sam_model = build_sam2(
            sam2_model_cfg,
            ckpt_path=sam2_checkpoint_path,
            device="cpu",
            mode="eval",
        )
        for attr in ("memory_encoder", "memory_attention", "mask_downsample", "obj_ptr_tpos_proj", "obj_ptr_proj"):
            if hasattr(sam_model, attr):
                delattr(sam_model, attr)

        self.sam_image_encoder = sam_model.image_encoder
        self.sam_encoder = self.sam_image_encoder.trunk
        self.sam_prompt_encoder = sam_model.sam_prompt_encoder
        self.sam_mask_decoder = sam_model.sam_mask_decoder
        self.use_high_res_features_in_sam = bool(getattr(sam_model, "use_high_res_features_in_sam", False))

        for p in self.sam_image_encoder.parameters():
            p.requires_grad = False
        self.sam_encoder.blocks = nn.Sequential(*[Adapter(b) for b in self.sam_encoder.blocks])
        for p in self.sam_prompt_encoder.parameters():
            p.requires_grad = False
        for p in self.sam_mask_decoder.parameters():
            p.requires_grad = False

        self.sam_input_size = int(os.environ.get("NNUNET_SAM2_INPUT_SIZE", "1024"))
        if self.sam_input_size <= 0:
            raise ValueError(f"NNUNET_SAM2_INPUT_SIZE must be > 0, got {self.sam_input_size}")

        self.slice_batch = int(os.environ.get("NNUNET_SAM2_SLICE_BATCH", "4"))
        if self.slice_batch <= 0:
            raise ValueError(f"NNUNET_SAM2_SLICE_BATCH must be > 0, got {self.slice_batch}")

        self.prompt_threshold = float(os.environ.get("NNUNET_SAM2_PROMPT_THRESHOLD", "0.5"))
        self.debug_shapes = _env_flag("NNUNET_SAM2_DEBUG_SHAPES", False)
        self.use_initial_mask_as_sam_prompt = _env_flag("NNUNET_SAM2_USE_INITIAL_MASK_AS_SAM_PROMPT", False)

        self.fuse_mode = os.environ.get("NNUNET_SAM2_FUSE_MODE", "gated_add").lower()
        if self.fuse_mode not in ("gated_add", "add"):
            raise ValueError(
                f"NNUNET_SAM2_FUSE_MODE must be one of ['gated_add', 'add'], got {self.fuse_mode}"
            )

        conv_cls = nn.Conv2d if self.net_dim == 2 else nn.Conv3d
        self.prompt_refine_levels = int(os.environ.get("NNUNET_SAM2_PROMPT_REFINE_LEVELS", "2"))
        if self.prompt_refine_levels < 0:
            raise ValueError("NNUNET_SAM2_PROMPT_REFINE_LEVELS must be >= 0.")
        self.prompt_refine_levels = min(self.prompt_refine_levels, len(self.encoder.output_channels))
        self.prompt_proj = nn.ModuleList(
            [conv_cls(1, x_ch, kernel_size=1, bias=False) for x_ch in self.encoder.output_channels]
        )
        self.prompt_gate = nn.ModuleList(
            [conv_cls(2 * x_ch, x_ch, kernel_size=1, bias=True) for x_ch in self.encoder.output_channels]
        )
        self.prompt_scale = nn.ParameterList(
            [nn.Parameter(torch.tensor(0.0)) for _ in self.encoder.output_channels]
        )
        corr_hidden_dim = int(os.environ.get("NNUNET_SAM2_CORR_MASK_HIDDEN_DIM", "64"))
        if corr_hidden_dim <= 0:
            raise ValueError("NNUNET_SAM2_CORR_MASK_HIDDEN_DIM must be > 0.")
        self.correction_mask_encoder = CorrectionMaskEncoder(
            prompt_embed_dim=int(self.sam_prompt_encoder.embed_dim),
            hidden_dim=corr_hidden_dim,
        )
        self._print_trainable_summary()

    @staticmethod
    def _count_parameters(module: nn.Module) -> Tuple[int, int]:
        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        return trainable, total

    def _print_trainable_summary(self) -> None:
        groups = {
            "nnUNet encoder": self.encoder,
            "nnUNet decoder": self.decoder,
            "SAM2 image encoder incl Adapter": self.sam_image_encoder,
            "SAM2 prompt encoder": self.sam_prompt_encoder,
            "SAM2 mask decoder": self.sam_mask_decoder,
            "CorrectionMaskEncoder": self.correction_mask_encoder,
            "prompt-conditioned skip fusion": nn.ModuleList([self.prompt_proj, self.prompt_gate]),
        }
        print("v3-1 SAM2 refinement requires_grad summary:")
        for name, module in groups.items():
            trainable, total = self._count_parameters(module)
            print(f"  {name}: trainable={trainable} / total={total}")
        adapter_trainable = sum(
            p.numel()
            for name, p in self.sam_image_encoder.named_parameters()
            if "prompt_learn" in name and p.requires_grad
        )
        trunk_trainable = sum(
            p.numel()
            for name, p in self.sam_image_encoder.named_parameters()
            if "prompt_learn" not in name and p.requires_grad
        )
        print(f"  Adapter trainable params: {adapter_trainable}")
        print(f"  SAM2 original image encoder trainable params: {trunk_trainable}")

    @staticmethod
    def _normalize_sam_outputs(sam_outputs) -> List[torch.Tensor]:
        if isinstance(sam_outputs, (list, tuple)):
            feats = [i for i in sam_outputs if torch.is_tensor(i)]
            if len(feats) == 0:
                raise RuntimeError("SAM2 trunk returned no tensor features.")
            return feats
        raise RuntimeError(f"Unsupported SAM2 trunk output type: {type(sam_outputs)}")

    @staticmethod
    def _slice_volume(x: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.shape
        return x.permute(0, 2, 1, 3, 4).reshape(b * d, c, h, w)

    def _build_mask_prompt(self, init_mask_2d: torch.Tensor) -> torch.Tensor:
        mask_prompt = F.interpolate(
            init_mask_2d.float(),
            size=self.sam_prompt_encoder.mask_input_size,
            mode="bilinear",
            align_corners=False,
        )
        return mask_prompt

    def _decode_prompt_logits(
        self,
        trunk_feats: List[torch.Tensor],
        initial_mask_prompt: Union[torch.Tensor, None],
        corr_masks: torch.Tensor,
    ) -> torch.Tensor:
        image_features, _ = self.sam_image_encoder.neck(trunk_feats)
        image_features_for_sam = image_features[-3:] if len(image_features) >= 3 else image_features
        image_embedding = image_features_for_sam[-1]
        high_res_features = None
        if self.use_high_res_features_in_sam and len(image_features_for_sam) >= 3:
            high_res_features = [
                self.sam_mask_decoder.conv_s0(image_features_for_sam[0]),
                self.sam_mask_decoder.conv_s1(image_features_for_sam[1]),
            ]
        empty_coords = torch.zeros((image_embedding.shape[0], 1, 2), device=image_embedding.device)
        empty_labels = -torch.ones((image_embedding.shape[0], 1), device=image_embedding.device, dtype=torch.int64)
        sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(
            points=(empty_coords, empty_labels),
            boxes=None,
            masks=initial_mask_prompt,
        )
        if dense_embeddings.shape[2:] != image_embedding.shape[2:]:
            dense_embeddings = F.interpolate(
                dense_embeddings,
                size=image_embedding.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
        dense_init_embeddings = dense_embeddings
        dense_corr_embeddings = self.correction_mask_encoder(corr_masks, target_hw=dense_init_embeddings.shape[2:])
        assert dense_corr_embeddings.shape[0] == image_embedding.shape[0], (
            f"dense correction batch {dense_corr_embeddings.shape[0]} != image batch {image_embedding.shape[0]}"
        )
        assert dense_corr_embeddings.shape == dense_init_embeddings.shape, (
            f"dense correction shape {dense_corr_embeddings.shape} != dense init shape {dense_init_embeddings.shape}"
        )
        dense_embeddings = dense_init_embeddings + dense_corr_embeddings
        image_pe = self.sam_prompt_encoder.get_dense_pe().to(image_embedding.device, dtype=image_embedding.dtype)
        if image_pe.shape[2:] != image_embedding.shape[2:]:
            image_pe = F.interpolate(image_pe, size=image_embedding.shape[2:], mode="bilinear", align_corners=False)
        assert sparse_embeddings.shape[0] == image_embedding.shape[0], (
            f"sparse prompt batch {sparse_embeddings.shape[0]} != image batch {image_embedding.shape[0]}"
        )
        assert dense_embeddings.shape[0] == image_embedding.shape[0], (
            f"dense prompt batch {dense_embeddings.shape[0]} != image batch {image_embedding.shape[0]}"
        )
        low_res_masks, _, _, _ = self.sam_mask_decoder(
            image_embeddings=image_embedding,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res_features,
        )
        assert low_res_masks.shape[0] == image_embedding.shape[0], "SAM prompt logits batch mismatch."
        return low_res_masks[:, 0:1]

    def _sam_prompt_conditioned_in_chunks(
        self,
        x_2d: torch.Tensor,
        init_mask_2d: torch.Tensor,
        pos_corr_2d: Union[torch.Tensor, None] = None,
        neg_corr_2d: Union[torch.Tensor, None] = None,
        source_hw: Tuple[int, int] = None,
    ) -> torch.Tensor:
        prompt_logits = []
        bs = x_2d.shape[0]
        assert init_mask_2d.shape[0] == bs, "Initial mask prompt batch must match SAM image batch."
        mask_prompt_all = self._build_mask_prompt(init_mask_2d) if self.use_initial_mask_as_sam_prompt else None
        if mask_prompt_all is not None:
            assert mask_prompt_all.shape[0] == bs, "Initial mask prompt batch must match SAM image batch."
        if pos_corr_2d is None:
            pos_corr_2d = init_mask_2d.new_zeros(init_mask_2d.shape)
        if neg_corr_2d is None:
            neg_corr_2d = init_mask_2d.new_zeros(init_mask_2d.shape)
        corr_masks_all = torch.cat((pos_corr_2d.float(), neg_corr_2d.float()), dim=1)
        assert corr_masks_all.shape[0] == bs and corr_masks_all.shape[1] == 2, (
            f"Correction masks must be [B*D,2,H,W], got {tuple(corr_masks_all.shape)}."
        )
        for i in range(0, bs, self.slice_batch):
            part = x_2d[i: i + self.slice_batch]
            part_feats = self._normalize_sam_outputs(self.sam_encoder(part))
            mask_prompt = None if mask_prompt_all is None else mask_prompt_all[i: i + self.slice_batch]
            corr_masks = corr_masks_all[i: i + self.slice_batch]
            assert part.shape[0] == corr_masks.shape[0], "SAM image and correction mask chunk batch mismatch."
            if mask_prompt is not None:
                assert part.shape[0] == mask_prompt.shape[0], "SAM image and initial mask prompt chunk batch mismatch."
            prompt_logits.append(self._decode_prompt_logits(part_feats, mask_prompt, corr_masks))
        return torch.cat(prompt_logits, dim=0)

    def _build_prompt_conditioned_output_2d(
        self,
        ct: torch.Tensor,
        init_mask: torch.Tensor,
        pos_corr: Union[torch.Tensor, None] = None,
        neg_corr: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:
        source_hw = (int(ct.shape[-2]), int(ct.shape[-1]))
        sam_x = self.sam_input_adapter(ct)
        sam_x = F.interpolate(
            sam_x,
            size=(self.sam_input_size, self.sam_input_size),
            mode="bilinear",
            align_corners=True,
        )
        assert sam_x.shape[0] == init_mask.shape[0], "SAM image batch must match initial mask batch."
        assert sam_x.shape[1] == 3 and sam_x.shape[2] == self.sam_input_size and sam_x.shape[3] == self.sam_input_size
        return self._sam_prompt_conditioned_in_chunks(
            sam_x,
            init_mask,
            pos_corr_2d=pos_corr,
            neg_corr_2d=neg_corr,
            source_hw=source_hw,
        )

    def _build_prompt_conditioned_output_3d(
        self,
        ct: torch.Tensor,
        init_mask: torch.Tensor,
        pos_corr: Union[torch.Tensor, None] = None,
        neg_corr: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:
        b, _, d, h, w = ct.shape
        slices = self._slice_volume(ct)
        init_slices = self._slice_volume(init_mask)
        pos_corr_slices = None if pos_corr is None else self._slice_volume(pos_corr)
        neg_corr_slices = None if neg_corr is None else self._slice_volume(neg_corr)
        prompt_logits_2d = self._build_prompt_conditioned_output_2d(
            slices,
            init_slices,
            pos_corr_slices,
            neg_corr_slices,
        )
        prompt_logits_2d = F.interpolate(prompt_logits_2d, size=(h, w), mode="bilinear", align_corners=False)
        prompt_logits_3d = (
            prompt_logits_2d.reshape(b, d, 1, h, w)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
        )
        assert prompt_logits_3d.shape == init_mask.shape, (
            f"SAM prompt logits shape {prompt_logits_3d.shape} != initial mask shape {init_mask.shape}"
        )
        return prompt_logits_3d

    def _fuse_prompt_conditioned_output(self, skips: List[torch.Tensor], prompt_output: torch.Tensor) -> List[torch.Tensor]:
        if self.prompt_refine_levels == 0:
            return skips
        fused = list(skips)
        start_idx = max(0, len(skips) - self.prompt_refine_levels)
        for i in range(start_idx, len(skips)):
            prompt_feat = prompt_output
            if prompt_feat.shape[2:] != fused[i].shape[2:]:
                mode = "bilinear" if self.net_dim == 2 else "trilinear"
                prompt_feat = F.interpolate(prompt_feat, size=fused[i].shape[2:], mode=mode, align_corners=False)
            prompt_proj = self.prompt_proj[i](prompt_feat)
            assert prompt_proj.shape == fused[i].shape, "Prompt fusion projection shape mismatch."
            gate = torch.sigmoid(self.prompt_gate[i](torch.cat((fused[i], prompt_proj), dim=1)))
            fused[i] = fused[i] + self.prompt_scale[i] * gate * prompt_proj
        return fused

    def forward(self, x: torch.Tensor):
        assert x.ndim in (4, 5), f"Expected 2D/3D input, got shape {tuple(x.shape)}."
        assert x.shape[1] >= 2, f"v3-1 expects at least 2 channels: CT + initial mask, got {x.shape[1]}."
        ct = x[:, 0:1]
        init_mask = x[:, 1:2]
        # Channel convention:
        # 0 CT, 1 initial CTV mask, 2 positive correction mask (FN),
        # 3 negative correction mask (FP). Point/box prompts are not used in this version.
        pos_corr = x[:, 2:3] if x.shape[1] > 2 else torch.zeros_like(init_mask)
        neg_corr = x[:, 3:4] if x.shape[1] > 3 else torch.zeros_like(init_mask)
        assert pos_corr.shape == init_mask.shape, "Positive correction mask shape must match initial mask."
        assert neg_corr.shape == init_mask.shape, "Negative correction mask shape must match initial mask."
        if self.net_dim == 3:
            assert ct.ndim == 5 and init_mask.ndim == 5, "3D mode expects [B,C,D,H,W]."
        else:
            assert ct.ndim == 4 and init_mask.ndim == 4, "2D mode expects [B,C,H,W]."

        main_x = torch.cat((ct, init_mask), dim=1)
        if self.main_input_channels > 2:
            pad_channels = self.main_input_channels - 2
            main_x = torch.cat((main_x, main_x.new_zeros((main_x.shape[0], pad_channels, *main_x.shape[2:]))), dim=1)
        skips = list(self.encoder(main_x))
        if self.net_dim == 2:
            prompt_conditioned_output = self._build_prompt_conditioned_output_2d(ct, init_mask, pos_corr, neg_corr)
        else:
            prompt_conditioned_output = self._build_prompt_conditioned_output_3d(ct, init_mask, pos_corr, neg_corr)

        enhanced_skips = self._fuse_prompt_conditioned_output(skips, prompt_conditioned_output)
        if self.debug_shapes:
            print(
                "v3-1 shapes:",
                f"ct={tuple(ct.shape)}",
                f"init_mask={tuple(init_mask.shape)}",
                f"prompt_conditioned_output={tuple(prompt_conditioned_output.shape)}",
                f"skips={[tuple(i.shape) for i in enhanced_skips]}",
            )
        return self.decoder(enhanced_skips)

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
