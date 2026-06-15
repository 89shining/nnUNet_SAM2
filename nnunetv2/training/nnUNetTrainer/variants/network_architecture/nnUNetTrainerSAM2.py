import pydoc
from typing import List, Tuple, Union

import torch
from torch import autocast, nn

from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.sam2_nnunet_arch import (
    SAM2DualEncoderResidualUNet,
    get_sam2_cfg_from_env,
    get_sam2_checkpoint_from_env,
)
from nnunetv2.utilities.helpers import dummy_context


class nnUNetTrainerSAM2(nnUNetTrainer):
    """
    nnUNet trainer using native nnUNet encoder/decoder plus SAM2 auxiliary encoder branch.

    v3-1 keeps the raw nnU-Net dataset input at two channels:
      data[:, 0:1] = CT
      data[:, 1:2] = v2-1 initial CTV mask

    During train/validation, this trainer builds oracle simulated FP/FN correction
    mask prompts from the current batch GT and temporarily feeds the network:
      [CT, initial_mask, FN_positive_correction, FP_negative_correction].
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
            input_channels=max(2, num_input_channels),
            num_classes=num_output_channels,
            **architecture_kwargs,
        )

    @staticmethod
    def _first_target(target):
        return target[0] if isinstance(target, list) else target

    @staticmethod
    def _as_binary_gt(gt: torch.Tensor) -> torch.Tensor:
        if gt.dtype == torch.bool:
            return gt.float()
        return (gt > 0).float()

    @staticmethod
    def _build_v3_1_correction_input(data: torch.Tensor, target) -> torch.Tensor:
        """
        Build oracle simulated correction prompts online from initial mask and GT.

        The selected slices are the worst 2D Dice top-3 within the GT SI range.
        Positive correction mask is FN; negative correction mask is FP.
        """
        if data.shape[1] < 2:
            raise RuntimeError(f"v3-1 expects raw data with CT + initial mask, got {tuple(data.shape)}")

        gt = nnUNetTrainerSAM2._first_target(target)
        if gt.shape[2:] != data.shape[2:]:
            raise RuntimeError(f"Target shape {tuple(gt.shape)} must match data spatial shape {tuple(data.shape)}")

        initial = data[:, 1:2] > 0.5
        gt_bin = nnUNetTrainerSAM2._as_binary_gt(gt[:, 0:1])
        pos_corr = torch.zeros_like(data[:, 1:2])
        neg_corr = torch.zeros_like(data[:, 1:2])

        if data.ndim == 5:
            for b in range(data.shape[0]):
                gt_area = gt_bin[b, 0].flatten(1).sum(dim=1)
                fg = torch.nonzero(gt_area > 0, as_tuple=False).flatten()
                if fg.numel() == 0:
                    continue
                z_min = int(fg[0].item())
                z_max = int(fg[-1].item())
                candidates = torch.arange(z_min, z_max + 1, device=data.device)
                dice_values = []
                for z in candidates:
                    pred_z = initial[b, 0, z]
                    gt_z = gt_bin[b, 0, z] > 0
                    denom = pred_z.float().sum() + gt_z.float().sum()
                    if denom <= 0:
                        dice = pred_z.new_tensor(1.0, dtype=torch.float32)
                    else:
                        inter = torch.logical_and(pred_z, gt_z).float().sum()
                        dice = (2.0 * inter) / (denom + 1e-6)
                    dice_values.append(dice)
                dice_values = torch.stack(dice_values)
                top_k = min(3, int(candidates.numel()))
                selected = candidates[torch.topk(dice_values, k=top_k, largest=False).indices]
                for z in selected:
                    pred_z = initial[b, 0, z]
                    gt_z = gt_bin[b, 0, z] > 0
                    pos_corr[b, 0, z] = torch.logical_and(gt_z, torch.logical_not(pred_z)).float()
                    neg_corr[b, 0, z] = torch.logical_and(pred_z, torch.logical_not(gt_z)).float()
        else:
            for b in range(data.shape[0]):
                pred = initial[b, 0]
                gt_b = gt_bin[b, 0] > 0
                pos_corr[b, 0] = torch.logical_and(gt_b, torch.logical_not(pred)).float()
                neg_corr[b, 0] = torch.logical_and(pred, torch.logical_not(gt_b)).float()

        v3_data = torch.cat((data[:, 0:1], data[:, 1:2], pos_corr, neg_corr), dim=1)
        assert v3_data.shape[1] == 4
        assert pos_corr.shape == data[:, 1:2].shape
        assert neg_corr.shape == data[:, 1:2].shape
        return v3_data

    def train_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        data = self._build_v3_1_correction_input(data, target)

        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(data)
            l = self.loss(output, target)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {"loss": l.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        data = self._build_v3_1_correction_input(data, target)

        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(data)
            del data
            l = self.loss(output, target)

        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]

        axes = [0] + list(range(2, output.ndim))
        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                target[target == self.label_manager.ignore_label] = 0
            else:
                if target.dtype == torch.bool:
                    mask = ~target[:, -1:]
                else:
                    mask = 1 - target[:, -1:]
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {"loss": l.detach().cpu().numpy(), "tp_hard": tp_hard, "fp_hard": fp_hard, "fn_hard": fn_hard}
