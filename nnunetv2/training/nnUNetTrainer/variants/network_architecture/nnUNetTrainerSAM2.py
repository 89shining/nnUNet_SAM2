import pydoc
import warnings
from importlib import import_module
from time import sleep
from typing import List, Tuple, Union

import numpy as np
import torch
from torch import autocast, nn

from nnunetv2.training.dataloading.data_loader_2d_prompt import nnUNetDataLoader2DPrompt
from nnunetv2.training.dataloading.data_loader_3d_prompt import nnUNetDataLoader3DPrompt
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from nnunetv2.training.dataloading.nnunet_dataset_prompt import nnUNetPromptDataset
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.sam2_nnunet_arch import (
    SAM2DualEncoderResidualUNet,
    get_sam2_cfg_from_env,
    get_sam2_checkpoint_from_env,
)
from nnunetv2.utilities.helpers import dummy_context

nnunet_trainer_module = import_module("nnunetv2.training.nnUNetTrainer.nnUNetTrainer")


class nnUNetTrainerSAM2(nnUNetTrainer):
    """
    nnUNet trainer using native nnUNet encoder/decoder plus SAM2 auxiliary encoder branch.

    v3-1 keeps the preprocessed main nnU-Net data at two channels:
      data[:, 0:1] = CT
      data[:, 1:2] = v2-1 initial CTV mask

    Case-level oracle correction prompts are stored as preprocessed sidecar files
    `<case_id>_v3_prompt.npy` with two channels:
      prompt[:, 0:1] = FN positive correction mask
      prompt[:, 1:2] = FP negative correction mask

    The dataloader crops/pads those prompt sidecars with the exact same bbox as the
    sampled training patch and concatenates them online right before the network
    forward, so the main nnU-Net dataset semantics remain two-channel.
    """

    prompt_file_suffix = "_v3_prompt.npy"

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
    def _concat_prompt_tensor(data: torch.Tensor, prompt: torch.Tensor, target) -> torch.Tensor:
        if data.shape[1] < 2:
            raise RuntimeError(f"v3-1 expects at least 2 data channels [CT, initial_mask], got {tuple(data.shape)}")
        if prompt.shape[1] != 2:
            raise RuntimeError(f"v3-1 prompt sidecar must provide exactly 2 channels, got {tuple(prompt.shape)}")
        if prompt.shape[2:] != data.shape[2:]:
            raise RuntimeError(f"Prompt shape {tuple(prompt.shape)} must match data shape {tuple(data.shape)}")

        gt = target[0] if isinstance(target, list) else target
        if gt.shape[2:] != data.shape[2:]:
            raise RuntimeError(f"Target shape {tuple(gt.shape)} must match data spatial shape {tuple(data.shape)}")

        return torch.cat((data[:, :2], prompt, data[:, 2:]), dim=1)

    @staticmethod
    def _concat_prompt_array(data: np.ndarray, prompt: np.ndarray) -> np.ndarray:
        if data.shape[0] < 2:
            raise RuntimeError(f"v3-1 expects at least 2 data channels [CT, initial_mask], got {tuple(data.shape)}")
        if prompt.shape[0] != 2:
            raise RuntimeError(f"v3-1 prompt sidecar must provide exactly 2 channels, got {tuple(prompt.shape)}")
        if prompt.shape[1:] != data.shape[1:]:
            raise RuntimeError(f"Prompt shape {tuple(prompt.shape)} must match data shape {tuple(data.shape)}")
        return np.concatenate((data[:2], prompt.astype(data.dtype, copy=False), data[2:]), axis=0)

    def get_tr_and_val_datasets(self):
        tr_keys, val_keys = self.do_split()
        dataset_tr = nnUNetPromptDataset(
            self.preprocessed_dataset_folder,
            tr_keys,
            folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
            num_images_properties_loading_threshold=0,
            prompt_file_suffix=self.prompt_file_suffix,
        )
        dataset_val = nnUNetPromptDataset(
            self.preprocessed_dataset_folder,
            val_keys,
            folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
            num_images_properties_loading_threshold=0,
            prompt_file_suffix=self.prompt_file_suffix,
        )
        return dataset_tr, dataset_val

    def get_dataloaders(self):
        patch_size = self.configuration_manager.patch_size
        dim = len(patch_size)
        deep_supervision_scales = self._get_deep_supervision_scales()

        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        tr_transforms = self.get_training_transforms(
            patch_size,
            rotation_for_DA,
            deep_supervision_scales,
            mirror_axes,
            do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label,
        )
        val_transforms = self.get_validation_transforms(
            deep_supervision_scales,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label,
        )

        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        if dim == 2:
            dl_tr = nnUNetDataLoader2DPrompt(
                dataset_tr,
                self.batch_size,
                initial_patch_size,
                self.configuration_manager.patch_size,
                self.label_manager,
                oversample_foreground_percent=self.oversample_foreground_percent,
                sampling_probabilities=None,
                pad_sides=None,
                transforms=tr_transforms,
            )
            dl_val = nnUNetDataLoader2DPrompt(
                dataset_val,
                self.batch_size,
                self.configuration_manager.patch_size,
                self.configuration_manager.patch_size,
                self.label_manager,
                oversample_foreground_percent=self.oversample_foreground_percent,
                sampling_probabilities=None,
                pad_sides=None,
                transforms=val_transforms,
            )
        else:
            dl_tr = nnUNetDataLoader3DPrompt(
                dataset_tr,
                self.batch_size,
                initial_patch_size,
                self.configuration_manager.patch_size,
                self.label_manager,
                oversample_foreground_percent=self.oversample_foreground_percent,
                sampling_probabilities=None,
                pad_sides=None,
                transforms=tr_transforms,
            )
            dl_val = nnUNetDataLoader3DPrompt(
                dataset_val,
                self.batch_size,
                self.configuration_manager.patch_size,
                self.configuration_manager.patch_size,
                self.label_manager,
                oversample_foreground_percent=self.oversample_foreground_percent,
                sampling_probabilities=None,
                pad_sides=None,
                transforms=val_transforms,
            )

        allowed_num_processes = nnunet_trainer_module.get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = nnunet_trainer_module.SingleThreadedAugmenter(dl_tr, None)
            mt_gen_val = nnunet_trainer_module.SingleThreadedAugmenter(dl_val, None)
        else:
            mt_gen_train = nnunet_trainer_module.NonDetMultiThreadedAugmenter(
                data_loader=dl_tr,
                transform=None,
                num_processes=allowed_num_processes,
                num_cached=max(6, allowed_num_processes // 2),
                seeds=None,
                pin_memory=self.device.type == "cuda",
                wait_time=0.002,
            )
            mt_gen_val = nnunet_trainer_module.NonDetMultiThreadedAugmenter(
                data_loader=dl_val,
                transform=None,
                num_processes=max(1, allowed_num_processes // 2),
                num_cached=max(3, allowed_num_processes // 4),
                seeds=None,
                pin_memory=self.device.type == "cuda",
                wait_time=0.002,
            )
        _ = next(mt_gen_train)
        _ = next(mt_gen_val)
        return mt_gen_train, mt_gen_val

    def train_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        prompt = batch["prompt"].to(self.device, non_blocking=True)
        target = batch["target"]
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        data = self._concat_prompt_tensor(data, prompt, target)

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
        prompt = batch["prompt"].to(self.device, non_blocking=True)
        target = batch["target"]
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        data = self._concat_prompt_tensor(data, prompt, target)

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

    def perform_actual_validation(self, save_probabilities: bool = False):
        self.set_deep_supervision_enabled(False)
        self.network.eval()

        if self.is_ddp and self.batch_size == 1 and self.enable_deep_supervision and self._do_i_compile():
            self.print_to_log_file(
                "WARNING! batch size is 1 during training and torch.compile is enabled. If you "
                "encounter crashes in validation then this is because torch.compile forgets "
                "to trigger a recompilation of the model with deep supervision disabled. "
                "This causes torch.flip to complain about getting a tuple as input. Just rerun the "
                "validation with --val (exactly the same as before) and then it will work. "
                "Why? Because --val triggers nnU-Net to ONLY run validation meaning that the first "
                "forward pass (where compile is triggered) already has deep supervision disabled. "
                "This is exactly what we need in perform_actual_validation"
            )

        predictor = nnunet_trainer_module.nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=True,
            device=self.device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=False,
        )
        predictor.manual_initialization(
            self.network,
            self.plans_manager,
            self.configuration_manager,
            None,
            self.dataset_json,
            self.__class__.__name__,
            self.inference_allowed_mirroring_axes,
        )

        with nnunet_trainer_module.multiprocessing.get_context("spawn").Pool(
            nnunet_trainer_module.default_num_processes
        ) as segmentation_export_pool:
            worker_list = [i for i in segmentation_export_pool._pool]
            validation_output_folder = nnunet_trainer_module.join(self.output_folder, "validation")
            nnunet_trainer_module.maybe_mkdir_p(validation_output_folder)

            _, val_keys = self.do_split()
            if self.is_ddp:
                last_barrier_at_idx = len(val_keys) // nnunet_trainer_module.dist.get_world_size() - 1
                val_keys = val_keys[self.local_rank :: nnunet_trainer_module.dist.get_world_size()]

            dataset_val = nnUNetPromptDataset(
                self.preprocessed_dataset_folder,
                val_keys,
                folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                num_images_properties_loading_threshold=0,
                prompt_file_suffix=self.prompt_file_suffix,
            )

            next_stages = self.configuration_manager.next_stage_names
            if next_stages is not None:
                _ = [
                    nnunet_trainer_module.maybe_mkdir_p(
                        nnunet_trainer_module.join(self.output_folder_base, "predicted_next_stage", n)
                    )
                    for n in next_stages
                ]

            results = []
            for i, k in enumerate(dataset_val.keys()):
                proceed = not nnunet_trainer_module.check_workers_alive_and_busy(
                    segmentation_export_pool, worker_list, results, allowed_num_queued=2
                )
                while not proceed:
                    sleep(0.1)
                    proceed = not nnunet_trainer_module.check_workers_alive_and_busy(
                        segmentation_export_pool, worker_list, results, allowed_num_queued=2
                    )

                self.print_to_log_file(f"predicting {k}")
                data, seg, properties = dataset_val.load_case(k)
                prompt = dataset_val.load_prompt(k)

                if self.is_cascaded:
                    data = np.vstack(
                        (
                            data,
                            nnunet_trainer_module.convert_labelmap_to_one_hot(
                                seg[-1], self.label_manager.foreground_labels, output_dtype=data.dtype
                            ),
                        )
                    )
                data = self._concat_prompt_array(data, prompt)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    data = torch.from_numpy(data)

                self.print_to_log_file(f"{k}, shape {data.shape}, rank {self.local_rank}")
                output_filename_truncated = nnunet_trainer_module.join(validation_output_folder, k)

                prediction = predictor.predict_sliding_window_return_logits(data)
                prediction = prediction.cpu()

                results.append(
                    segmentation_export_pool.starmap_async(
                        nnunet_trainer_module.export_prediction_from_logits,
                        (
                            (
                                prediction,
                                properties,
                                self.configuration_manager,
                                self.plans_manager,
                                self.dataset_json,
                                output_filename_truncated,
                                save_probabilities,
                            ),
                        ),
                    )
                )

                if next_stages is not None:
                    for n in next_stages:
                        next_stage_config_manager = self.plans_manager.get_configuration(n)
                        expected_preprocessed_folder = nnunet_trainer_module.join(
                            nnunet_trainer_module.nnUNet_preprocessed,
                            self.plans_manager.dataset_name,
                            next_stage_config_manager.data_identifier,
                        )

                        try:
                            tmp = nnUNetDataset(expected_preprocessed_folder, [k], num_images_properties_loading_threshold=0)
                            d, s, p = tmp.load_case(k)
                        except FileNotFoundError:
                            self.print_to_log_file(
                                f"Predicting next stage {n} failed for case {k} because the preprocessed file is missing! "
                                f"Run the preprocessing for this configuration first!"
                            )
                            continue

                        target_shape = d.shape[1:]
                        output_folder = nnunet_trainer_module.join(self.output_folder_base, "predicted_next_stage", n)
                        output_file = nnunet_trainer_module.join(output_folder, k + ".npz")
                        results.append(
                            segmentation_export_pool.starmap_async(
                                nnunet_trainer_module.resample_and_save,
                                (
                                    (
                                        prediction,
                                        target_shape,
                                        output_file,
                                        self.plans_manager,
                                        self.configuration_manager,
                                        properties,
                                        self.dataset_json,
                                    ),
                                ),
                            )
                        )

                if self.is_ddp and i < last_barrier_at_idx and (i + 1) % 20 == 0:
                    nnunet_trainer_module.dist.barrier()

            _ = [r.get() for r in results]

        if self.is_ddp:
            nnunet_trainer_module.dist.barrier()

        if self.local_rank == 0:
            metrics = nnunet_trainer_module.compute_metrics_on_folder(
                nnunet_trainer_module.join(self.preprocessed_dataset_folder_base, "gt_segmentations"),
                validation_output_folder,
                nnunet_trainer_module.join(validation_output_folder, "summary.json"),
                self.plans_manager.image_reader_writer_class(),
                self.dataset_json["file_ending"],
                self.label_manager.foreground_regions if self.label_manager.has_regions else self.label_manager.foreground_labels,
                self.label_manager.ignore_label,
                chill=True,
                num_processes=(
                    nnunet_trainer_module.default_num_processes * nnunet_trainer_module.dist.get_world_size()
                    if self.is_ddp
                    else nnunet_trainer_module.default_num_processes
                ),
            )
            self.print_to_log_file("Validation complete", also_print_to_console=True)
            self.print_to_log_file("Mean Validation Dice: ", metrics["foreground_mean"]["Dice"], also_print_to_console=True)

        self.set_deep_supervision_enabled(True)
        nnunet_trainer_module.compute_gaussian.cache_clear()
