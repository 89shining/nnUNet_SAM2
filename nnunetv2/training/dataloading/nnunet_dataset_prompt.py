from typing import List

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import isfile, join

from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset


class nnUNetPromptDataset(nnUNetDataset):
    def __init__(
        self,
        folder: str,
        case_identifiers: List[str] = None,
        num_images_properties_loading_threshold: int = 0,
        folder_with_segs_from_previous_stage: str = None,
        prompt_file_suffix: str = "_v3_prompt.npy",
    ):
        super().__init__(
            folder=folder,
            case_identifiers=case_identifiers,
            num_images_properties_loading_threshold=num_images_properties_loading_threshold,
            folder_with_segs_from_previous_stage=folder_with_segs_from_previous_stage,
        )
        self.prompt_file_suffix = prompt_file_suffix
        for c in self.dataset.keys():
            self.dataset[c]["prompt_file"] = join(folder, f"{c}{prompt_file_suffix}")

    def load_prompt(self, key: str) -> np.ndarray:
        entry = self[key]
        if "open_prompt_file" in entry.keys():
            prompt = entry["open_prompt_file"]
        else:
            prompt_file = entry["prompt_file"]
            if not isfile(prompt_file):
                raise FileNotFoundError(
                    f"Missing v3-1 prompt sidecar for case '{key}': {prompt_file}. "
                    "Generate it in the preprocessed folder before training/validation."
                )
            prompt = np.load(prompt_file, mmap_mode="r")
            if self.keep_files_open:
                self.dataset[key]["open_prompt_file"] = prompt

        if prompt.ndim < 3 or prompt.shape[0] != 2:
            raise RuntimeError(
                f"Prompt sidecar for case '{key}' must have shape [2, ...], got {tuple(prompt.shape)}"
            )
        return prompt
