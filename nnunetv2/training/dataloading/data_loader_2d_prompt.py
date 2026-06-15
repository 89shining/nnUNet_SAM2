import numpy as np
import torch
from threadpoolctl import threadpool_limits

from nnunetv2.training.dataloading.data_loader_2d import nnUNetDataLoader2D


class nnUNetDataLoader2DPrompt(nnUNetDataLoader2D):
    def generate_train_batch(self):
        selected_keys = self.get_indices()
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)
        prompt_shape = (self.batch_size, 2, *self.patch_size)
        prompt_all = np.zeros(prompt_shape, dtype=np.int16)

        for j, current_key in enumerate(selected_keys):
            force_fg = self.get_do_oversample(j)
            data, seg, properties = self._data.load_case(current_key)
            prompt = self._data.load_prompt(current_key)
            if prompt.shape[1:] != data.shape[1:]:
                raise RuntimeError(
                    f"Prompt/data spatial shape mismatch for case '{current_key}': prompt={prompt.shape}, data={data.shape}"
                )

            if not force_fg:
                if self.has_ignore:
                    selected_class_or_region = (
                        self.annotated_classes_key
                        if len(properties["class_locations"][self.annotated_classes_key]) > 0
                        else None
                    )
                else:
                    selected_class_or_region = None
            else:
                eligible_classes_or_regions = [
                    i for i in properties["class_locations"].keys() if len(properties["class_locations"][i]) > 0
                ]
                tmp = [i == self.annotated_classes_key if isinstance(i, tuple) else False for i in eligible_classes_or_regions]
                if any(tmp) and len(eligible_classes_or_regions) > 1:
                    eligible_classes_or_regions.pop(np.where(tmp)[0][0])
                selected_class_or_region = (
                    eligible_classes_or_regions[np.random.choice(len(eligible_classes_or_regions))]
                    if len(eligible_classes_or_regions) > 0
                    else None
                )

            if selected_class_or_region is not None:
                selected_slice = np.random.choice(properties["class_locations"][selected_class_or_region][:, 1])
            else:
                selected_slice = np.random.choice(len(data[0]))

            data = data[:, selected_slice]
            seg = seg[:, selected_slice]
            prompt = prompt[:, selected_slice]

            class_locations = (
                {
                    selected_class_or_region: properties["class_locations"][selected_class_or_region][
                        properties["class_locations"][selected_class_or_region][:, 1] == selected_slice
                    ][:, (0, 2, 3)]
                }
                if (selected_class_or_region is not None)
                else None
            )

            shape = data.shape[1:]
            dim = len(shape)
            bbox_lbs, bbox_ubs = self.get_bbox(
                shape,
                force_fg if selected_class_or_region is not None else False,
                class_locations,
                overwrite_class=selected_class_or_region,
            )

            valid_bbox_lbs = [max(0, bbox_lbs[i]) for i in range(dim)]
            valid_bbox_ubs = [min(shape[i], bbox_ubs[i]) for i in range(dim)]

            data_slice = tuple([slice(0, data.shape[0])] + [slice(a, b) for a, b in zip(valid_bbox_lbs, valid_bbox_ubs)])
            seg_slice = tuple([slice(0, seg.shape[0])] + [slice(a, b) for a, b in zip(valid_bbox_lbs, valid_bbox_ubs)])
            prompt_slice = tuple(
                [slice(0, prompt.shape[0])] + [slice(a, b) for a, b in zip(valid_bbox_lbs, valid_bbox_ubs)]
            )

            data = data[data_slice]
            seg = seg[seg_slice]
            prompt = prompt[prompt_slice]

            padding = [(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0)) for i in range(dim)]
            data_all[j] = np.pad(data, ((0, 0), *padding), "constant", constant_values=0)
            seg_all[j] = np.pad(seg, ((0, 0), *padding), "constant", constant_values=-1)
            prompt_all[j] = np.pad(prompt, ((0, 0), *padding), "constant", constant_values=0)

        if self.transforms is not None:
            with torch.no_grad():
                with threadpool_limits(limits=1, user_api=None):
                    data_all = torch.from_numpy(data_all).float()
                    seg_prompt_all = torch.from_numpy(np.concatenate((seg_all, prompt_all), axis=1)).to(torch.int16)
                    images = []
                    segs = []
                    for b in range(self.batch_size):
                        tmp = self.transforms(**{"image": data_all[b], "segmentation": seg_prompt_all[b]})
                        images.append(tmp["image"])
                        segs.append(tmp["segmentation"])
                    data_all = torch.stack(images)
                    if isinstance(segs[0], list):
                        seg_all = [torch.stack([s[i][: self.seg_shape[1]] for s in segs]) for i in range(len(segs[0]))]
                        prompt_all = torch.stack([s[0][self.seg_shape[1] : self.seg_shape[1] + 2] for s in segs]).float()
                    else:
                        seg_prompt_all = torch.stack(segs)
                        prompt_all = seg_prompt_all[:, self.seg_shape[1] : self.seg_shape[1] + 2].float()
                        seg_all = seg_prompt_all[:, : self.seg_shape[1]]
                    del segs, images
            return {"data": data_all, "target": seg_all, "prompt": prompt_all, "keys": selected_keys}

        return {
            "data": torch.from_numpy(data_all).float(),
            "target": torch.from_numpy(seg_all).to(torch.int16),
            "prompt": torch.from_numpy(prompt_all).float(),
            "keys": selected_keys,
        }
