import os
import cv2
import torch
import numpy as np

from torch.utils.data import Dataset


class ChangeDetectionDataset(Dataset):

    def __init__(self, root_dir):

        self.root_dir = root_dir

        self.pre_dir = os.path.join(
            root_dir,
            "pre_event"
        )

        self.post_dir = os.path.join(
            root_dir,
            "post_event"
        )

        self.target_dir = os.path.join(
            root_dir,
            "target"
        )

        self.images = sorted(
            os.listdir(self.pre_dir)
        )

    def __len__(self):

        return len(self.images)

    def remap_mask(self, mask):

        binary_mask = np.zeros_like(mask)

        binary_mask[mask == 2] = 1
        binary_mask[mask == 3] = 1

        return binary_mask

    def __getitem__(self, idx):

        filename = self.images[idx]

        pre_path = os.path.join(
            self.pre_dir,
            filename
        )

        post_path = os.path.join(
            self.post_dir,
            filename
        )

        target_path = os.path.join(
            self.target_dir,
            filename
        )

        pre = cv2.imread(pre_path)

        post = cv2.imread(post_path)

        mask = cv2.imread(
            target_path,
            cv2.IMREAD_GRAYSCALE
        )

        pre = cv2.cvtColor(
            pre,
            cv2.COLOR_BGR2RGB
        )

        post = cv2.cvtColor(
            post,
            cv2.COLOR_BGR2RGB
        )

        image = np.concatenate(
            [pre, post],
            axis=-1
        )

        image = image / 255.0

        image = torch.tensor(
            image,
            dtype=torch.float32
        ).permute(2, 0, 1)

        mask = self.remap_mask(mask)

        mask = torch.tensor(
            mask,
            dtype=torch.float32
        ).unsqueeze(0)

        return image, mask