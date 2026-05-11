import os
import cv2
import torch
import numpy as np

from torch.utils.data import Dataset


class ChangeDetectionDataset(Dataset):

    def __init__(self, root_dir, transform=None):

        self.root_dir = root_dir

        self.transform = transform

        self.images = sorted(
            os.listdir(
                os.path.join(
                    root_dir,
                    "pre_event"
                )
            )
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
            self.root_dir,
            "pre_event",
            filename
        )

        post_path = os.path.join(
            self.root_dir,
            "post_event",
            filename
        )

        target_path = os.path.join(
            self.root_dir,
            "target",
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

        mask = self.remap_mask(mask)

        if self.transform:

            augmented = self.transform(
                image=image,
                mask=mask
            )

            image = augmented["image"]

            mask = augmented["mask"]

        mask = mask.unsqueeze(0).float()

        return image.float(), mask