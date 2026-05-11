import torch.nn as nn

import segmentation_models_pytorch as smp


class ChangeDetectionModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.model = smp.Unet(

            encoder_name="resnet34",

            encoder_weights="imagenet",

            in_channels=6,

            classes=1

        )

    def forward(self, x):

        return self.model(x)