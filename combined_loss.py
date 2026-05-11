import torch.nn as nn

from .dice_loss import DiceLoss


class CombinedLoss(nn.Module):

    def __init__(self):

        super().__init__()

        self.bce = nn.BCEWithLogitsLoss()

        self.dice = DiceLoss()

    def forward(self, preds, targets):

        bce_loss = self.bce(
            preds,
            targets
        )

        dice_loss = self.dice(
            preds,
            targets
        )

        return bce_loss + dice_loss