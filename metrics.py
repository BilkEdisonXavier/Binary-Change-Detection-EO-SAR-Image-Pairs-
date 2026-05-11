import torch

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def calculate_metrics(preds, masks):

    preds = torch.sigmoid(preds)

    preds = (preds > 0.5).float()

    preds = preds.cpu().numpy().flatten()

    masks = masks.cpu().numpy().flatten()

    precision = precision_score(
        masks,
        preds,
        zero_division=0
    )

    recall = recall_score(
        masks,
        preds,
        zero_division=0
    )

    f1 = f1_score(
        masks,
        preds,
        zero_division=0
    )

    intersection = (
        preds * masks
    ).sum()

    union = (
        preds + masks
    ).sum() - intersection

    iou = (
        intersection + 1e-6
    ) / (
        union + 1e-6
    )

    cm = confusion_matrix(
        masks,
        preds
    )

    return precision, recall, f1, iou, cm