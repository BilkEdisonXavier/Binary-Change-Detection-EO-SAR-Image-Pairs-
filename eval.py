import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from datasets.change_dataset import (
    ChangeDetectionDataset
)

from models.unet_model import (
    ChangeDetectionModel
)

from utils.metrics import (
    calculate_metrics
)


def main():

    # LOAD CONFIG

    with open("config.yaml") as f:

        config = yaml.safe_load(f)

    # LOAD TEST DATASET

    test_dataset = ChangeDetectionDataset(
        config["test_dir"]
    )

    print(
        "Test Dataset Size:",
        len(test_dataset)
    )

    if len(test_dataset) == 0:

        print(
            "ERROR: Test dataset is empty."
        )

        return

    # CREATE DATALOADER

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    # DEVICE

    device = "cpu"

    # LOAD MODEL

    model = ChangeDetectionModel().to(device)

    model.load_state_dict(
        torch.load(
            "checkpoints/best_model.pth",
            map_location=device
        )
    )

    model.eval()

    # METRIC STORAGE

    precision_scores = []
    recall_scores = []
    f1_scores = []
    iou_scores = []

    total_cm = np.zeros((2, 2))

    # EVALUATION LOOP

    with torch.no_grad():

        for images, masks in test_loader:

            images = images.to(device)

            masks = masks.to(device)

            outputs = model(images)

            precision, recall, f1, iou, cm = \
                calculate_metrics(
                    outputs,
                    masks
                )

            precision_scores.append(
                precision
            )

            recall_scores.append(
                recall
            )

            f1_scores.append(
                f1
            )

            iou_scores.append(
                iou
            )

            # HANDLE CONFUSION MATRIX SAFELY

            if cm.shape == (2, 2):

                total_cm += cm

    # FINAL METRICS

    print("\n===== FINAL RESULTS =====\n")

    print(
        "Precision:",
        np.mean(precision_scores)
    )

    print(
        "Recall:",
        np.mean(recall_scores)
    )

    print(
        "F1 Score:",
        np.mean(f1_scores)
    )

    print(
        "IoU:",
        np.mean(iou_scores)
    )

    # PLOT CONFUSION MATRIX

    plt.figure(figsize=(6, 6))

    plt.imshow(
        total_cm,
        cmap="Blues"
    )

    plt.title(
        "Confusion Matrix"
    )

    plt.colorbar()

    plt.xticks(
        [0, 1],
        ["No Change", "Change"]
    )

    plt.yticks(
        [0, 1],
        ["No Change", "Change"]
    )

    # ADD VALUES INSIDE MATRIX

    for i in range(2):

        for j in range(2):

            plt.text(
                j,
                i,
                int(total_cm[i, j]),
                ha="center",
                va="center",
                color="black"
            )

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "Actual"
    )

    plt.tight_layout()

    # SAVE IMAGE

    plt.savefig(
        "confusion_matrix.png"
    )

    print(
        "\nconfusion_matrix.png saved successfully."
    )


if __name__ == "__main__":

    main()