import yaml
import torch

from torch.utils.data import DataLoader

from datasets.change_dataset import (
    ChangeDetectionDataset
)

from utils.augmentations import (
    get_val_transforms
)

from models.unet_model import (
    ChangeDetectionModel
)

from utils.metrics import (
    calculate_metrics
)


with open("config.yaml") as f:

    config = yaml.safe_load(f)


test_dataset = ChangeDetectionDataset(

    config["test_dir"],

    transform=get_val_transforms(
        config["image_size"]
    )
)

test_loader = DataLoader(

    test_dataset,

    batch_size=1,

    shuffle=False
)


device = config["device"]

model = ChangeDetectionModel().to(device)

model.load_state_dict(
    torch.load(
        "checkpoints/best_model.pth"
    )
)

model.eval()


precision_scores = []
recall_scores = []
f1_scores = []
iou_scores = []


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

        precision_scores.append(precision)

        recall_scores.append(recall)

        f1_scores.append(f1)

        iou_scores.append(iou)


print("Precision:",
      sum(precision_scores) / len(precision_scores))

print("Recall:",
      sum(recall_scores) / len(recall_scores))

print("F1:",
      sum(f1_scores) / len(f1_scores))

print("IoU:",
      sum(iou_scores) / len(iou_scores))