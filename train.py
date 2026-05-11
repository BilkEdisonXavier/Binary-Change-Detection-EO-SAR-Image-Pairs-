import os

print(os.path.exists("./dataset/train/pre_event"))
import yaml
import torch
import os

from tqdm import tqdm

from torch.utils.data import DataLoader

from datasets.change_dataset import (
    ChangeDetectionDataset
)

from utils.augmentations import (
    get_train_transforms,
    get_val_transforms
)

from models.unet_model import (
    ChangeDetectionModel
)

from losses.combined_loss import (
    CombinedLoss
)

from utils.metrics import (
    calculate_metrics
)


with open("config.yaml") as f:

    config = yaml.safe_load(f)


train_dataset = ChangeDetectionDataset(

    config["train_dir"],

    transform=get_train_transforms(
        config["image_size"]
    )
)

val_dataset = ChangeDetectionDataset(

    config["val_dir"],

    transform=get_val_transforms(
        config["image_size"]
    )
)


train_loader = DataLoader(

    train_dataset,

    batch_size=config["batch_size"],

    shuffle=True,

    num_workers=config["num_workers"]
)

val_loader = DataLoader(

    val_dataset,

    batch_size=1,

    shuffle=False
)


device = config["device"]

model = ChangeDetectionModel().to(device)

criterion = CombinedLoss()

optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=config["lr"]
)

best_iou = 0


for epoch in range(config["epochs"]):

    model.train()

    train_loss = 0

    loop = tqdm(train_loader)

    for images, masks in loop:

        images = images.to(device)

        masks = masks.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, masks)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        loop.set_description(
            f"Epoch [{epoch+1}/{config['epochs']}]"
        )

        loop.set_postfix(
            loss=loss.item()
        )

    model.eval()

    val_iou = []

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(device)

            masks = masks.to(device)

            outputs = model(images)

            _, _, _, iou, _ = \
                calculate_metrics(
                    outputs,
                    masks
                )

            val_iou.append(iou)

    avg_iou = sum(val_iou) / len(val_iou)

    print(
        f"Epoch {epoch+1} "
        f"Train Loss: {train_loss:.4f} "
        f"Val IoU: {avg_iou:.4f}"
    )

    if avg_iou > best_iou:

        best_iou = avg_iou

        torch.save(

            model.state_dict(),

            "checkpoints/best_model.pth"
        )

        print("Best model saved.")