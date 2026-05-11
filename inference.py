import cv2
import torch
import yaml
import matplotlib.pyplot as plt

from datasets.change_dataset import ChangeDetectionDataset 

from utils.augmentations import get_val_transforms

from models.unet_model import ChangeDetectionModel


with open("config.yaml") as f:

    config = yaml.safe_load(f)


dataset = ChangeDetectionDataset(

    config["test_dir"],

    transform=get_val_transforms(
        config["image_size"]
    )
)


device = config["device"]

model = ChangeDetectionModel().to(device)

model.load_state_dict(
    torch.load(
        "checkpoints/best_model.pth"
    )
)

model.eval()


image, mask = dataset[0]

input_image = image.unsqueeze(0).to(device)

with torch.no_grad():

    pred = model(input_image)

pred = torch.sigmoid(pred)

pred = (pred > 0.5).float()

pred = pred.squeeze().cpu().numpy()

mask = mask.squeeze().numpy()


plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.imshow(mask, cmap="gray")
plt.title("Ground Truth")

plt.subplot(1,2,2)
plt.imshow(pred, cmap="gray")
plt.title("Prediction")

plt.show()