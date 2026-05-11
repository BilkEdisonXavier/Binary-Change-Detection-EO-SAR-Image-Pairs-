import albumentations as A

from albumentations.pytorch import ToTensorV2


def get_train_transforms(size):

    return A.Compose([

        A.Resize(size, size),

        A.HorizontalFlip(p=0.5),

        A.VerticalFlip(p=0.5),

        A.RandomRotate90(p=0.5),

        A.Normalize(),

        ToTensorV2()

    ])


def get_val_transforms(size):

    return A.Compose([

        A.Resize(size, size),

        A.Normalize(),

        ToTensorV2()

    ])