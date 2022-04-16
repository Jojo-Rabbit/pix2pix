import numpy as np
from PIL import Image
import os
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


transform_only_input = A.Compose(
    [
      A.Resize(height=256, width=256),
      A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
      ToTensorV2(),
    ]
)

transform_only_mask = A.Compose(
    [
        A.Resize(height=256, width=256),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)


class AnimeDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        image_file = self.list_files[index]
        image_path = os.path.join(self.root_dir, image_file)
        image = np.array(Image.open(image_path))
        input_image = image[:, 512:, :]
        target_image = image[:, :512, :]

        input_image = transform_only_input(image=input_image)["image"]
        target_image = transform_only_mask(image=target_image)["image"]

        return input_image, target_image


class MapDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        image_file = self.list_files[index]
        image_path = os.path.join(self.root_dir, image_file)
        image = np.array(Image.open(image_path))
        input_image = image[:, 600:, :]
        target_image = image[:, :600, :]

        input_image = transform_only_input(image=input_image)["image"]
        target_image = transform_only_mask(image=target_image)["image"]

        return input_image, target_image
