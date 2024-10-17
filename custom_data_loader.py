import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class BraTS_dataset(Dataset):
    def __init__(self, img_dir, img_list, mask_dir, mask_list):
        self.img_dir = img_dir
        self.img_list = img_list
        self.mask_dir = mask_dir
        self.mask_list = mask_list

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_name = self.img_list[index]
        img_path = os.path.join(self.img_dir, img_name)

        mask_name = self.mask_list[index]
        mask_path = os.path.join(self.mask_dir, mask_name)

        if not os.path.exists(img_path) or os.path.getsize(img_path) == 0:
            raise ValueError(f"Image file not found or empty: {img_path}")

        if not os.path.exists(mask_path) or os.path.getsize(mask_path) == 0:
            raise ValueError(f"Mask file not found or empty: {mask_path}")

        try:
            img = np.load(img_path)
        except Exception as e:
            raise ValueError(f"Error loading image file: {img_path}, {e}")

        try:
            mask = np.load(mask_path)
        except Exception as e:
            raise ValueError(f"Error loading mask file: {mask_path}, {e}")


def image_loader(img_dir, img_list, mask_dir, mask_list, batch_size, num_workers=24):
    dataset = BraTS_dataset(img_dir, img_list, mask_dir, mask_list)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers)
    return data_loader
