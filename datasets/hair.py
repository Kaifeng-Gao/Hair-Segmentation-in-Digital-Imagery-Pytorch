import glob
import os
import zipfile

import numpy as np
from PIL import Image
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url


def get_class_label(filename, binary=True):
    """
    0: straight: frame00001-00150
    1: wavy: frame00151-00300
    2: curly: frame00301-00450
    3: kinky: frame00451-00600
    4: braids: frame00601-00750
    5: dreadlocks: frame00751-00900
    6: short-men: frame00901-01050
    """
    idx = int(filename.strip('Frame').strip('-gt.pbm'))
    if binary:
        return 1
    else:
        if 0 < idx <= 150:
            return 1
        elif 150 < idx <= 300:
            return 2
        elif 300 < idx <= 450:
            return 3
        elif 450 < idx <= 600:
            return 4
        elif 600 < idx <= 750:
            return 5
        elif 750 < idx <= 900:
            return 6
        elif 900 < idx <= 1050:
            return 7
        raise ValueError


class FigaroDataset(Dataset):
    cmap = np.array([
                [0, 0, 0],  # background
                [128, 128, 128],  # hair
            ], dtype=np.uint8)

    def __init__(self, root_dir, mode='train', download=False, joint_transforms=None,
                 image_transforms=None, mask_transforms=None):
        """
        Args:
            root_dir (str): root directory of dataset
            joint_transforms (torchvision.transforms.Compose): tranformation on both data and target
            image_transforms (torchvision.transforms.Compose): tranformation only on data
            mask_transforms (torchvision.transforms.Compose): tranformation only on target
        """
        self.root = os.path.expanduser(root_dir)
        self.url = "http://projects.i-ctm.eu/sites/default/files/AltroMateriale/207_Michele%20Svanera/Figaro1k.zip"
        self.filename = "Figaro1k.zip"
        self.md5 = None

        if download:
            download_extract(self.url, self.root, self.filename, self.md5)

        if mode not in ['train', 'val', 'test']:
            raise ValueError("Mode must be 'train', 'val', or 'test'")
        if mode == 'train' or mode == 'val':
            img_dir = os.path.join(root_dir, 'Figaro1k', 'Original', 'Training')
            mask_dir = os.path.join(root_dir, 'Figaro1k', 'GT', 'Training')
        elif mode == 'test':
            img_dir = os.path.join(root_dir, 'Figaro1k', 'Original', 'Testing')
            mask_dir = os.path.join(root_dir, 'Figaro1k', 'GT', 'Testing')
        img_path_list = [os.path.join(img_dir, img) for img in sorted(os.listdir(img_dir))]
        mask_path_list = [os.path.join(mask_dir, mask) for mask in sorted(os.listdir(mask_dir))]

        train_imgs, val_imgs, train_masks, val_masks = train_test_split(img_path_list,
                                                                        mask_path_list,
                                                                        test_size=0.2,
                                                                        random_state=42)
        if mode == 'train':
            self.img_path_list = train_imgs
            self.mask_path_list = train_masks
        elif mode == 'val':
            self.img_path_list = val_imgs
            self.mask_path_list = val_masks
        else:
            self.img_path_list = img_path_list
            self.mask_path_list = mask_path_list

        self.joint_transforms = joint_transforms
        self.image_transforms = image_transforms
        self.mask_transforms = mask_transforms

    def __getitem__(self, idx):
        img_path = self.img_path_list[idx]
        img = Image.open(img_path)

        mask_path = self.mask_path_list[idx]
        mask = Image.open(mask_path)
        class_label = get_class_label(os.path.basename(mask_path), binary=True)
        mask_array = np.array(mask, dtype=np.uint8)
        class_mask = np.zeros_like(mask_array)
        class_mask[mask_array > 0] = class_label
        class_mask = Image.fromarray(class_mask)

        if self.joint_transforms is not None:
            img, class_mask = self.joint_transforms(img, class_mask)

        if self.image_transforms is not None:
            img = self.image_transforms(img)

        if self.mask_transforms is not None:
            class_mask = self.mask_transforms(class_mask)
            if class_mask.ndim == 3 and class_mask.shape[0] == 1:  # Check if mask has a channel dimension
                class_mask = class_mask.squeeze(0)  # Remove the channel dimension
        else:
            class_mask = torch.from_numpy(np.array(class_mask, dtype=np.int64))
            if class_mask.ndim == 3 and class_mask.shape[0] == 1:  # Check if mask has a channel dimension
                class_mask = class_mask.squeeze(0)  # Remove the channel dimension

        return img, class_mask

    def __len__(self):
        return len(self.mask_path_list)

    @classmethod
    def decode_target(cls, mask):
        """decode semantic mask to RGB image"""
        return cls.cmap[mask]


def download_extract(url, root, filename, md5):
    file_path = os.path.join(root, filename)
    download_url(url, root, filename, md5)

    data_dir = os.path.join(root, "Figaro1k")
    if os.path.isdir(data_dir):
        for r, dirs, files in os.walk(data_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(r, name))
            for name in dirs:
                os.rmdir(os.path.join(r, name))
        os.rmdir(data_dir)

    print("Unzip Figaro1k.zip ...")

    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(root)

    print("Removing unnecessary files ...")
    dir_path = os.path.join(root, "Figaro1k/GT/Training")

    pattern = os.path.join(dir_path, "*(1).pbm")
    files_to_remove = glob.glob(pattern)
    for file_path in files_to_remove:
        os.remove(file_path)

    os.remove(os.path.join(root, "Figaro1k/.DS_Store"))
    macosx_dir = os.path.join(root, "__MACOSX")
    if os.path.isdir(macosx_dir):
        for root, dirs, files in os.walk(macosx_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(macosx_dir)
    print("Finished Downloading!")


if __name__ == "__main__":
    # Assuming the root directory you want to use is "./data"
    root_dir = "./data"
    # Instantiate the dataset object with download=True to trigger the download
    dataset = FigaroDataset(root_dir=root_dir, download=False)
    img, mask = dataset.__getitem__(200)
