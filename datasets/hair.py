import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class FigaroDataset(Dataset):
    
    cmap = np.array([
                        [0, 0, 0],        # 背景
                        [128, 0, 0],      # straight
                        [0, 128, 0],      # wavy
                        [128, 128, 0],    # curly
                        [0, 0, 128],      # kinky
                        [128, 0, 128],    # braids
                        [0, 128, 128],    # dreadlocks
                        [128, 128, 128],  # short-men
                        # ... 以此类推，为每个类别定义颜色
                    ], dtype=np.uint8)

    def __init__(self, root_dir, train=True, joint_transforms=None,
                 image_transforms=None, mask_transforms=None, gray_image=False):
        """
        Args:
            root_dir (str): root directory of dataset
            joint_transforms (torchvision.transforms.Compose): tranformation on both data and target
            image_transforms (torchvision.transforms.Compose): tranformation only on data
            mask_transforms (torchvision.transforms.Compose): tranformation only on target
            gray_image (bool): whether to return gray image image or not.
                               If True, returns img, mask, gray.
        """
        mode = 'Training' if train else 'Testing'
        img_dir = os.path.join(root_dir, 'Figaro1k', 'Original', mode)
        mask_dir = os.path.join(root_dir, 'Figaro1k', 'GT', mode)

        self.img_path_list = [os.path.join(img_dir, img) for img in sorted(os.listdir(img_dir))]
        self.mask_path_list = [os.path.join(mask_dir, mask) for mask in sorted(os.listdir(mask_dir))]
        self.joint_transforms = joint_transforms
        self.image_transforms = image_transforms
        self.mask_transforms = mask_transforms
        self.gray_image = gray_image

    def __getitem__(self,idx):
        img_path = self.img_path_list[idx]
        img = Image.open(img_path)

        mask_path = self.mask_path_list[idx]
        mask = Image.open(mask_path)

        if self.joint_transforms is not None:
            img, mask = self.joint_transforms(img, mask)
    
        if self.image_transforms is not None:
            img = self.image_transforms(img)

        if self.mask_transforms is not None:
            mask = self.mask_transforms(mask)
            if mask.ndim == 3 and mask.shape[0] == 1:  # Check if mask has a channel dimension
                mask = mask.squeeze(0)  # Remove the channel dimension
        else:
            mask = transforms.ToTensor()(mask)
            if mask.ndim == 3 and mask.shape[0] == 1:  # Check if mask has a channel dimension
                mask = mask.squeeze(0)  # Remove the channel dimension

        if self.gray_image:
            gray = img.convert('L')
            gray = np.array(gray, dtype=np.float32)[np.newaxis, ] / 255
            return img, mask, gray
        else:
            return img, mask

    def __len__(self):
        return len(self.mask_path_list)

    def get_class_label(self, filename):
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

        if 0 < idx <= 150:
            return 0
        elif 150 < idx <= 300:
            return 1
        elif 300 < idx <= 450:
            return 2
        elif 450 < idx <= 600:
            return 3
        elif 600 < idx <= 750:
            return 4
        elif 750 < idx <= 900:
            return 5
        elif 900 < idx <= 1050:
            return 6
        raise ValueError


    @classmethod
    def decode_target(cls, mask):
        """decode semantic mask to RGB image"""
        # mask是一个二维数组，其中的每个值对应一个类别标签
        # cmap是一个N x 3的数组，其中N是类别的数目，每一行是对应类别标签的RGB颜色
        return cls.cmap[mask]


    
