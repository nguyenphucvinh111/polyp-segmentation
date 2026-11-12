"""
Dataloader for Polyp Segmentation Training & Testing.

Matches the official PraNet specifications:
- Training resolution: 352x352 (multi-scale handled in Trainer)
- Image normalization: ImageNet mean [0.485, 0.456, 0.406], std [0.229, 0.224, 0.225]
- Mask: binary floating point [0, 1]
- Augmentations: RandomFlip, RandomRotation, ColorJitter
"""

import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from utils.augmentation import PolypAugmentation


class PolypDataset(Dataset):
    """Training dataset for polyp segmentation."""

    def __init__(self, image_root: str, gt_root: str, trainsize: int = 352, is_train: bool = True):
        self.trainsize = trainsize
        self.is_train = is_train

        valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
        img_files = sorted([f for f in os.listdir(image_root) if f.lower().endswith(valid_exts)])
        gt_files = sorted([f for f in os.listdir(gt_root) if f.lower().endswith(valid_exts)])

        # Pair images and masks by basename stem
        self.images = []
        self.gts = []
        gt_map = {os.path.splitext(f)[0]: f for f in gt_files}
        for img_f in img_files:
            stem = os.path.splitext(img_f)[0]
            if stem in gt_map:
                self.images.append(os.path.join(image_root, img_f))
                self.gts.append(os.path.join(gt_root, gt_map[stem]))

        self.size = len(self.images)
        assert self.size > 0, f"No paired images found in {image_root} and {gt_root}"

        self.aug = PolypAugmentation() if is_train else None

        # Transforms
        color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1) if is_train else None
        
        img_tf_list = []
        if color_jitter:
            img_tf_list.append(color_jitter)
        img_tf_list.extend([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.img_transform = transforms.Compose(img_tf_list)

        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int):
        with open(self.images[index], 'rb') as f:
            image = Image.open(f).convert('RGB')
        with open(self.gts[index], 'rb') as f:
            gt = Image.open(f).convert('L')

        if self.is_train and self.aug:
            image, gt = self.aug(image, gt)

        image = self.img_transform(image)
        gt = (self.gt_transform(gt) > .5).float()

        return image, gt


def get_loader(
    image_root: str,
    gt_root: str,
    batchsize: int = 16,
    trainsize: int = 352,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    is_train: bool = True
) -> DataLoader:
    """Create DataLoader for training or validation."""
    dataset = PolypDataset(image_root, gt_root, trainsize=trainsize, is_train=is_train)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batchsize,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=is_train
    )
    return data_loader


class TestDataset:
    """
    Test dataset loader conforming to PraNet's evaluation pipeline:
    returns (normalized_tensor, original_gt_numpy_array, filename).
    """

    def __init__(self, image_root: str, gt_root: str, testsize: int = 352):
        self.testsize = testsize
        valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
        img_files = sorted([f for f in os.listdir(image_root) if f.lower().endswith(valid_exts)])
        gt_files = sorted([f for f in os.listdir(gt_root) if f.lower().endswith(valid_exts)])

        gt_map = {os.path.splitext(f)[0]: f for f in gt_files}
        self.images = []
        self.gts = []
        for img_f in img_files:
            stem = os.path.splitext(img_f)[0]
            if stem in gt_map:
                self.images.append(os.path.join(image_root, img_f))
                self.gts.append(os.path.join(gt_root, gt_map[stem]))

        self.size = len(self.images)
        self.index = 0

        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def load_data(self):
        img_path = self.images[self.index]
        gt_path = self.gts[self.index]
        with open(img_path, 'rb') as f:
            image = Image.open(f).convert('RGB')
        with open(gt_path, 'rb') as f:
            gt = Image.open(f).convert('L')

        image_t = self.transform(image).unsqueeze(0)
        name = os.path.basename(img_path)
        if not name.lower().endswith('.png'):
            name = os.path.splitext(name)[0] + '.png'

        self.index += 1
        return image_t, gt, name

    def reset(self):
        self.index = 0
