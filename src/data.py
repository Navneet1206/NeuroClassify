import os
import glob
from typing import Tuple, List, Optional
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import SimpleITK as sitk
from .utils import CLASS_TO_IDX, IDX_TO_CLASS


class BrainMRIDataset(Dataset):
    def __init__(self, image_paths: List[str], labels: List[int], img_size: int = 224, augment: bool = False):
        self.image_paths = image_paths
        self.labels = labels
        self.img_size = img_size
        self.augment = augment
        self.train_tf = A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.1, rotate_limit=10, p=0.5),
            A.RandomBrightnessContrast(0.1, 0.1, p=0.3),
            A.HueSaturationValue(hue_shift_limit=2, sat_shift_limit=10, val_shift_limit=10, p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
        self.val_tf = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        label = self.labels[idx]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tf = self.train_tf if self.augment else self.val_tf
        img = tf(image=img)["image"]
        return img, torch.tensor(label, dtype=torch.long)


def collect_image_paths(root: str) -> Tuple[List[str], List[int]]:
    # expects directory structure: root/class_name/*.png|jpg|jpeg
    exts = ("*.png", "*.jpg", "*.jpeg")
    image_paths, labels = [], []
    for cls, idx in CLASS_TO_IDX.items():
        for ext in exts:
            files = glob.glob(os.path.join(root, cls, ext))
            for f in files:
                image_paths.append(f)
                labels.append(idx)
    return image_paths, labels


def skull_strip_simpleitk(input_path: str, output_path: str) -> None:
    """Basic skull-stripping using Otsu + morphological ops. Works on many 2D PNG/JPG brain MRIs.
    For 3D DICOM/NIfTI, consider using MONAI or FSL BET externally.
    """
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Cannot read: {input_path}")
    # Otsu threshold
    _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Keep largest component
    nb_components, output, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    sizes = stats[1:, -1]
    if len(sizes) > 0:
        max_label = 1 + np.argmax(sizes)
        skull_mask = np.zeros(output.shape, dtype=np.uint8)
        skull_mask[output == max_label] = 255
    else:
        skull_mask = mask
    # Morphology to clean
    skull_mask = cv2.morphologyEx(skull_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    skull_mask = cv2.medianBlur(skull_mask, 5)
    # Apply mask
    brain = cv2.bitwise_and(img, skull_mask)
    cv2.imwrite(output_path, brain)


def preprocess_folder(raw_dir: str, processed_dir: str, do_skull_strip: bool = True) -> None:
    os.makedirs(processed_dir, exist_ok=True)
    for cls in CLASS_TO_IDX.keys():
        os.makedirs(os.path.join(processed_dir, cls), exist_ok=True)
        src_dir = os.path.join(raw_dir, cls)
        dst_dir = os.path.join(processed_dir, cls)
        if not os.path.isdir(src_dir):
            continue
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            for src in glob.glob(os.path.join(src_dir, ext)):
                fname = os.path.basename(src)
                dst = os.path.join(dst_dir, fname)
                if do_skull_strip:
                    try:
                        skull_strip_simpleitk(src, dst)
                    except Exception:
                        # fallback: copy as-is
                        img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
                        cv2.imwrite(dst, img)
                else:
                    img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
                    cv2.imwrite(dst, img)
