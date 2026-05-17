import os
import json
import random
import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as F

from utils import batch_mask_to_bbox_fast, safe_bbox

class CODTrainTransform:
    
    def __init__(self, input_size=(416, 416),
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225),
                 do_color=True):
        self.input_size = input_size
        self.mean = mean
        self.std = std
        self.do_color = do_color

    def _random_resized_crop_params(self, width, height,
                                    scale=(0.7, 1.0), ratio=(0.9, 1.1)):
        area = width * height
        for _ in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (float(torch.log(torch.tensor(ratio[0]))),
                         float(torch.log(torch.tensor(ratio[1]))))
            aspect = float(torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])))
            w = int(round((target_area * aspect) ** 0.5))
            h = int(round((target_area / aspect) ** 0.5))
            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                return i, j, h, w
        in_ratio = width / height
        if in_ratio < ratio[0]:
            w = width
            h = int(round(w / ratio[0]))
        elif in_ratio > ratio[1]:
            h = height
            w = int(round(h * ratio[1]))
        else:
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def _maybe_motion_blur(self, pil_img: Image.Image):
        dx = random.choice([-1, 0, 1])
        dy = random.choice([-1, 0, 1])
        if dx == 0 and dy == 0:
            return pil_img
        w, h = pil_img.size
        shifted = pil_img.transform((w, h), Image.AFFINE, (1, 0, dx, 0, 1, dy),
                                    resample=Image.BILINEAR)
        return Image.blend(pil_img, shifted, alpha=0.3)

    def __call__(self, img: Image.Image, mask: Image.Image, edge: Image.Image | None):
        w, h = img.size
        i, j, th, tw = self._random_resized_crop_params(
            w, h, scale=(0.7, 1.0), ratio=(0.9, 1.1)
        )
        img  = F.resized_crop(img,  i, j, th, tw, size=self.input_size,
                              interpolation=T.InterpolationMode.BILINEAR)
        mask = F.resized_crop(mask, i, j, th, tw, size=self.input_size,
                              interpolation=T.InterpolationMode.NEAREST)
        if edge is not None:
            edge = F.resized_crop(edge, i, j, th, tw, size=self.input_size,
                                  interpolation=T.InterpolationMode.NEAREST)

        if random.random() < 0.5:
            img  = F.hflip(img)
            mask = F.hflip(mask)
            if edge is not None:
                edge = F.hflip(edge)

        if self.do_color and random.random() < 0.8:
            b = random.uniform(0.9, 1.1)
            c = random.uniform(0.9, 1.1)
            s = random.uniform(0.9, 1.1)
            img = F.adjust_brightness(img, b)
            img = F.adjust_contrast(img, c)
            img = F.adjust_saturation(img, s)

        if random.random() < 0.4:
            if random.random() < 0.5:
                radius = random.uniform(0.6, 1.4)
                img = img.filter(ImageFilter.GaussianBlur(radius=radius))
            else:
                img = self._maybe_motion_blur(img)

        img_tensor  = F.to_tensor(img)
        img_tensor  = F.normalize(img_tensor, mean=self.mean, std=self.std)
        mask_tensor = F.to_tensor(mask)
        if edge is None:
            edge_tensor = torch.zeros_like(mask_tensor)
        else:
            edge_tensor = F.to_tensor(edge)

        return img_tensor, mask_tensor, edge_tensor


class CODEvalTransform:
    def __init__(self, input_size=(416, 416),
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225)):
        self.input_size = input_size
        self.mean = mean
        self.std = std

    def __call__(self, img: Image.Image, mask: Image.Image, edge: Image.Image | None):
        img  = F.resize(img,  self.input_size, interpolation=T.InterpolationMode.BILINEAR)
        mask = F.resize(mask, self.input_size, interpolation=T.InterpolationMode.NEAREST)
        if edge is not None:
            edge = F.resize(edge, self.input_size, interpolation=T.InterpolationMode.NEAREST)

        img_tensor  = F.to_tensor(img)
        img_tensor  = F.normalize(img_tensor, mean=self.mean, std=self.std)
        mask_tensor = F.to_tensor(mask)
        if edge is None:
            edge_tensor = torch.zeros_like(mask_tensor)
        else:
            edge_tensor = F.to_tensor(edge)

        return img_tensor, mask_tensor, edge_tensor

def load_and_preprocess_image_from_pil(img_pil: Image.Image, target_size=(416, 416)):
    original_w, original_h = img_pil.size
    preprocess = T.Compose([
        T.Resize(target_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    img_tensor = preprocess(img_pil)
    return img_tensor, original_h, original_w

class CustomDataset(Dataset):
    def __init__(self, img_dir, mask_dir, gt_bboxes_path,edge_dir=None,
                 is_train=True, transform=None, input_size=(416, 416)):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.edge_dir = edge_dir
        self.is_train = is_train
        self.input_size = input_size

        exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
        self.img_names = [f for f in os.listdir(img_dir) if f.lower().endswith(exts)]
        self.img_names.sort()
        if len(self.img_names) == 0:
            raise RuntimeError(f"No images found in {img_dir}")

        self.gt_bboxes = {}

        if transform is not None:
            self.transform = transform
        else:
            self.transform = CODTrainTransform(input_size=input_size, do_color=True) if is_train \
                             else CODEvalTransform(input_size=input_size)

    def __len__(self):
        return len(self.img_names)

    def _read_img(self, path: str) -> Image.Image:
        return Image.open(path).convert("RGB")

    def _read_mask(self, path: str) -> Image.Image:
        return Image.open(path).convert("L")

    def _read_edge_maybe(self, stem: str) -> Image.Image | None:
        if self.edge_dir is None:
            return None
        p_png = os.path.join(self.edge_dir, stem + ".png")
        p_jpg = os.path.join(self.edge_dir, stem + ".jpg")
        p_jpeg = os.path.join(self.edge_dir, stem + ".jpeg")
        for p in [p_png, p_jpg, p_jpeg]:
            if os.path.exists(p):
                return Image.open(p).convert("L")
        return None

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        stem, _ = os.path.splitext(img_name)

        img_path  = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, stem + '.png')
        if not os.path.exists(mask_path):
            alt = os.path.join(self.mask_dir, stem + '.jpg')
            if os.path.exists(alt):
                mask_path = alt

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = self._read_img(img_path)

        if not os.path.exists(mask_path):
            print(f"[Warn] Mask not found for {img_name} at {mask_path}. Using zeros mask.")
            mask = Image.fromarray(np.zeros((self.input_size[1], self.input_size[0]), dtype=np.uint8))
        else:
            mask = self._read_mask(mask_path)

        edge = self._read_edge_maybe(stem)

        img_tensor, mask_tensor, edge_tensor = self.transform(img, mask, edge)

        mask_tensor = (mask_tensor > 0.5).float()
        edge_tensor = (edge_tensor > 0.5).float()

        try:
            gt_bbox_norm = batch_mask_to_bbox_fast(mask_tensor.unsqueeze(0))[0]
        except Exception:
            try:
                gt_bbox_norm = safe_bbox(mask_tensor)
            except Exception:
                gt_bbox_norm = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32)

        return {
            'img_tensor': img_tensor,     
            'mask_tensor': mask_tensor,   
            'edge_tensor': edge_tensor,   
            'img_name': img_name,
            'bbox': gt_bbox_norm
        }


def custom_collate_fn(batch):
    img_tensors  = torch.stack([item['img_tensor']  for item in batch])
    mask_tensors = torch.stack([item['mask_tensor'] for item in batch])
    edge_tensors = torch.stack([item['edge_tensor'] for item in batch])
    bboxes       = torch.stack([item['bbox']        for item in batch])
    img_names    = [item['img_name'] for item in batch]
    return {
        'img_tensor':  img_tensors,
        'mask_tensor': mask_tensors,
        'edge_tensor': edge_tensors,
        'bbox':        bboxes,
        'img_name':    img_names
    }
