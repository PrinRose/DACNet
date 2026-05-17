import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

from DACNet import CamouflageDetectionModelSpatialMamba as CamouflageDetectionModel

ROOT = "../.."
MODEL_PATH = os.path.join(ROOT, "checkpoint.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_INPUT_SIZE = (416, 416)
SAVE_DIR = ROOT

MASK_SAVE_DIR = os.path.join(SAVE_DIR, "pred_masks")
os.makedirs(MASK_SAVE_DIR, exist_ok=True)

preprocess = transforms.Compose([
    transforms.Resize(MODEL_INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

model = CamouflageDetectionModel(img_size=MODEL_INPUT_SIZE).to(DEVICE)

ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
state_dict = ckpt.get("model_state_dict", ckpt)
state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}

missing, unexpected = model.load_state_dict(state_dict, strict=False)
if missing:
    print(f"Missing keys: {len(missing)} (first 10): {missing[:10]}")
if unexpected:
    print(f"Unexpected keys: {len(unexpected)} (first 10): {unexpected[:10]}")

model.eval()

test_root = os.path.join(ROOT, "COD-TestDataset")
datasets = sorted([
    d for d in os.listdir(test_root)
    if os.path.isdir(os.path.join(test_root, d))
])

with torch.no_grad():
    for dataset in datasets:
        print(f"Processing {dataset}")

        img_dir = os.path.join(test_root, dataset, "Imgs")
        mask_save_dir = os.path.join(MASK_SAVE_DIR, dataset)
        os.makedirs(mask_save_dir, exist_ok=True)

        img_list = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ])

        for img_name in tqdm(img_list, total=len(img_list), desc=f"Inferencing {dataset}"):
            img_path = os.path.join(img_dir, img_name)
            image_pil = Image.open(img_path).convert("RGB")

            input_tensor = preprocess(image_pil).unsqueeze(0).to(DEVICE, dtype=torch.float32)

            out = model(input_tensor)

            if isinstance(out, tuple):
                if len(out) >= 2:
                    seg_logits, seg_prob = out[0], out[1]
                else:
                    raise RuntimeError(f"Unexpected model output tuple length: {len(out)}")
            else:
                seg_logits = out
                seg_prob = torch.sigmoid(seg_logits)

            final_pred = seg_prob.detach().float().clamp(0.0, 1.0)
            final_pred = final_pred.squeeze(0).squeeze(0).cpu().numpy()

            h, w = image_pil.height, image_pil.width
            pred_mask_resized = cv2.resize(
                (final_pred * 255.0).astype(np.uint8),
                (w, h),
                interpolation=cv2.INTER_LINEAR
            )

            save_name = os.path.splitext(img_name)[0] + ".png"
            Image.fromarray(pred_mask_resized).save(os.path.join(mask_save_dir, save_name))