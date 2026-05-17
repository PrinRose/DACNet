import numpy as np
import os
from PIL import Image
import numpy as np
import torch
import torchvision.utils as vutils
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
import random
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn as nn

from DACNet import CamouflageDetectionModelSpatialMamba as CamouflageDetectionModel

def plot_loss_curves(loss_history, save_path="loss_curves.png"):
    plt.figure(figsize=(12, 6))
    for loss_name, losses in loss_history.items():
        if isinstance(losses, list) and len(losses) > 0:
            plt.plot(losses, label=loss_name)
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Training Loss Curves')
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.savefig(save_path); plt.close()

def save_checkpoint(epoch, model, model_ema, optimizer, scheduler, loss_history, checkpoint_dir):
    ckpt = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'model_ema_state_dict': (model_ema.state_dict() if model_ema is not None else None),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss_history': loss_history,
    }
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(ckpt, os.path.join(checkpoint_dir, 'latest_checkpoint.pth'))
    if (epoch + 1) % 10 == 0:
        torch.save(ckpt, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth'))

def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    missing, unexpected = model.load_state_dict(ckpt['model_state_dict'], strict=False)
    arch_changed = len(missing) > 0
    if arch_changed:
        start_epoch = 0
        model_ema_sd = None
    else:
        start_epoch = ckpt.get('epoch', 0)
        if optimizer is not None and ckpt.get('optimizer_state_dict') is not None:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if scheduler is not None and ckpt.get('scheduler_state_dict') is not None:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        model_ema_sd = ckpt.get('model_ema_state_dict', None)

    if missing:
        print(f"Missing keys: {len(missing)}（示例）: {missing[:8]}")
    if unexpected:
        print(f"Unexpected keys: {len(unexpected)}（示例）: {unexpected[:8]}")

    loss_history = ckpt.get('loss_history', {
        'total_loss': [], 'seg_loss': [], 'uncertainty_loss': [], 'l1_loss': [], 'aux_loss': [], 'cl_loss': []
    })
    for k in ['aux_loss', 'cl_loss']:
        if k not in loss_history: loss_history[k] = []

    return model, start_epoch, loss_history, model_ema_sd

class ModelEMA(torch.nn.Module):
    def __init__(self, model, decay=0.999):
        super().__init__()
        self.ema = CamouflageDetectionModel(
            img_size=model.img_size,
            pretrained_encoder_path=None
        ).to(next(model.parameters()).device)
        self.ema.load_state_dict(model.state_dict(), strict=True)
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.ema.eval()

    def update(self, model):
        with torch.no_grad():
            msd = model.state_dict()
            for k, v in self.ema.state_dict().items():
                if k in msd:
                    self.ema.state_dict()[k].copy_(v * self.decay + msd[k] * (1.0 - self.decay))

def dice_loss_from_logits(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    targets = targets.float()
    dims = (1,2,3)
    inter = torch.sum(probs * targets, dims)
    union = torch.sum(probs, dims) + torch.sum(targets, dims)
    dice = (2*inter + eps) / (union + eps)
    return 1 - dice.mean()

class AuxCriterion(nn.Module):
    def __init__(self, bce_w=1.0, dice_w=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_w = bce_w
        self.dice_w = dice_w
    def forward(self, logits, targets):
        return self.bce_w * self.bce(logits, targets.float()) + self.dice_w * dice_loss_from_logits(logits, targets)

def draw_bbox_on_image(img_tensor, 
                       img_name,
                       gt_bbox=None, 
                       pred_bbox=None, 
                       paste_bbox=None, 
                       save_dir="debug_vis", 
                       mean=None, 
                       std=None):
    os.makedirs(save_dir, exist_ok=True)

    if img_tensor.is_cuda:
        img_tensor = img_tensor.cpu()

    if mean is not None and std is not None:
        mean = mean.to(img_tensor.device)
        std = std.to(img_tensor.device)
        img_tensor = img_tensor * std + mean

    img_np = img_tensor.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    img_np = np.ascontiguousarray((img_np * 255).astype(np.uint8))

    H, W = img_np.shape[:2]

    def draw_box(bbox, color, label):
        if bbox is not None:
            x1, y1, x2, y2 = bbox.clamp(0, 1).tolist()
            x1, y1 = int(x1 * W), int(y1 * H)
            x2, y2 = int(x2 * W), int(y2 * H)
            cv2.rectangle(img_np, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img_np, label, (x1, max(y1-5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    draw_box(gt_bbox, (0, 255, 0), "GT")
    draw_box(pred_bbox, (0, 0, 255), "Pred")
    draw_box(paste_bbox, (255, 0, 0), "Paste")

    save_path = os.path.join(save_dir, f"{img_name}.jpg")
    cv2.imwrite(save_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

def plot_loss_curves(loss_history, save_path=None, title="Training Loss Curves"):
 
    plt.figure(figsize=(10, 6))

    if isinstance(loss_history, dict) and 'total_loss' in loss_history:
        epochs = range(1, len(loss_history['total_loss']) + 1)
        plt.plot(epochs, loss_history['total_loss'], label='Total Loss', color='red', linestyle='-')
        plt.plot(epochs, loss_history['segmentation_loss'], label='Segmentation Loss', color='blue', linestyle='--')
        plt.plot(epochs, loss_history['bbox_loss'], label='BBox Loss', color='green', linestyle=':')
        plt.plot(epochs, loss_history['uncertainty_loss'], label='Uncertainty Loss', color='purple', linestyle='-.')
    elif isinstance(loss_history, dict) and 'epoch_loss' in loss_history:
        epochs = range(1, len(loss_history['epoch_loss']) + 1)
        plt.plot(epochs, loss_history['epoch_loss'], label='Epoch Loss', color='red', linestyle='-')
    else:
        return

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    else:
        plt.show()

def is_valid_bbox(bbox):
    return (bbox[2] > bbox[0]) and (bbox[3] > bbox[1])

def denormalize(img_tensor, mean, std):
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    return (img_tensor * std + mean).clamp(0, 1)


def safe_bbox(bbox):
    # bbox: (N,4) [x1,y1,x2,y2]
    x1y1 = bbox[:, :2]
    x2y2 = bbox[:, 2:]
    bbox_min = torch.minimum(x1y1, x2y2)
    bbox_max = torch.maximum(x1y1, x2y2)
    return torch.cat([bbox_min, bbox_max], dim=1).clamp(0, 1)


def dice_score(pred_prob, target, eps=1e-7):
    """
    pred_prob: [B,1,H,W] in [0,1]
    target:    [B,1,H,W] in {0,1}
    """
    pred = (pred_prob > 0.5).float()
    inter = (pred * target).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    dice = (2 * inter + eps) / (union + eps)
    return dice.mean().item()

def iou_score(pred_prob, target, eps=1e-7):
    pred = (pred_prob > 0.5).float()
    inter = (pred * target).sum(dim=(1,2,3))
    union = (pred + target).clamp(0,1).sum(dim=(1,2,3))
    iou = (inter + eps) / (union + eps)
    return iou.mean().item()

def plot_loss_curves(loss_history, save_path="loss_curves.png"):

    epochs = list(range(1, len(loss_history['total_loss']) + 1))
    plt.figure(figsize=(12, 6))

    if loss_history.get('total_loss'):
        plt.plot(epochs, loss_history['total_loss'], label='train_total_loss', linewidth=2)
        marks_x = [e for e in epochs if e % 5 == 0]
        marks_y = [loss_history['total_loss'][e-1] for e in marks_x]
        if marks_x:
            plt.scatter(marks_x, marks_y, s=40, zorder=5, label='train_total @5-epoch')

    if loss_history.get('val_total_loss'):
        plt.plot(epochs, loss_history['val_total_loss'], label='val_total_loss', linewidth=2)

    aux_keys = ['global_pixel_loss','global_bbox_loss','uncertainty_loss','final_mask_loss']
    for k in aux_keys:
        if k in loss_history and len(loss_history[k]) == len(epochs):
            plt.plot(epochs, loss_history[k], '--', alpha=0.6, label=f'train_{k}')

    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Training & Validation Loss Curves')
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.savefig(save_path); plt.close()

def save_checkpoint(epoch, model, model_ema, optimizer, scheduler, loss_history, checkpoint_dir, tag='latest'):
    ckpt = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'model_ema_state_dict': (model_ema.state_dict() if model_ema is not None else None),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss_history': loss_history,
    }
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(ckpt, os.path.join(checkpoint_dir, f'{tag}_checkpoint.pth'))
    if (epoch + 1) % 10 == 0 and tag == 'latest':
        torch.save(ckpt, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth'))

def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer is not None and ckpt.get('optimizer_state_dict') is not None:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scheduler is not None and ckpt.get('scheduler_state_dict') is not None:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    start_epoch = ckpt['epoch']
    loss_history = ckpt.get('loss_history', {
        'total_loss': [], 'global_pixel_loss': [], 'global_bbox_loss': [],
        'uncertainty_loss': [], 'final_mask_loss': [],
        'val_total_loss': [], 'val_dice': [], 'val_iou': []
    })
    model_ema_sd = ckpt.get('model_ema_state_dict', None)
    return model, start_epoch, loss_history, model_ema_sd

def light_copy_paste(img, mask, prob=0.15):
    if random.random() > prob or img.size(0) < 2:
        return img, mask
    B, _, H, W = img.shape
    idx_src = random.randrange(B)
    idx_dst = random.randrange(B)
    if idx_src == idx_dst:
        return img, mask
    with torch.no_grad():
        m = (mask[idx_src] > 0.5).float()
        if m.sum() < 50:
            return img, mask
        k = 3
        m_d = F.max_pool2d(m, kernel_size=2*k+1, stride=1, padding=k)
        img[idx_dst] = img[idx_dst] * (1 - m_d) + img[idx_src] * m_d
        mask[idx_dst] = torch.clamp(mask[idx_dst] + m, 0, 1)
    return img, mask