import torch
import torch.nn as nn
import torch.nn.functional as F


class IoULoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    def forward(self, prob, target):
        # prob/target: [B,1,H,W] in [0,1]
        inter = (prob * target).sum(dim=(1, 2, 3))
        union = (prob + target - prob * target).sum(dim=(1, 2, 3))
        iou = (inter + self.smooth) / (union + self.smooth)
        return (1 - iou).mean()


class CombinedLoss(nn.Module):
    def __init__(
        self,
        bce_weight: float = 1.0,
        iou_weight: float = 1.0,
        aux_bce_w: float = 1.0,
        aux_iou_w: float = 1.0,
        aux_level_weights: dict | tuple | list | None = None
    ):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.iou = IoULoss()
        self.bce_w = float(bce_weight)
        self.iou_w = float(iou_weight)

        self.aux_bce_w = float(aux_bce_w)
        self.aux_iou_w = float(aux_iou_w)
        if aux_level_weights is None:
            self.aux_level_weights = {"p3": 0.4, "p2": 0.8, "pf": 1.0, "main": 1.2}
        elif isinstance(aux_level_weights, (tuple, list)):
            assert len(aux_level_weights) in (3, 4)
            if len(aux_level_weights) == 3:
                self.aux_level_weights = {"p3": aux_level_weights[0], "p2": aux_level_weights[1], "pf": aux_level_weights[2], "main": 1.2}
            else:
                self.aux_level_weights = {"p3": aux_level_weights[0], "p2": aux_level_weights[1], "pf": aux_level_weights[2], "main": aux_level_weights[3]}
        else:
            self.aux_level_weights = dict(aux_level_weights)

    def forward(
        self,
        *,
        seg_logits: torch.Tensor,
        target_mask: torch.Tensor,
        seg_prob: torch.Tensor | None = None,
        attn_dict: dict | None = None
    ):
        device = seg_logits.device
        target = target_mask.float().to(device)
        prob = torch.sigmoid(seg_logits) if seg_prob is None else seg_prob.float()
        if (prob.min() < -1e-3) or (prob.max() > 1.0 + 1e-3):
            prob = torch.sigmoid(prob)

        main_w = float(self.aux_level_weights.get("main", 1.2))
        bce_l = self.bce(seg_logits, target)
        iou_l = self.iou(prob, target)
        seg_loss_base = main_w * (self.bce_w * bce_l + self.iou_w * iou_l)

        aux_term = torch.tensor(0.0, device=device)
        if isinstance(attn_dict, dict):
            aux = attn_dict.get("aux_logits", None)
            if isinstance(aux, dict) and len(aux) > 0:
                aux_terms = []
                for k in ("p3", "p2", "pf"):
                    if k in aux and aux[k] is not None:
                        lg = aux[k].float().to(device)     # raw logits
                        tgt = target
                        if lg.shape[-2:] != tgt.shape[-2:]:
                            tgt = F.interpolate(tgt, size=lg.shape[-2:], mode="nearest")
                        bce_k = self.bce(lg, tgt)
                        iou_k = self.iou(torch.sigmoid(lg), tgt)
                        w_level = float(self.aux_level_weights.get(k, 1.0))
                        aux_terms.append(w_level * (self.aux_bce_w * bce_k + self.aux_iou_w * iou_k))
                if aux_terms:
                    aux_term = sum(aux_terms) / len(aux_terms)

        total_loss = seg_loss_base + aux_term

        return total_loss, seg_loss_base.detach(), aux_term.detach()
