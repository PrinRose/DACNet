import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt
from encoder.pvtv2_encoder import pvt_v2_b4
from VMamba.vmamba import SS2D


class SpatialMambaLayer(nn.Module):
    def __init__(self, dim: int, state_dim: int = 16, *, expand: float = 2.0,
                 forward_type: str = "v01", d_conv: int = 3, conv_bias: bool = True,
                 dropout: float = 0.0, dt_rank: str | int = "auto", initialize: str = "v0",
                 channel_first: bool = True):
        super().__init__()
        assert channel_first
        self.core = SS2D(
            d_model=dim, d_state=state_dim, ssm_ratio=expand, dt_rank=dt_rank,
            act_layer=nn.SiLU, d_conv=d_conv, conv_bias=conv_bias, dropout=dropout, bias=False,
            dt_min=0.001, dt_max=0.1, dt_init="random", dt_scale=1.0, dt_init_floor=1e-4,
            initialize=initialize, forward_type=forward_type, channel_first=True,
        )
    def forward(self, x): return self.core(x)


def load_pretrained_pvtv2_weights(model, path, prefix_strip="backbone."):
    sd = torch.load(path, map_location='cpu')
    if "model" in sd: sd = sd["model"]
    new_sd = {(k[len(prefix_strip):] if k.startswith(prefix_strip) else k): v for k, v in sd.items()}
    msd = model.state_dict()
    for k, v in new_sd.items():
        if k in msd and msd[k].shape == v.shape: msd[k].copy_(v)
    model.load_state_dict(msd, strict=False)


class ConvBNAct(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=None, dilation=1, g=1, act=True, bias=False):
        super().__init__()
        if p is None: p = (k // 2) * dilation if k > 1 else 0
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, dilation=dilation, groups=g, bias=bias)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()
    def forward(self, x): return self.act(self.bn(self.conv(x)))


class DSConv(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, dilation=1, act=True):
        super().__init__()
        pad = dilation if k == 3 else (k // 2) * dilation
        self.dw = ConvBNAct(in_c, in_c, k=k, s=s, p=pad, dilation=dilation, g=in_c, act=True)
        self.pw = ConvBNAct(in_c, out_c, k=1, s=1, p=0, dilation=1, g=1, act=act)
    def forward(self, x): return self.pw(self.dw(x))


class UpSampleShuffle(nn.Module):
    def __init__(self, in_c, out_c, s=2):
        super().__init__()
        self.pre = DSConv(in_c, out_c * (s ** 2), k=3, s=1)
        self.ps  = nn.PixelShuffle(s)
        self.bn  = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x): return self.act(self.bn(self.ps(self.pre(x))))


class DeformAlign(nn.Module):
    def __init__(self, in_c, out_c, k=3, groups=1, use_bias=False):
        super().__init__()
        self.k = k
        self.offset_conv = nn.Sequential(ConvBNAct(in_c * 2, in_c, 3, 1),
                                         nn.Conv2d(in_c, 2 * k * k, 1, bias=True))
        self.weight = nn.Parameter(torch.randn(out_c, in_c, k, k) * (2.0 / (in_c * k * k)) ** 0.5)
        self.bias   = nn.Parameter(torch.zeros(out_c)) if use_bias else None
        self.proj   = ConvBNAct(out_c, out_c, k=1, act=True)
        self.has_deform = True
        try:
            from torchvision.ops import deform_conv2d  # noqa
        except Exception:
            self.has_deform = False
            self.fallback_conv = DSConv(in_c, out_c, k=3)
    def forward(self, lo, hi):
        if self.has_deform:
            from torchvision.ops import deform_conv2d
            offset = self.offset_conv(torch.cat([lo, hi], dim=1))
            out = deform_conv2d(lo, offset, self.weight, self.bias, stride=1,
                                padding=self.k // 2, dilation=1, mask=None)
        else:
            out = self.fallback_conv(lo)
        out = out + hi
        return self.proj(out)


class ASPPLite(nn.Module):
    def __init__(self, c, rates=(1, 2), gpool=False, bottleneck=4):
        super().__init__()
        mid = max(c // bottleneck, 8)
        self.branches = nn.ModuleList([nn.Sequential(DSConv(c, c, k=3, dilation=r)) for r in rates])
        self.gpool = None
        if gpool:
            self.gpool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, c, 1, bias=False), nn.ReLU(inplace=True))
        fuse_in = c * (len(rates) + (1 if gpool else 0))
        self.fuse = nn.Sequential(nn.Conv2d(fuse_in, mid, 1, bias=False),
                                  nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
                                  nn.Conv2d(mid, c, 1, bias=False))
    def forward(self, x):
        feats = [b(x) for b in self.branches]
        if self.gpool is not None:
            g = self.gpool(x); g = F.interpolate(g, size=x.shape[-2:], mode='bilinear', align_corners=False)
            feats.append(g)
        return self.fuse(torch.cat(feats, dim=1))


class LargeKernelDW(nn.Module):
    def __init__(self, c, k=13):
        super().__init__()
        pad = k // 2
        self.dw = nn.Conv2d(c, c, k, padding=pad, groups=c, bias=False)
        self.pw = nn.Conv2d(c, c, 1, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.dw(x); x = self.pw(x)
        return self.act(self.bn(x))


class ICMBlock(nn.Module):
    def __init__(self, c, state_dim=32, use_mamba=True, lam=2e-2):
        super().__init__()
        self.use_mamba = use_mamba
        self.lam = lam
        self.aspp  = ASPPLite(c, rates=(1, 2), gpool=False, bottleneck=4)
        self.lk    = LargeKernelDW(c, k=13)
        self.mamba = SpatialMambaLayer(dim=c, state_dim=state_dim) if use_mamba else nn.Identity()
        self.res_gamma = nn.Parameter(torch.ones(1, c, 1, 1) * 0.1)
    def forward(self, x):
        f_local = self.lk(self.aspp(x))
        f_long  = self.mamba(f_local) if self.use_mamba else f_local
        y_core  = f_local + self.lam * f_long
        return x + self.res_gamma * y_core


class CamouflageDetectionModelSpatialMamba(nn.Module):
    def __init__(self, img_size=(416, 416), pretrained_encoder_path=None,
                 emb_dim=96, mamba_state_dim=32):
        super().__init__()
        self.img_size = img_size

        self.encoder = pvt_v2_b4()
        if pretrained_encoder_path:
            load_pretrained_pvtv2_weights(self.encoder, pretrained_encoder_path)

        C = 224
        self.lat4 = nn.Conv2d(512, C, 1, bias=False)
        self.lat3 = nn.Conv2d(320, C, 1, bias=False)
        self.lat2 = nn.Conv2d(128, C, 1, bias=False)
        self.lat1 = nn.Conv2d( 64, C, 1, bias=False)

        self.up43 = UpSampleShuffle(C, C, 2)
        self.up32 = UpSampleShuffle(C, C, 2)
        self.up21 = UpSampleShuffle(C, C, 2)

        self.da3 = DeformAlign(C, C, k=3)
        self.da2 = DeformAlign(C, C, k=3)
        self.da1 = DeformAlign(C, C, k=3)

        self.conv4 = DSConv(C, C, 3)
        self.conv3 = DSConv(C, C, 3)
        self.conv2 = DSConv(C, C, 3)
        self.conv1 = DSConv(C, C, 3)

        self.icm_p3  = ICMBlock(C, state_dim=mamba_state_dim, use_mamba=True, lam=2e-2)
        self.icm_p2  = ICMBlock(C, state_dim=mamba_state_dim, use_mamba=True, lam=2e-2)
        self.icm_fin = ICMBlock(C, state_dim=mamba_state_dim, use_mamba=True, lam=2e-2)

        fused_in = C * 3
        self.simple_head = nn.Sequential(
            DSConv(fused_in, C, k=3, dilation=1, act=True),
            nn.Conv2d(C, 1, kernel_size=1, bias=True)
        )

        self.aux_head_p3 = nn.Conv2d(C, 1, 1, bias=True)
        self.aux_head_p2 = nn.Conv2d(C, 1, 1, bias=True)
        self.aux_head_pf = nn.Conv2d(C, 1, 1, bias=True)

        for m in [self.lat1, self.lat2, self.lat3, self.lat4,
                  self.aux_head_p3, self.aux_head_p2, self.aux_head_pf]:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if getattr(m, "bias", None) is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        xin = F.interpolate(x, size=self.img_size, mode='bilinear', align_corners=False)

        c4, c3, c2, c1 = self.encoder(xin)

        p4 = self.conv4(self.lat4(c4))
        h3 = self.up43(p4); l3 = self.lat3(c3); p3 = self.conv3(self.da3(l3, h3))
        h2 = self.up32(p3); l2 = self.lat2(c2); p2 = self.conv2(self.da2(l2, h2))
        h1 = self.up21(p2); l1 = self.lat1(c1); pf = self.conv1(self.da1(l1, h1))

        p3 = ckpt.checkpoint(lambda z: self.icm_p3(z), p3)
        p2 = ckpt.checkpoint(lambda z: self.icm_p2(z), p2)
        pf = ckpt.checkpoint(lambda z: self.icm_fin(z), pf)

        H, W = x.shape[-2:]
        f3   = F.interpolate(p3, size=(H, W), mode='bilinear', align_corners=False)
        f2   = F.interpolate(p2, size=(H, W), mode='bilinear', align_corners=False)
        ffin = F.interpolate(pf, size=(H, W), mode='bilinear', align_corners=False)

        Fms_cat = torch.cat([f3, f2, ffin], dim=1)  # [B, 3*C, H, W]

        seg_logits = ckpt.checkpoint(lambda z: self.simple_head(z), Fms_cat)
        seg_prob   = torch.sigmoid(seg_logits)

        aux_logits = {
            'p3': self.aux_head_p3(p3),
            'p2': self.aux_head_p2(p2),
            'pf': self.aux_head_pf(pf),
        }

        tm_probs = torch.cat([seg_prob, torch.zeros_like(seg_prob), 1.0 - seg_prob], dim=1)
        attn = {
            'p_fg': seg_prob,
            'p_bg': 1.0 - seg_prob,
            'uncert': torch.zeros_like(seg_prob),
            'tm_probs': tm_probs,
            'tm_logits': None,
            'ms_weights': torch.ones((seg_prob.shape[0],3,1,1), device=seg_prob.device) / 3.0,
            'aux_logits': aux_logits
        }
        return seg_logits, seg_prob, attn
