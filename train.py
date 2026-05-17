import os
import math
import json
import random
import shutil
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.amp import autocast as autocast_amp, GradScaler as AmpGradScaler
import torchvision.utils as vutils
import torch.nn.functional as F

from DACNet import CamouflageDetectionModelSpatialMamba as CamouflageDetectionModel
from data_loader import CustomDataset, custom_collate_fn
from loss import CombinedLoss
from utils import save_fullsize_visualization_with_bbox, ModelEMA, load_checkpoint, plot_loss_curves, save_checkpoint


def _unnorm_img(x):
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    return (x * std + mean).clamp(0, 1)


def _to_rgb(m):
    return m.repeat(1, 3, 1, 1)


@torch.no_grad()
def save_aux_visuals(imgs, gt, main_prob, aux_p3, aux_p2, aux_pf, save_path):
    batch_size = imgs.size(0)
    n = min(4, batch_size)

    vis_imgs = _unnorm_img(imgs[:n].float())
    gt_vis = _to_rgb(gt[:n].float())
    main_vis = _to_rgb(main_prob[:n].float())
    p3_vis = _to_rgb(aux_p3[:n].float())
    p2_vis = _to_rgb(aux_p2[:n].float())
    pf_vis = _to_rgb(aux_pf[:n].float())

    rows = []
    for i in range(n):
        row = torch.cat(
            [
                vis_imgs[i:i + 1],
                gt_vis[i:i + 1],
                main_vis[i:i + 1],
                p3_vis[i:i + 1],
                p2_vis[i:i + 1],
                pf_vis[i:i + 1],
            ],
            dim=3,
        )
        rows.append(row)

    grid = torch.cat(rows, dim=2)
    vutils.save_image(grid, save_path)


def train():
    config = {
        "DATASET_ROOT": "./data/TrainDataset",
        "CHECKPOINTS_DIR": "./checkpoints",
        "MODEL_INPUT_SIZE": (416, 416),

        "BATCH_SIZE": 6,
        "NUM_EPOCHS": 100,
        "WARMUP_EPOCHS": 0,

        "BASE_LR": 1e-5,
        "MAX_LR": 1e-4,
        "MIN_LR": 1e-6,
        "WEIGHT_DECAY": 1e-5,

        "PVT_PRETRAINED_PATH": "./pretrained/pvt_v2_b4.pth",
        "GRADIENT_ACCUMULATION_STEPS": 8,
        "USE_AMP": False,
        "USE_BF16": False,
        "EMA_DECAY": 0.999,

        "SAVE_DIR": "./checkpoints",
        "VISUALIZE_FREQ": 10,

        "BCE_WEIGHT": 1.0,
        "IOU_WEIGHT": 1.0,
        "AUX_LEVEL_WEIGHTS": [0.4, 0.8, 1.0, 1.2],

        "USE_COMPILE": False,
        "DROP_LAST": False,
        "REFINE_LAST_EPOCHS": 0,
    }

    os.makedirs(config["CHECKPOINTS_DIR"], exist_ok=True)
    visual_dir = os.path.join(config["CHECKPOINTS_DIR"], "visualizations")
    os.makedirs(visual_dir, exist_ok=True)
    plot_save_dir = os.path.join(config["SAVE_DIR"], "plots")
    os.makedirs(plot_save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    model = CamouflageDetectionModel(
        img_size=config["MODEL_INPUT_SIZE"],
        pretrained_encoder_path=config["PVT_PRETRAINED_PATH"],
    ).to(device)

    if not hasattr(model, "img_size"):
        model.img_size = tuple(config["MODEL_INPUT_SIZE"])

    for _, p in model.named_parameters():
        p.requires_grad = True

    print("All model parameters are trainable.")

    if config.get("USE_COMPILE", False):
        try:
            model = torch.compile(model)
            print("torch.compile is enabled.")
        except Exception as e:
            print(f"torch.compile failed: {e}")

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["BASE_LR"],
        weight_decay=config["WEIGHT_DECAY"],
    )

    base_lr = config["BASE_LR"]
    max_lr = config["MAX_LR"]
    min_lr = config["MIN_LR"]
    warm_epochs = config["WARMUP_EPOCHS"]
    total_epochs = config["NUM_EPOCHS"]

    def lr_lambda(epoch):
        if epoch < warm_epochs and warm_epochs > 0:
            cur_lr = base_lr + (max_lr - base_lr) * (epoch + 1) / max(1, warm_epochs)
            return cur_lr / base_lr

        if total_epochs == warm_epochs:
            progress = 0.0
        else:
            progress = (epoch - warm_epochs) / max(1, total_epochs - warm_epochs)

        cur_lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))
        return cur_lr / base_lr

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    criterion = CombinedLoss(
        bce_weight=config["BCE_WEIGHT"],
        iou_weight=config["IOU_WEIGHT"],
        aux_bce_w=1.0,
        aux_iou_w=1.0,
        aux_level_weights=config["AUX_LEVEL_WEIGHTS"],
        edge_weight=0.0,
    ).to(device)

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported() and config["USE_BF16"]
    scaler = AmpGradScaler("cuda", enabled=config["USE_AMP"] and not use_bf16)

    train_dataset = CustomDataset(
        img_dir=os.path.join(config["DATASET_ROOT"], "Image"),
        mask_dir=os.path.join(config["DATASET_ROOT"], "GT_Object"),
        edge_dir=None,
        gt_bboxes_path=None,
        is_train=True,
        input_size=config["MODEL_INPUT_SIZE"],
    )

    num_workers = min(8, os.cpu_count() or 8)
    dl_kwargs = dict(
        batch_size=config["BATCH_SIZE"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn,
        persistent_workers=(num_workers > 0),
        drop_last=config.get("DROP_LAST", True),
    )

    if num_workers > 0:
        dl_kwargs["prefetch_factor"] = 4

    try:
        dl_kwargs["pin_memory_device"] = "cuda" if torch.cuda.is_available() else ""
    except Exception:
        pass

    train_loader = DataLoader(train_dataset, **dl_kwargs)
    model_ema = ModelEMA(model, decay=config["EMA_DECAY"])

    checkpoint_path = os.path.join(config["CHECKPOINTS_DIR"], "latest_checkpoint.pth")
    best_path = os.path.join(config["CHECKPOINTS_DIR"], "best_checkpoint.pth")

    start_epoch = 0
    best_metric = float("inf")
    allowed_keys = ["total_loss", "seg_loss", "aux_loss"]

    if os.path.exists(checkpoint_path):
        model, start_epoch, loss_history, ema_sd = load_checkpoint(
            model,
            optimizer,
            scheduler,
            checkpoint_path,
            device,
        )

        if isinstance(loss_history, dict):
            for k in list(loss_history.keys()):
                if k not in allowed_keys:
                    del loss_history[k]
            for k in allowed_keys:
                loss_history.setdefault(k, [])
        else:
            loss_history = {k: [] for k in allowed_keys}

        if ema_sd is not None:
            model_ema.ema.load_state_dict(ema_sd, strict=False)

        print(f"Resumed training from epoch {start_epoch}.")

        try:
            loss_history_path = os.path.join(config["CHECKPOINTS_DIR"], "loss_history.json")
            if os.path.exists(loss_history_path):
                with open(loss_history_path, "r") as f:
                    hist_tmp = json.load(f)
                if "total_loss" in hist_tmp and len(hist_tmp["total_loss"]) > 0:
                    best_metric = min(hist_tmp["total_loss"])
        except Exception:
            pass
    else:
        loss_history = {k: [] for k in allowed_keys}

    grad_accum_steps = max(1, config["GRADIENT_ACCUMULATION_STEPS"])
    refine_start_epoch = max(0, config["NUM_EPOCHS"] - config["REFINE_LAST_EPOCHS"])

    for epoch in range(start_epoch, config["NUM_EPOCHS"]):
        model.train()

        epoch_losses = {k: 0.0 for k in allowed_keys}
        processed_batches = 0
        step_in_epoch = 0

        use_amp_this_epoch = config["USE_AMP"] and (epoch < refine_start_epoch)

        if use_amp_this_epoch and use_bf16:
            amp_dtype = torch.bfloat16
        elif use_amp_this_epoch:
            amp_dtype = torch.float16
        else:
            amp_dtype = torch.float32

        amp_active = use_amp_this_epoch and scaler.is_enabled()

        desc = f"Epoch {epoch + 1}/{config['NUM_EPOCHS']} AMP:{use_amp_this_epoch}"
        progress = tqdm(
            train_loader,
            desc=desc,
            dynamic_ncols=True,
            smoothing=0.1,
            mininterval=0.3,
        )

        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(progress):
            img_tensor = batch["img_tensor"].to(device, non_blocking=True)
            mask_tensor = batch["mask_tensor"].to(device, non_blocking=True)

            if (
                torch.isnan(img_tensor).any()
                or torch.isinf(img_tensor).any()
                or torch.isnan(mask_tensor).any()
                or torch.isinf(mask_tensor).any()
            ):
                print(f"Skipped invalid batch: epoch={epoch + 1}, batch={batch_idx}")
                continue

            try:
                with autocast_amp(
                    device_type="cuda",
                    dtype=amp_dtype,
                    enabled=use_amp_this_epoch,
                ):
                    out = model(img_tensor)

                    if isinstance(out, tuple):
                        if len(out) == 2:
                            seg_logits, seg_prob = out
                            attn_dict = None
                        elif len(out) == 3 and isinstance(out[2], dict):
                            seg_logits, seg_prob, attn_dict = out
                        else:
                            raise RuntimeError(f"Unexpected model output tuple length: {len(out)}")
                    else:
                        seg_logits = out
                        seg_prob = torch.sigmoid(seg_logits)
                        attn_dict = None

                    loss_out = criterion(
                        seg_logits=seg_logits.float(),
                        seg_prob=seg_prob.float(),
                        target_mask=mask_tensor.float(),
                        attn_dict=attn_dict,
                        edge_gt=None,
                    )

                    if isinstance(loss_out, tuple):
                        if len(loss_out) == 4:
                            total_loss, seg_loss, _, aux_loss = loss_out
                        elif len(loss_out) == 3:
                            total_loss, seg_loss, aux_loss = loss_out
                        elif len(loss_out) == 2:
                            total_loss, seg_loss = loss_out
                            aux_loss = torch.zeros((), device=device)
                        else:
                            total_loss = loss_out[0]
                            seg_loss = total_loss
                            aux_loss = torch.zeros((), device=device)
                    else:
                        total_loss = loss_out
                        seg_loss = total_loss
                        aux_loss = torch.zeros((), device=device)

                    if not torch.is_tensor(aux_loss):
                        aux_loss = torch.tensor(float(aux_loss), device=device)

                    if not torch.isfinite(total_loss):
                        print(f"Skipped non-finite loss at epoch={epoch + 1}, batch={batch_idx}")
                        continue

                    loss_for_backward = total_loss / grad_accum_steps

            except Exception as e:
                print(f"Forward or loss computation failed: {e}")
                continue

            processed_batches += 1
            step_in_epoch += 1

            if amp_active:
                scaler.scale(loss_for_backward).backward()
            else:
                loss_for_backward.backward()

            if step_in_epoch % grad_accum_steps == 0:
                if amp_active:
                    scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                if amp_active:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)
                model_ema.update(model)

            epoch_losses["total_loss"] += float(total_loss.item())
            epoch_losses["seg_loss"] += float(seg_loss.item())
            epoch_losses["aux_loss"] += float(aux_loss.item())

            if batch_idx % 10 == 0:
                progress.set_postfix(
                    {
                        "Total": f"{total_loss.item():.4f}",
                        "Seg": f"{seg_loss.item():.4f}",
                        "Aux": f"{aux_loss.item():.4f}",
                        "LR": f"{optimizer.param_groups[0]['lr']:.2e}",
                        "AMP": use_amp_this_epoch,
                    }
                )

            if (epoch + 1) % config["VISUALIZE_FREQ"] == 0 and batch_idx == 0:
                img_names = batch.get("img_name", [])[:4]

                save_fullsize_visualization_with_bbox(
                    imgs_orig=img_tensor.detach(),
                    masks_gt_full=mask_tensor.detach(),
                    masks_pred_full=seg_prob.detach(),
                    pred_bboxes_norm=None,
                    save_dir=visual_dir,
                    img_names=img_names,
                    prefix=f"ep{epoch}_",
                )

                if isinstance(attn_dict, dict):
                    h, w = img_tensor.shape[-2:]
                    aux_p3 = torch.zeros_like(seg_prob)
                    aux_p2 = torch.zeros_like(seg_prob)
                    aux_pf = torch.zeros_like(seg_prob)

                    if "aux_logits" in attn_dict:
                        aux = attn_dict["aux_logits"]

                        if "p3" in aux:
                            aux_p3 = torch.sigmoid(
                                F.interpolate(
                                    aux["p3"],
                                    size=(h, w),
                                    mode="bilinear",
                                    align_corners=False,
                                )
                            )

                        if "p2" in aux:
                            aux_p2 = torch.sigmoid(
                                F.interpolate(
                                    aux["p2"],
                                    size=(h, w),
                                    mode="bilinear",
                                    align_corners=False,
                                )
                            )

                        if "pf" in aux:
                            aux_pf = torch.sigmoid(
                                F.interpolate(
                                    aux["pf"],
                                    size=(h, w),
                                    mode="bilinear",
                                    align_corners=False,
                                )
                            )

                    aux_path = os.path.join(visual_dir, f"ep{epoch}_aux_demo.png")
                    save_aux_visuals(
                        img_tensor.detach().float(),
                        mask_tensor.detach().float(),
                        seg_prob.detach().float(),
                        aux_p3.detach().float(),
                        aux_p2.detach().float(),
                        aux_pf.detach().float(),
                        aux_path,
                    )

        if step_in_epoch % grad_accum_steps != 0 and step_in_epoch > 0:
            if amp_active:
                scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            if amp_active:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            model_ema.update(model)

        scheduler.step()

        num_batches = max(1, processed_batches)
        for k in allowed_keys:
            epoch_val = epoch_losses.get(k, 0.0)
            loss_history[k].append(epoch_val / num_batches)

        plot_save = os.path.join(plot_save_dir, "loss_curves_epoch.png")
        plot_loss_curves(loss_history, save_path=plot_save)

        save_checkpoint(
            epoch,
            model,
            model_ema.ema,
            optimizer,
            scheduler,
            loss_history,
            config["CHECKPOINTS_DIR"],
        )

        cur_metric = loss_history["total_loss"][-1]
        if cur_metric <= best_metric:
            try:
                latest_path = os.path.join(config["CHECKPOINTS_DIR"], "latest_checkpoint.pth")
                if os.path.exists(latest_path):
                    shutil.copy2(latest_path, best_path)
                    best_metric = cur_metric
                    print(f"Updated best checkpoint: total_loss={best_metric:.6f}")
            except Exception as e:
                print(f"Failed to update best checkpoint: {e}")

        try:
            with open(os.path.join(config["CHECKPOINTS_DIR"], "loss_history.json"), "w") as f:
                json.dump(loss_history, f, indent=4)
        except Exception as e:
            print(f"Failed to write loss history: {e}")

        print(
            f"\nEpoch {epoch + 1} summary | "
            f"Total: {loss_history['total_loss'][-1]:.4f} | "
            f"Seg: {loss_history['seg_loss'][-1]:.4f} | "
            f"Aux: {loss_history['aux_loss'][-1]:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
            f"AMP: {use_amp_this_epoch}\n"
        )


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    random.seed(42)
    torch.manual_seed(42)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    train()