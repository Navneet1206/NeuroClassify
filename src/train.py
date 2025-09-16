import os
import yaml
from typing import Dict
import numpy as np
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda import amp
from torchvision.ops import misc as misc_ops
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .data import BrainMRIDataset, collect_image_paths
from .model import ResNet50Classifier
from .utils import set_seed, compute_metrics


def get_optimizer(model, cfg):
    name = cfg["optimizer"]["name"].lower()
    lr = cfg["optimizer"]["lr"]
    wd = cfg["optimizer"]["weight_decay"]
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd, nesterov=True)
    else:
        raise ValueError(f"Unsupported optimizer: {name}")


def get_scheduler(optimizer, cfg):
    name = cfg["scheduler"]["name"].lower()
    if name == "cosineannealinglr":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["scheduler"]["T_max"], eta_min=cfg["scheduler"]["eta_min"]
        )
    elif name == "reducelronplateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3
        )
    else:
        return None


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, clip_norm=None):
    model.train()
    losses = []
    all_targets, all_preds, all_probs = [], [], []
    pbar = tqdm(loader, desc="Train", leave=False)
    for images, targets in pbar:
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        with amp.autocast(enabled=scaler is not None):
            logits = model(images)
            loss = criterion(logits, targets)
        if scaler is not None:
            scaler.scale(loss).backward()
            if clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
        losses.append(loss.item())
        probs = torch.softmax(logits.detach(), dim=1)
        preds = probs.argmax(dim=1)
        all_targets.append(targets.detach().cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        pbar.set_postfix(loss=np.mean(losses))
    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)
    metrics = compute_metrics(y_true, y_pred, y_prob, num_classes=logits.size(1))
    metrics["loss"] = float(np.mean(losses))
    return metrics


def evaluate(model, loader, criterion, device):
    model.eval()
    losses = []
    all_targets, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Val", leave=False):
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            loss = criterion(logits, targets)
            losses.append(loss.item())
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            all_targets.append(targets.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)
    metrics = compute_metrics(y_true, y_pred, y_prob, num_classes=logits.size(1))
    metrics["loss"] = float(np.mean(losses))
    return metrics


def run_training(config_path: str = "configs/config.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg["seed"])
    requested_device = cfg.get("device", "auto")
    if requested_device == "cpu":
        device = "cpu"
    else:
        # Use CUDA if available; otherwise fall back to CPU regardless of request
        device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(cfg["paths"]["runs"], exist_ok=True)
    os.makedirs(cfg["paths"]["checkpoints"], exist_ok=True)

    images, labels = [], []
    X, y = [], []
    img_paths, lbls = collect_image_paths(cfg["paths"]["processed"])
    X.extend(img_paths)
    y.extend(lbls)
    X = np.array(X)
    y = np.array(y)
    # Fail fast if no data found
    if X.size == 0:
        raise RuntimeError(
            "No images found under processed data directory.\n"
            f"Checked: {cfg['paths']['processed']} with expected subfolders ['glioma','meningioma','pituitary','normal'].\n"
            "Please place your dataset as data/raw/<class>/*.png|jpg and run preprocessing: \n"
            "  python -m src.preprocess --raw data/raw --out data/processed\n"
            "Then re-run training."
        )

    skf = StratifiedKFold(n_splits=cfg["cv"]["folds"], shuffle=cfg["cv"]["shuffle"], random_state=cfg["seed"])

    writer = SummaryWriter(log_dir=cfg["paths"]["logs"]) if cfg["logging"]["tensorboard"] else None

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"===== Fold {fold+1}/{cfg['cv']['folds']} =====")
        train_ds = BrainMRIDataset(X[train_idx].tolist(), y[train_idx].tolist(), img_size=cfg["img_size"], augment=True)
        val_ds = BrainMRIDataset(X[val_idx].tolist(), y[val_idx].tolist(), img_size=cfg["img_size"], augment=False)
        train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"], pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"], pin_memory=True)

        model = ResNet50Classifier(num_classes=cfg["num_classes"], freeze_until_layer=cfg["train"]["freeze_until_layer"]).to(device)
        criterion = nn.CrossEntropyLoss(label_smoothing=cfg["train"].get("label_smoothing", 0.0))
        optimizer = get_optimizer(model, cfg)
        scheduler = get_scheduler(optimizer, cfg)
        scaler = amp.GradScaler(enabled=cfg["train"]["amp"]) if device.startswith("cuda") else None

        best_metric = -np.inf
        patience = cfg["train"]["early_stopping_patience"]
        patience_counter = 0
        for epoch in range(cfg["epochs"]):
            print(f"Epoch {epoch+1}/{cfg['epochs']}")
            train_metrics = train_one_epoch(
                model, train_loader, criterion, optimizer, scaler, device, clip_norm=cfg["train"].get("gradient_clip_norm")
            )
            val_metrics = evaluate(model, val_loader, criterion, device)
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_metrics.get("macro_f1", val_metrics["accuracy"]))
                else:
                    scheduler.step()

            score = val_metrics.get("macro_f1", val_metrics["accuracy"])  # early stopping on macro-F1
            if writer is not None:
                writer.add_scalar(f"Fold{fold}/Train/Loss", train_metrics["loss"], epoch)
                writer.add_scalar(f"Fold{fold}/Val/Loss", val_metrics["loss"], epoch)
                writer.add_scalar(f"Fold{fold}/Val/Accuracy", val_metrics["accuracy"], epoch)
                writer.add_scalar(f"Fold{fold}/Val/MacroF1", val_metrics.get("macro_f1", 0.0), epoch)

            if score > best_metric:
                best_metric = score
                patience_counter = 0
                ckpt_path = os.path.join(cfg["paths"]["checkpoints"], f"resnet50_fold{fold}.pt")
                torch.save({
                    "model_state": model.state_dict(),
                    "cfg": cfg,
                    "fold": fold,
                    "val_metrics": val_metrics,
                }, ckpt_path)
                print(f"Saved best model to {ckpt_path}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered")
                    break
        print(f"Best val macro-F1 (fold {fold}): {best_metric:.4f}")
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    run_training()
