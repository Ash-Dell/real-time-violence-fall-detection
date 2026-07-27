import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from pytorchvideo.transforms import (
    ApplyTransformToKey,
    UniformTemporalSubsample,
    ShortSideScale,   
)
from pytorchvideo.data import LabeledVideoDataset, make_clip_sampler
from torchvision.transforms import Compose, Lambda, CenterCrop
from torchvision.transforms._transforms_video import NormalizeVideo

# ---------------------------
# Reproducibility
# ---------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# ---------------------------
# Config
# ---------------------------
TRAIN_CSV = "./train_list.csv"
VAL_CSV = "./val_list.csv"
NUM_FRAMES = 13
CROP_SIZE = 160
BATCH_SIZE = 1
ACCUM_STEPS = 8
NUM_EPOCHS = 20
LR = 1e-4
CHECKPOINT_DIR = "./checkpoints"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def load_labeled_paths(csv_path):
    entries = []
    with open(csv_path, "r") as f:
        for line in f:
            path, label = line.strip().rsplit(" ", 1)
            entries.append((path, {"label": int(label)}))
    return entries

train_data = load_labeled_paths(TRAIN_CSV)
val_data = load_labeled_paths(VAL_CSV)


# ---------------------------
# Transforms
# ---------------------------
def make_train_transform():
    return ApplyTransformToKey(
        key="video",
        transform=Compose([
            UniformTemporalSubsample(NUM_FRAMES),
            Lambda(lambda x: x / 255.0),
            NormalizeVideo(mean=[0.45, 0.45, 0.45],
                           std=[0.225, 0.225, 0.225]),
            ShortSideScale(size=182),
            CenterCrop(CROP_SIZE),
        ]),
    )

def make_val_transform():
    return ApplyTransformToKey(
        key="video",
        transform=Compose([
            UniformTemporalSubsample(NUM_FRAMES),
            Lambda(lambda x: x / 255.0),
            NormalizeVideo(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
            ShortSideScale(size=182),
            CenterCrop(CROP_SIZE),
        ]),
    )

train_transform = make_train_transform()
val_transform = make_val_transform()

# ---------------------------
# Datasets
# ---------------------------
train_dataset = LabeledVideoDataset(
    labeled_video_paths=train_data,
    clip_sampler=make_clip_sampler("random", 2.0),
    transform=train_transform,
    decode_audio=False,
)

val_dataset = LabeledVideoDataset(
    labeled_video_paths=val_data,
    clip_sampler=make_clip_sampler("uniform", 2.0),
    transform=val_transform,
    decode_audio=False,
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=1, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=1, pin_memory=True)

# ---------------------------
# Model: X3D-S, Kinetics-PRETRAINED this time, swap head for binary classification
# ---------------------------
model = torch.hub.load("facebookresearch/pytorchvideo", "x3d_s", pretrained=True)

in_features = model.blocks[-1].proj.in_features
model.blocks[-1].proj = nn.Linear(in_features, 2)

model = model.to(DEVICE)

# ---------------------------
# Loss, optimizer, scheduler, AMP scaler
# ---------------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
scaler = GradScaler()

CLASS_NAMES = {0: "NonFight", 1: "Fight"}

# ---------------------------
# Evaluation with per-class breakdown
# ---------------------------
def evaluate():
    model.eval()
    correct, total = 0, 0
    class_correct = {0: 0, 1: 0}
    class_total = {0: 0, 1: 0}

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["video"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            with autocast():
                outputs = model(inputs)
                preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            for label_val in [0, 1]:
                mask = labels == label_val
                class_total[label_val] += mask.sum().item()
                class_correct[label_val] += (preds[mask] == labels[mask]).sum().item()

    model.train()
    overall_acc = correct / total if total > 0 else 0.0
    nonfight_acc = class_correct[0] / class_total[0] if class_total[0] > 0 else 0.0
    fight_acc = class_correct[1] / class_total[1] if class_total[1] > 0 else 0.0
    return overall_acc, nonfight_acc, fight_acc


best_val_acc = 0.0

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    num_batches = 0
    optimizer.zero_grad()

    for step, batch in enumerate(train_loader):
        inputs = batch["video"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels) / ACCUM_STEPS

        scaler.scale(loss).backward()

        if (step + 1) % ACCUM_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        running_loss += loss.item() * ACCUM_STEPS
        num_batches += 1
    if (step + 1) % ACCUM_STEPS != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()   

    
    current_lr = optimizer.param_groups[0]["lr"]

    val_acc, nonfight_acc, fight_acc = evaluate()

    print(
    f"Epoch {epoch+1}/{NUM_EPOCHS} - "
    f"loss: {running_loss/num_batches:.4f} - "
    f"lr: {current_lr:.6f} - "
    f"val_acc: {val_acc:.4f} "
    f"(NonFight: {nonfight_acc:.4f}, Fight: {fight_acc:.4f})"
)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_acc": best_val_acc,
        }, os.path.join(CHECKPOINT_DIR, "best_x3d_s_v2_pretrained.pth"))

        print(f"  Saved new best checkpoint (val_acc={val_acc:.4f})")

    # <-- ALWAYS step once per epoch
    scheduler.step()

    if (epoch + 1) % 5 == 0 and (epoch + 1) != NUM_EPOCHS:
        print("\nCooling down GPU for 1 minute...\n")
        time.sleep(60)




print(f"Training complete. Best val_acc: {best_val_acc:.4f}")
torch.save({
    "epoch": NUM_EPOCHS,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "best_val_acc": best_val_acc,
}, os.path.join(CHECKPOINT_DIR, "last_x3d_s_v2_pretrained.pth"))

print("Saved final checkpoint.")