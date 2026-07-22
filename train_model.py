import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

import pytorchvideo.data
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    UniformTemporalSubsample,
    ShortSideScale,
)
from pytorchvideo.data import LabeledVideoDataset, make_clip_sampler
from torchvision.transforms import Compose, Lambda, CenterCrop
from torchvision.transforms._transforms_video import NormalizeVideo

# ---------------------------
# Config
# ---------------------------
TRAIN_CSV = "./train_list.csv"
VAL_CSV = "./val_list.csv"
NUM_FRAMES = 13          # X3D-S default
CROP_SIZE = 160          # X3D-S default
BATCH_SIZE = 2           # real batch size (VRAM-safe for 4GB)
ACCUM_STEPS = 4          # simulate effective batch size of 8
NUM_EPOCHS = 35
LR = 1e-4
CHECKPOINT_DIR = "./checkpoints"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ---------------------------
# Read CSV lists into labeled video paths format PyTorchVideo expects
# ---------------------------
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
def make_transform():
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

train_transform = make_transform()
val_transform = make_transform()

# ---------------------------
# Datasets
# ---------------------------
train_dataset = LabeledVideoDataset(
    labeled_video_paths=train_data,
    clip_sampler=make_clip_sampler("random", 2.0),  # 2 sec clips
    transform=train_transform,
    decode_audio=False,
)

val_dataset = LabeledVideoDataset(
    labeled_video_paths=val_data,
    clip_sampler=make_clip_sampler("uniform", 2.0),
    transform=val_transform,
    decode_audio=False,
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

# ---------------------------
# Model: X3D-S, pretrained on Kinetics, swap head for binary classification
# ---------------------------
model = torch.hub.load("facebookresearch/pytorchvideo", "x3d_s", pretrained=True)

# Replace the final projection layer for binary output (Fight / NonFight)
in_features = model.blocks[-1].proj.in_features
model.blocks[-1].proj = nn.Linear(in_features, 2)

model = model.to(DEVICE)

# ---------------------------
# Loss, optimizer, AMP scaler
# ---------------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scaler = GradScaler()

# ---------------------------
# Training loop
# ---------------------------
def evaluate():
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["video"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            with autocast():
                outputs = model(inputs)
                preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    model.train()
    return correct / total if total > 0 else 0.0

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

    val_acc = evaluate()
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} - loss: {running_loss/num_batches:.4f} - val_acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_x3d_s.pth"))
        print(f"  Saved new best checkpoint (val_acc={val_acc:.4f})")

print(f"Training complete. Best val_acc: {best_val_acc:.4f}")