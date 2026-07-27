import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from collections import deque

from torchvision.transforms import Compose, Lambda, CenterCrop
from torchvision.transforms._transforms_video import NormalizeVideo
from pytorchvideo.transforms import UniformTemporalSubsample, ShortSideScale

# ---------------------------
# Config — must match train_model.py exactly
# ---------------------------
NUM_FRAMES = 13
CROP_SIZE = 160
CHECKPOINT_PATH = "checkpoints/best_x3d_s.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = {0: "NonFight", 1: "Fight"}


class X3DViolenceDetector:
    """
    Wraps a fine-tuned X3D-S checkpoint for clip-level violence
    classification. Expects a buffer of raw BGR frames (as read by
    OpenCV) and returns a (label, confidence) prediction.
    """

    def __init__(self, checkpoint_path: str = CHECKPOINT_PATH, device: str = DEVICE):
        self.device = device

        # Rebuild the same architecture used in training
        model = torch.hub.load("facebookresearch/pytorchvideo", "x3d_s", pretrained=False)
        in_features = model.blocks[-1].proj.in_features
        model.blocks[-1].proj = nn.Linear(in_features, 2)

        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        model.to(device)

        self.model = model

        self.transform = Compose([
            UniformTemporalSubsample(NUM_FRAMES),
            Lambda(lambda x: x / 255.0),
            NormalizeVideo(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
            ShortSideScale(size=182),
            CenterCrop(CROP_SIZE),
        ])

    def predict(self, frames: list[np.ndarray]) -> tuple[str, float]:
        """
        frames: list of BGR frames (H, W, 3) as read by cv2.VideoCapture,
        ideally NUM_FRAMES or more (will be subsampled internally).
        """
        if len(frames) < 4:
            raise ValueError("Need at least a few frames to run X3D inference.")

        # BGR -> RGB, stack into (T, H, W, C) -> tensor (C, T, H, W)
        rgb_frames = []

        for f in frames:
            rgb = f[:, :, ::-1]

            rgb = cv2.resize(
                rgb,
                (320, 320)
            )

            rgb_frames.append(rgb)

        clip = np.stack(rgb_frames, axis=0)




        clip_tensor = torch.from_numpy(clip).permute(3, 0, 1, 2).float()  # (C, T, H, W)

        clip_tensor = self.transform(clip_tensor)
        clip_tensor = clip_tensor.unsqueeze(0).to(self.device)  # (1, C, T, H, W)

        with torch.no_grad():
            """print(
                f"Clip tensor: shape={clip_tensor.shape}, "
                f"min={clip_tensor.min().item():.3f}, "
                f"max={clip_tensor.max().item():.3f}, "
                f"mean={clip_tensor.mean().item():.3f}"
            )
            """
            outputs = self.model(clip_tensor)
            print("Raw logits:", outputs.cpu().numpy())
            probs = torch.softmax(outputs, dim=1)
            conf, pred_idx = torch.max(probs, dim=1)

            label = CLASS_NAMES[int(pred_idx.item())]
            return label, float(conf.item())


           


class ClipBuffer:
    """
    Rolling buffer of raw frames per tracked scene/person-group,
    used to accumulate enough frames before calling X3D.
    Enhanced with motion-based frame selection for better accuracy.
    """

    def __init__(self, maxlen: int = 32):
        self.frames = deque(maxlen=maxlen)
        self.motion_scores = deque(maxlen=maxlen)

    def add(self, frame: np.ndarray):
        self.frames.append(frame.copy())
        # Calculate motion score (simple frame difference from previous)
        if len(self.frames) > 1:
            prev_frame = self.frames[-2]

            if prev_frame.shape != frame.shape:
                prev_frame = cv2.resize(
                    prev_frame,
                    (frame.shape[1], frame.shape[0])
                )

            motion = np.abs(
                frame.astype(np.float32) -
                prev_frame.astype(np.float32)
            ).mean()
            self.motion_scores.append(motion)
        else:
            self.motion_scores.append(0.0)

    def is_ready(self, min_frames: int = 13) -> bool:
        return len(self.frames) >= min_frames

    def get_clip(self) -> list[np.ndarray]:
        # Return motion-selected frames for better X3D input
        if len(self.frames) <= 13:
            return list(self.frames)
        
        # Select frames with highest motion (more likely to contain violence)
        motion_array = np.array(self.motion_scores)
        num_frames = 13
        # Get indices of frames with highest motion
        top_indices = np.argsort(motion_array)[-num_frames:]
        top_indices = np.sort(top_indices)  # Keep temporal order
        
        return [self.frames[i] for i in top_indices]