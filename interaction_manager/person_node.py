from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class PersonNode:
    """
    Represents one tracked person in the current frame.
    """

    tracker_id: int

    bbox: np.ndarray              # [x1, y1, x2, y2]
    keypoints: np.ndarray         # RTMPose keypoints
    confidence: float

    frame_id: int

    def update(
        self,
        bbox: np.ndarray,
        keypoints: np.ndarray,
        confidence: float,
        frame_id: int,
    ) -> None:
        """
        Update the person's latest state.
        """
        self.bbox = bbox
        self.keypoints = keypoints
        self.confidence = confidence
        self.frame_id = frame_id

    @property
    def centroid(self):
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def width(self):
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self):
        return self.bbox[3] - self.bbox[1]