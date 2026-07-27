from collections import deque
from x3d_adapter import ClipBuffer
from enum import Enum
from typing import List

from .person_node import PersonNode


class InteractionState(Enum):
    CREATED = 0
    BUFFERING = 1
    READY = 2
    ACTIVE = 3
    ENDING = 4
    EXPIRED = 5


class InteractionTrack:
    """
    Represents one persistent interaction.
    """

    def __init__(self, interaction_id: int, members: list[PersonNode], frame_id: int):

        self.interaction_id = interaction_id
        self.members = members

        self.created_frame = frame_id
        self.last_seen_frame = frame_id

        self.state = InteractionState.CREATED

        self.clip_buffer = ClipBuffer(maxlen=32)

        self.predictions: List[float] = []
        self.last_prediction = None
        self.last_confidence = 0.0

        self.roi = None
        self.update_roi()

    def update(self, members: list[PersonNode], frame_id: int):

        self.members = members
        self.last_seen_frame = frame_id
        self.update_roi()

    def update_roi(self):

        x1 = min(p.bbox[0] for p in self.members)
        y1 = min(p.bbox[1] for p in self.members)
        x2 = max(p.bbox[2] for p in self.members)
        y2 = max(p.bbox[3] for p in self.members)

        self.roi = (x1, y1, x2, y2)

    def add_frame(self, frame):

        self.clip_buffer.add(frame)
        print(
            f"Interaction {self.interaction_id}: "
            f"buffer={len(self.clip_buffer.frames)}"
        )

    def store_prediction(self, label: str, confidence: float):

        self.last_prediction = label
        self.last_confidence = confidence

        self.predictions.append(confidence)

    def is_ready(self, min_frames=13):
        return self.clip_buffer.is_ready(min_frames)

    def get_clip(self):
        return self.clip_buffer.get_clip()

    def is_violent(self):
        return (
            self.last_prediction == "Fight"
            and self.last_confidence >= 0.25
        )    


    def contains_person(self, tracker_id):

        return any(
            p.tracker_id == tracker_id
            for p in self.members
        )    