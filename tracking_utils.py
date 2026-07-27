import numpy as np


def update_track_state(track_state, bbox, confidence):
    """
    Smooth bounding boxes and keep track confidence.
    """

    alpha = 0.7

    previous_bbox = track_state.get("bbox")

    if previous_bbox is None:
        smoothed_bbox = bbox.copy()
    else:
        smoothed_bbox = (
            alpha * previous_bbox
            + (1 - alpha) * bbox
        )

    track_state["bbox"] = smoothed_bbox
    track_state["confidence"] = confidence
    track_state["lost_frames"] = 0

    return smoothed_bbox, confidence, True


def mark_track_lost(track_state, max_lost=10):
    """
    Returns True if the track should be kept alive.
    Returns False if it should be deleted.
    """

    lost = track_state.get("lost_frames", 0) + 1
    track_state["lost_frames"] = lost

    return lost <= max_lost