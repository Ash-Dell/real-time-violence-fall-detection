"""
action_math.py — Advanced Kinematics Module for Violence Detection

Implements 3rd-order kinematic analysis (Jerk) and vector alignment
(momentum transfer via dot product) to distinguish genuine violent
impacts from periodic exercise motions like jumping jacks.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict


# ============================================================
# CONFIGURABLE THRESHOLDS
# ============================================================
JERK_THRESHOLD_NORM = 0.6  # scale-invariant jerk threshold (fraction of bbox_height per frame^3) — needs tuning
MOMENTUM_TRANSFER_THRESHOLD = 0.05
MIN_HISTORY_FRAMES = 5
DEFAULT_EMA_ALPHA = 0.5


# ============================================================
# SMOOTHING: Exponential Moving Average (EMA)
# ============================================================
def smooth_keypoints(
    history: List[Tuple[float, float]],
    alpha: float = DEFAULT_EMA_ALPHA
) -> List[Tuple[float, float]]:
    if not history:
        return []
    smoothed = [history[0]]
    for i in range(1, len(history)):
        prev_x, prev_y = smoothed[i - 1]
        curr_x, curr_y = history[i]
        sx = alpha * curr_x + (1.0 - alpha) * prev_x
        sy = alpha * curr_y + (1.0 - alpha) * prev_y
        smoothed.append((sx, sy))
    return smoothed


# ============================================================
# KINEMATICS: Velocity, Acceleration, Jerk
# ============================================================
def calculate_kinematics(
    positions: List[Tuple[float, float]],
    dt: float = 1.0
) -> Optional[Dict]:
    if len(positions) < 4:
        return None
    pos = np.array(positions, dtype=np.float64)
    velocity = np.diff(pos, axis=0) / dt
    velocity_mag = np.linalg.norm(velocity, axis=1)
    acceleration = np.diff(velocity, axis=0) / dt
    acceleration_mag = np.linalg.norm(acceleration, axis=1)
    jerk = np.diff(acceleration, axis=0) / dt
    jerk_mag = np.linalg.norm(jerk, axis=1)
    return {
        'velocity_vectors': velocity,
        'velocity_mag': velocity_mag,
        'acceleration_mag': acceleration_mag,
        'jerk_mag': jerk_mag,
    }


# ============================================================
# VIOLENCE DETECTION: Jerk + Momentum Transfer (scale-invariant)
# ============================================================
def detect_advanced_violence(
    history_A: Dict[str, List[Tuple[float, float]]],
    history_B: Dict[str, List[Tuple[float, float]]],
    bbox_height: float,
    jerk_threshold: float = JERK_THRESHOLD_NORM,
    momentum_threshold: float = MOMENTUM_TRANSFER_THRESHOLD,
    alpha: float = DEFAULT_EMA_ALPHA
) -> Tuple[bool, Dict[str, float]]:
    """
    Determines if Person A is striking Person B. Jerk is now normalized
    by bbox_height so detection works consistently regardless of subject
    distance from camera (matches detect_fall()'s scale-invariant approach).
    """
    _null_telem: Dict[str, float] = {'jerk': 0.0, 'alignment': 0.0}

    head_history_B = history_B.get('head', [])
    if len(head_history_B) < MIN_HISTORY_FRAMES:
        return False, _null_telem

    head_history_B = _fill_missing(head_history_B)
    smoothed_head_B = smooth_keypoints(head_history_B, alpha)
    kin_head_B = calculate_kinematics(smoothed_head_B)
    if kin_head_B is None:
        return False, _null_telem

    head_vel = kin_head_B['velocity_vectors'][-1]

    best_jerk = 0.0
    best_cos_sim = 0.0
    is_violence = False

    for wrist_key in ['wrist_R', 'wrist_L']:
        wrist_history_A = history_A.get(wrist_key, [])
        if len(wrist_history_A) < MIN_HISTORY_FRAMES:
            continue

        wrist_history_A = _fill_missing(wrist_history_A)
        smoothed_wrist_A = smooth_keypoints(wrist_history_A, alpha)
        kin_wrist_A = calculate_kinematics(smoothed_wrist_A)

        if kin_wrist_A is None:
            continue

        recent_jerk = kin_wrist_A['jerk_mag'][-3:]
        peak_jerk_raw = float(np.max(recent_jerk)) if len(recent_jerk) > 0 else 0.0
        peak_jerk = peak_jerk_raw / max(bbox_height, 1e-5)  # scale-invariant
        wrist_vel = kin_wrist_A['velocity_vectors'][-1]

        cos_sim = _cosine_similarity(wrist_vel, head_vel)

        if peak_jerk > best_jerk:
            best_jerk = peak_jerk
            best_cos_sim = cos_sim

        if peak_jerk > jerk_threshold * 1.5 and cos_sim > 0.0:
            is_violence = True
        elif peak_jerk >= jerk_threshold and cos_sim >= momentum_threshold:
            is_violence = True

    telemetry: Dict[str, float] = {
        'jerk': best_jerk,
        'alignment': best_cos_sim,
    }

    return is_violence, telemetry


# ============================================================
# FALL DETECTION (Robust CV Heuristic) — with debounce to reduce
# false positives from fast-but-brief head movements
# ============================================================
def detect_fall(
    history: Dict[str, List[Tuple[float, float]]],
    bbox_width: float,
    bbox_height: float,
    velocity_threshold_norm: float = 0.08,   # raised from 0.05 to reduce false triggers on quick head bobs
    crumple_threshold_norm: float = 0.25
) -> Tuple[bool, Dict[str, float]]:
    _null_telem: Dict[str, float] = {'dy_norm': 0.0, 'ar': 0.0, 'crumple': 0.0}

    if bbox_height < 1e-5:
        return False, _null_telem

    head_hist = history.get('head', [])
    if len(head_hist) < 3:
        return False, _null_telem

    smoothed_head = smooth_keypoints(head_hist)
    recent_positions = smoothed_head[-3:]
    dy = float(recent_positions[-1][1] - recent_positions[-2][1])
    dy_norm = dy / bbox_height

    bbox_ar = bbox_width / bbox_height

    hip_L = history.get('hip_L', [])
    hip_R = history.get('hip_R', [])

    crumple_ratio = 1.0
    if hip_L and hip_R:
        hip_mid_y = (hip_L[-1][1] + hip_R[-1][1]) / 2.0
        head_y = recent_positions[-1][1]
        head_to_hip_dist = max(0.0, hip_mid_y - head_y)
        crumple_ratio = head_to_hip_dist / bbox_height

    telemetry: Dict[str, float] = {
        'dy_norm': dy_norm,
        'ar': bbox_ar,
        'crumple': crumple_ratio
    }

    is_dropping = dy_norm > velocity_threshold_norm
    is_wide = bbox_ar > 1.0
    is_crumpled = crumple_ratio < crumple_threshold_norm

    if is_dropping and (is_wide or is_crumpled):
        return True, telemetry

    return False, telemetry


# ============================================================
# UTILITY FUNCTIONS
# ============================================================
def _fill_missing(
    history: List[Tuple[float, float]]
) -> List[Tuple[float, float]]:
    if not history:
        return history
    cleaned = [history[0]]
    for i in range(1, len(history)):
        x, y = history[i]
        if x == 0.0 and y == 0.0:
            cleaned.append(cleaned[i - 1])
        else:
            cleaned.append((x, y))
    return cleaned


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    mag_a = np.linalg.norm(vec_a)
    mag_b = np.linalg.norm(vec_b)
    if mag_a < 1e-8 or mag_b < 1e-8:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (mag_a * mag_b))


def extract_keypoint(
    keypoints: np.ndarray,
    index: int
) -> Optional[Tuple[float, float]]:
    if keypoints is None or len(keypoints) == 0:
        return None
    if index < 0 or index >= len(keypoints):
        return None
    kp = keypoints[index]
    if len(kp) >= 3:
        x, y, conf = float(kp[0]), float(kp[1]), float(kp[2])
        if conf < 0.3:
            return None
        return (x, y)
    x, y = float(kp[0]), float(kp[1])
    if x == 0.0 and y == 0.0:
        return None
    return (x, y)