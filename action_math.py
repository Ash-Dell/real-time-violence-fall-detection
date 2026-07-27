"""
action_math.py — Kinematics Module for Violence Detection

Implements 3rd-order kinematic analysis (Jerk) combined with an
approach-vector / contact-distance check to distinguish a genuine
strike from incidental fast motion (e.g. jumping jacks, waving).
"""

import numpy as np
from typing import List, Tuple, Optional, Dict


# ============================================================
# CONFIGURABLE THRESHOLDS — all four of these are first-pass
# estimates and need empirical tuning against the eval set,
# same as the original JERK_THRESHOLD_NORM was flagged before.
# ============================================================
JERK_THRESHOLD_NORM = 0.6          # scale-invariant jerk threshold (fraction of bbox_height per frame^3)
ALIGNMENT_THRESHOLD = 0.05         # min cosine similarity between limb velocity and approach direction
CONTACT_THRESHOLD_NORM = 0.7       # max limb-to-target distance (fraction of bbox_height) to count as "in range"
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
    """
    NOTE: dt currently defaults to 1.0 (per-frame). This means jerk
    thresholds are effectively tuned against whatever frame rate the
    eval script processes at. If live inference runs at a different
    effective FPS than eval (dropped frames, slower hardware), the same
    real-world motion will produce a different apparent jerk value.
    The correct long-term fix is to thread real elapsed wall-clock time
    through from the capture loop into this dt parameter — flagging
    this here rather than silently leaving it as an unstated assumption.
    """
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
# TARGET SELECTION: where on person B are we checking for contact?
# ============================================================
def _get_target_point(
    history_B: Dict[str, List[Tuple[float, float]]]
) -> Optional[Tuple[float, float]]:
    """
    Best-effort location on person B to treat as the 'target'.
    Prefers head; falls back to shoulder midpoint, then hip midpoint,
    since the head is often exactly what gets occluded during a
    scuffle (arms/hands covering the face).
    """
    head = history_B.get('head', [])
    if head and head[-1] != (0.0, 0.0):
        return head[-1]

    sl = history_B.get('shoulder_L', [])
    sr = history_B.get('shoulder_R', [])
    if sl and sr and sl[-1] != (0.0, 0.0) and sr[-1] != (0.0, 0.0):
        return ((sl[-1][0] + sr[-1][0]) / 2.0, (sl[-1][1] + sr[-1][1]) / 2.0)

    hl = history_B.get('hip_L', [])
    hr = history_B.get('hip_R', [])
    if hl and hr and hl[-1] != (0.0, 0.0) and hr[-1] != (0.0, 0.0):
        return ((hl[-1][0] + hr[-1][0]) / 2.0, (hl[-1][1] + hr[-1][1]) / 2.0)

    return None


# ============================================================
# VIOLENCE DETECTION: Jerk + Approach Vector + Contact Distance
# ============================================================
def detect_advanced_violence(
    history_A: Dict[str, List[Tuple[float, float]]],
    history_B: Dict[str, List[Tuple[float, float]]],
    bbox_height: float,
    jerk_threshold: float = JERK_THRESHOLD_NORM,
    alignment_threshold: float = ALIGNMENT_THRESHOLD,
    contact_threshold_norm: float = CONTACT_THRESHOLD_NORM,
    alpha: float = DEFAULT_EMA_ALPHA,
) -> Tuple[bool, Dict[str, float]]:
    """
    Is a limb of person A striking person B?

    Redesigned from the original: instead of comparing the attacker's
    wrist velocity to person B's OWN head velocity (which is near-zero,
    and therefore gates the whole check off, whenever the victim is
    still or only reacting), this checks the wrist's velocity against
    the direction FROM the wrist TO person B's nearest landmark, and
    additionally requires the wrist to actually be within striking
    range of that landmark. That's the real physical signature of a
    strike: a limb accelerating toward a target that's close enough to
    hit — independent of what the target's own body happens to be doing.

    Three conditions must all hold for a given wrist:
      1. peak jerk over the recent window exceeds jerk_threshold (a sharp,
         non-smooth motion — distinguishes a strike from a steady swing)
      2. the wrist's velocity is meaningfully aligned with the direction
         toward the target (alignment_threshold)
      3. the wrist is within contact_threshold_norm of the target when
         this happens (rules out a fast hand moving toward a distant,
         unrelated person)
    """
    _null_telem: Dict[str, float] = {'jerk': 0.0, 'alignment': 0.0, 'distance_norm': 999.0}

    target_point = _get_target_point(history_B)
    if target_point is None:
        return False, _null_telem

    best_jerk = 0.0
    best_alignment = 0.0
    best_distance_norm = 999.0
    is_violence = False

    for wrist_key in ('wrist_R', 'wrist_L'):
        wrist_history_A = history_A.get(wrist_key, [])
        if len(wrist_history_A) < MIN_HISTORY_FRAMES:
            continue

        filled = _fill_missing(wrist_history_A)
        smoothed = smooth_keypoints(filled, alpha)
        kin = calculate_kinematics(smoothed)
        if kin is None:
            continue

        wrist_pos = np.array(smoothed[-1], dtype=np.float64)
        wrist_vel = kin['velocity_vectors'][-1]

        to_target = np.array(target_point, dtype=np.float64) - wrist_pos
        distance_norm = float(np.linalg.norm(to_target)) / max(bbox_height, 1e-5)

        recent_jerk = kin['jerk_mag'][-3:]
        peak_jerk_raw = float(np.max(recent_jerk)) if len(recent_jerk) > 0 else 0.0
        peak_jerk = peak_jerk_raw / max(bbox_height, 1e-5)

        alignment = _cosine_similarity(wrist_vel, to_target)

        if peak_jerk > best_jerk:
            best_jerk = peak_jerk
            best_alignment = alignment
            best_distance_norm = distance_norm

        within_range = distance_norm < contact_threshold_norm
        approaching = alignment > alignment_threshold

        if within_range and approaching and peak_jerk >= jerk_threshold:
            is_violence = True

    telemetry: Dict[str, float] = {
        'jerk': best_jerk,
        'alignment': best_alignment,
        'distance_norm': best_distance_norm,
    }
    return is_violence, telemetry


# ============================================================
# FALL DETECTION (unchanged — not flagged as broken)
# ============================================================
def detect_fall(
    history: Dict[str, List[Tuple[float, float]]],
    bbox_width: float,
    bbox_height: float,
    velocity_threshold_norm: float = 0.08,
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
    """
    Fill gaps left by low-confidence keypoints (encoded upstream as the
    (0.0, 0.0) sentinel) via linear interpolation between the nearest
    valid points on either side of the gap.

    The previous version repeated the last valid point forward through
    the whole gap — which flattens velocity to zero for the gap's
    duration and then produces an artificial one-frame velocity/jerk
    spike the instant tracking recovers. That's backwards for this use
    case: fast motion (a punch, a shove) is exactly when pose confidence
    tends to dip from motion blur. Interpolating instead approximates
    the real, continuous motion through the gap.

    Leading/trailing gaps (no valid point on one side) fall back to the
    nearest available valid value, since there's nothing to interpolate
    from.
    """
    n = len(history)
    if n == 0:
        return history

    is_valid = [not (p[0] == 0.0 and p[1] == 0.0) for p in history]
    if not any(is_valid):
        return list(history)

    filled: List[Tuple[float, float]] = list(history)

    i = 0
    while i < n:
        if is_valid[i]:
            i += 1
            continue

        gap_start = i
        while i < n and not is_valid[i]:
            i += 1
        gap_end = i  # first valid index after the gap, or n

        left_val = filled[gap_start - 1] if gap_start > 0 else None
        right_val = filled[gap_end] if gap_end < n else None

        if left_val is not None and right_val is not None:
            span = gap_end - (gap_start - 1)
            for k in range(gap_start, gap_end):
                t = (k - (gap_start - 1)) / span
                x = left_val[0] + (right_val[0] - left_val[0]) * t
                y = left_val[1] + (right_val[1] - left_val[1]) * t
                filled[k] = (x, y)
        elif left_val is not None:
            for k in range(gap_start, gap_end):
                filled[k] = left_val
        elif right_val is not None:
            for k in range(gap_start, gap_end):
                filled[k] = right_val

    return filled


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