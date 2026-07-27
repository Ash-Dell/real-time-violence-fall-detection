import os
import argparse
from collections import defaultdict, deque

import cv2
import numpy as np
import supervision as sv

from action_math import (
    detect_advanced_violence,
    detect_fall,
    extract_keypoint,
)
from behavior_helpers import mark_track_lost, should_flag_violence, update_track_state
from vision_models import (
    create_pose_pipeline,
    match_keypoints_to_bbox,
)
from x3d_adapter import X3DViolenceDetector, ClipBuffer

# Constants
HISTORY_LENGTH = 15
PROXIMITY_THRESHOLD = 500
X3D_MIN_PEOPLE = 2
X3D_CLIP_LEN = 32  # frames buffered before running X3D
MOTION_THRESHOLD = 6.0
POSE_DETECTION_CONFIDENCE = 0.35

KP_NOSE = 0
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_HIP = 11
KP_RIGHT_HIP = 12
KP_LEFT_WRIST = 9
KP_RIGHT_WRIST = 10
KP_LEFT_ELBOW = 7
KP_RIGHT_ELBOW = 8


def update_person_state(tracker_id: int, keypoints: np.ndarray, state: dict):
    head = extract_keypoint(keypoints, KP_NOSE)
    wr = extract_keypoint(keypoints, KP_RIGHT_WRIST)
    wl = extract_keypoint(keypoints, KP_LEFT_WRIST)
    er = extract_keypoint(keypoints, KP_RIGHT_ELBOW)
    el = extract_keypoint(keypoints, KP_LEFT_ELBOW)
    sl = extract_keypoint(keypoints, KP_LEFT_SHOULDER)
    sr = extract_keypoint(keypoints, KP_RIGHT_SHOULDER)
    hl = extract_keypoint(keypoints, KP_LEFT_HIP)
    hr = extract_keypoint(keypoints, KP_RIGHT_HIP)

    state['head'].append(head if head else (0.0, 0.0))
    state['wrist_R'].append(wr if wr else (0.0, 0.0))
    state['wrist_L'].append(wl if wl else (0.0, 0.0))
    state['elbow_R'].append(er if er else (0.0, 0.0))
    state['elbow_L'].append(el if el else (0.0, 0.0))
    state['shoulder_L'].append(sl if sl else (0.0, 0.0))
    state['shoulder_R'].append(sr if sr else (0.0, 0.0))
    state['hip_L'].append(hl if hl else (0.0, 0.0))
    state['hip_R'].append(hr if hr else (0.0, 0.0))

    valid_points = [p for p in [head, sl, sr, hl, hr] if p is not None]
    if valid_points:
        cx = sum(p[0] for p in valid_points) / len(valid_points)
        cy = sum(p[1] for p in valid_points) / len(valid_points)
        state['centroid'] = (cx, cy)


def check_proximity_trigger(person_states: dict) -> bool:
    """
    Returns True if 2+ tracked people currently exist and at least one
    pair is within PROXIMITY_THRESHOLD of each other. This is the gate
    that decides whether X3D is worth calling this frame.
    """
    ids = list(person_states.keys())
    if len(ids) < X3D_MIN_PEOPLE:
        return False

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            cx_a, cy_a = person_states[ids[i]]['centroid']
            cx_b, cy_b = person_states[ids[j]]['centroid']
            dist = np.sqrt((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2)
            if dist < PROXIMITY_THRESHOLD:
                return True
    return False


def check_motion_intensity(person_states: dict) -> bool:
    """
    Returns True if there's significant motion among tracked people.
    This adds an additional gate to ensure X3D only runs when people
    are actually moving (not just standing close together).
    """
    if len(person_states) < X3D_MIN_PEOPLE:
        return False
    
    total_motion = 0.0
    for tracker_id, state in person_states.items():
        # Check wrist motion as indicator of overall body movement
        wrist_r = list(state.get('wrist_R', deque(maxlen=HISTORY_LENGTH)))
        wrist_l = list(state.get('wrist_L', deque(maxlen=HISTORY_LENGTH)))
        
        if len(wrist_r) >= 2 and len(wrist_l) >= 2:
            # Calculate motion from recent frames
            motion_r = np.linalg.norm(np.array(wrist_r[-1]) - np.array(wrist_r[-2]))
            motion_l = np.linalg.norm(np.array(wrist_l[-1]) - np.array(wrist_l[-2]))
            total_motion += motion_r + motion_l
    
    avg_motion = total_motion / len(person_states) if person_states else 0
    return avg_motion > MOTION_THRESHOLD

COCO_SKELETON = [
    (5, 7), (7, 9), (6, 8), (8, 10),      # arms
    (5, 6), (5, 11), (6, 12), (11, 12),   # torso
    (11, 13), (13, 15), (12, 14), (14, 16),  # legs
]

def draw_keypoints(frame, keypoints, conf_thresh=0.3):
    for x, y, conf in keypoints:
        if conf > conf_thresh:
            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
    for i, j in COCO_SKELETON:
        if keypoints[i][2] > conf_thresh and keypoints[j][2] > conf_thresh:
            pt1 = (int(keypoints[i][0]), int(keypoints[i][1]))
            pt2 = (int(keypoints[j][0]), int(keypoints[j][1]))
            cv2.line(frame, pt1, pt2, (255, 255, 0), 2)



def main():
    parser = argparse.ArgumentParser(description="Real-Time Webcam Test for Surveillance Pipeline")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    args = parser.parse_args()

    print("[*] Loading RTMPose model...")
    pose_model = create_pose_pipeline()

    print("[*] Loading X3D-S violence detector...")
    x3d_detector = X3DViolenceDetector()
    clip_buffer = ClipBuffer(maxlen=X3D_CLIP_LEN)

    print("[*] Initializing ByteTrack...")
    byte_tracker = sv.ByteTrack(
        track_activation_threshold=0.55,
        lost_track_buffer=10,
        minimum_matching_threshold=0.8,
        frame_rate=30,
    )

    print(f"[*] Opening Camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[!] Error opening camera index {args.camera}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    person_states = defaultdict(lambda: {
        'wrist_R': deque(maxlen=HISTORY_LENGTH),
        'wrist_L': deque(maxlen=HISTORY_LENGTH),
        'elbow_R': deque(maxlen=HISTORY_LENGTH),
        'elbow_L': deque(maxlen=HISTORY_LENGTH),
        'head': deque(maxlen=HISTORY_LENGTH),
        'shoulder_L': deque(maxlen=HISTORY_LENGTH),
        'shoulder_R': deque(maxlen=HISTORY_LENGTH),
        'hip_L': deque(maxlen=HISTORY_LENGTH),
        'hip_R': deque(maxlen=HISTORY_LENGTH),
        'centroid': (0.0, 0.0),
        'fall_streak': 0,
        'drop_streak': 0,
        'is_fallen': False,
        'bbox_ar': 1.0,
        'bbox_height': 1.0,
        'violence_streak': 0,
        'bbox_history': deque(maxlen=5),  # For bounding box smoothing
    })

    x3d_label = "N/A"
    x3d_confidence = 0.0
    x3d_confidence_history = deque(maxlen=5)  # For confidence smoothing
    track_states = {}

    print("[*] Starting Live Camera Feed. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[!] Failed to grab frame from camera.")
            break

        frame = cv2.flip(frame, 1)

        pose_results = pose_model.predict(frame, confidence=POSE_DETECTION_CONFIDENCE, det_thr=POSE_DETECTION_CONFIDENCE)
        detections = pose_results.to_supervision()

        tracked = byte_tracker.update_with_detections(detections)

        active_tracker_ids = set()
        red_boxes = set()
        fall_telemetry = {}
        violence_telemetry = {}

        for idx in range(len(tracked)):
            tracker_id = int(tracked.tracker_id[idx])
            active_tracker_ids.add(tracker_id)
            track_state = track_states.setdefault(tracker_id, {})

            tracked_bbox = tracked.xyxy[idx]
            confidence_value = float(tracked.confidence[idx]) if tracked.confidence is not None else 0.0
            smoothed_bbox, track_confidence, _ = update_track_state(track_state, tracked_bbox, confidence_value)
            tracked.xyxy[idx] = smoothed_bbox
            person_states[tracker_id]['track_confidence'] = track_confidence

            width = max(1e-5, smoothed_bbox[2] - smoothed_bbox[0])
            height = max(1e-5, smoothed_bbox[3] - smoothed_bbox[1])
            person_states[tracker_id]['bbox_ar'] = width / height
            person_states[tracker_id]['bbox_height'] = height

            person_states[tracker_id]['bbox_history'].append(smoothed_bbox)

            kps = match_keypoints_to_bbox(pose_results, smoothed_bbox)
            if kps is not None:
                update_person_state(tracker_id, kps, person_states[tracker_id])
                draw_keypoints(frame, kps)

            history = {k: list(v) for k, v in person_states[tracker_id].items() if isinstance(v, deque)}
            is_fall_now, t_fall = detect_fall(history, width, height)

            if is_fall_now:
                person_states[tracker_id]['drop_streak'] += 1
            else:
                person_states[tracker_id]['drop_streak'] = 0

            # Require the drop condition to persist for 3 consecutive frames
            # before starting the fall latch — filters out single-frame
            # spikes from fast head bows/nods that aren't real falls.
            if person_states[tracker_id]['drop_streak'] >= 3 and person_states[tracker_id]['fall_streak'] == 0:
                person_states[tracker_id]['fall_streak'] = 1
            elif person_states[tracker_id]['fall_streak'] > 0:
                is_wide = t_fall['ar'] > 1.0
                is_crumpled = t_fall['crumple'] < 0.25
                if is_wide or is_crumpled:
                    person_states[tracker_id]['fall_streak'] += 1
                else:
                    person_states[tracker_id]['fall_streak'] = 0
                    person_states[tracker_id]['is_fallen'] = False

            if person_states[tracker_id]['fall_streak'] > 10:
                person_states[tracker_id]['is_fallen'] = True

            fall_telemetry[tracker_id] = {
                'dy_norm': t_fall['dy_norm'],
                'ar': t_fall['ar'],
                'crumple': t_fall['crumple'],
                'is_fall': person_states[tracker_id]['is_fallen']
            }

        stale_ids = [tid for tid in list(person_states.keys()) if tid not in active_tracker_ids]
        for tid in stale_ids:
            state = track_states.get(tid)
            if state is None:
                del person_states[tid]
                continue
            if not mark_track_lost(state):
                del person_states[tid]
                track_states.pop(tid, None)

        tracked_ids = list(person_states.keys())
        for i in range(len(tracked_ids)):
            for j in range(i + 1, len(tracked_ids)):
                id_a = tracked_ids[i]
                id_b = tracked_ids[j]

                cx_a, cy_a = person_states[id_a]['centroid']
                cx_b, cy_b = person_states[id_b]['centroid']
                dist = np.sqrt((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2)

                if dist < PROXIMITY_THRESHOLD:
                    hist_a = {k: list(v) for k, v in person_states[id_a].items() if isinstance(v, deque)}
                    hist_b = {k: list(v) for k, v in person_states[id_b].items() if isinstance(v, deque)}

                    v_a_b, telem_a_b = detect_advanced_violence(hist_a, hist_b, person_states[id_a]['bbox_height'])
                    v_b_a, telem_b_a = detect_advanced_violence(hist_b, hist_a, person_states[id_b]['bbox_height'])

                    telemetry = {
                        'score': max(telem_a_b['score'], telem_b_a['score']),
                        'relative_velocity': max(telem_a_b['relative_velocity'], telem_b_a['relative_velocity']),
                        'relative_acceleration': max(telem_a_b['relative_acceleration'], telem_b_a['relative_acceleration']),
                        'entropy': max(telem_a_b['entropy'], telem_b_a['entropy']),
                        'distance_norm': np.sqrt((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2) / max(person_states[id_a]['bbox_height'], 1e-5),
                    }
                    is_violence = should_flag_violence(telemetry, person_states[id_a], person_states[id_b]) or (v_a_b and v_b_a)
                    if is_violence:
                        person_states[id_a]['violence_streak'] += 1
                        person_states[id_b]['violence_streak'] += 1
                    else:
                        person_states[id_a]['violence_streak'] = max(0, person_states[id_a]['violence_streak'] - 1)
                        person_states[id_b]['violence_streak'] = max(0, person_states[id_b]['violence_streak'] - 1)

                    violence_telemetry[(id_a, id_b)] = {
                        'jerk': max(telem_a_b['jerk'], telem_b_a['jerk']),
                        'alignment': max(telem_a_b['alignment'], telem_b_a['alignment']),
                        'score': telemetry['score'],
                        'is_violence': is_violence
                    }

        # ---------------------------------------------------------
        # X3D gating: only buffer + run X3D when 2+ people are close AND moving
        # ---------------------------------------------------------
        trigger_active = check_proximity_trigger(person_states)
        motion_active = check_motion_intensity(person_states)
        
        if trigger_active and motion_active:
            clip_buffer.add(frame)
            if clip_buffer.is_ready(min_frames=13):
                new_label, new_confidence = x3d_detector.predict(clip_buffer.get_clip())
                
                # Confidence smoothing: average with recent predictions
                if new_label != "N/A":
                    x3d_confidence_history.append(new_confidence)
                    smoothed_confidence = sum(x3d_confidence_history) / len(x3d_confidence_history)
                    x3d_label = new_label
                    x3d_confidence = smoothed_confidence
                else:
                    x3d_label = new_label
                    x3d_confidence = new_confidence
                    x3d_confidence_history.clear()
        else:
            # Not enough people/proximity/motion — don't waste compute, reset buffer
            clip_buffer = ClipBuffer(maxlen=X3D_CLIP_LEN)
            x3d_label, x3d_confidence = "N/A", 0.0
            x3d_confidence_history.clear()

        # Visual Annotation Overlay
        headcount = len(tracked)
        cv2.putText(frame, f"Headcount: {headcount}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        x3d_color = (0, 0, 255) if x3d_label == "Fight" else (0, 255, 0)
        cv2.putText(frame, f"X3D: {x3d_label} ({x3d_confidence:.2f})", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, x3d_color, 2)

        x3d_flagging_violence = (x3d_label == "Fight" and x3d_confidence >= 0.25)  # Lowered threshold for higher sensitivity

        for idx in range(len(tracked)):
            tid = int(tracked.tracker_id[idx])
            x1, y1, x2, y2 = map(int, tracked.xyxy[idx])

            label = "Normal"
            if x3d_flagging_violence and tid in person_states:
                # X3D is the primary violence signal now — if it fired,
                # mark everyone currently in the triggering proximity group.
                label = "Violence! (X3D)"
                red_boxes.add(tid)
            else:
                # Re-enable heuristic with high-confidence gating as secondary system
                for pair, v_telem in violence_telemetry.items():
                    if tid in pair and v_telem['is_violence'] and v_telem.get('score', 0) >= 3.0:
                        label = "Violence? (heuristic high-conf)"
                        red_boxes.add(tid)
                        break

            color = (0, 0, 255) if tid in red_boxes else (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

          

        for pair, v_telem in violence_telemetry.items():
            id_a, id_b = pair
            if id_a in person_states and id_b in person_states:
                cx_a, cy_a = person_states[id_a]['centroid']
                cx_b, cy_b = person_states[id_b]['centroid']
                mx, my = int((cx_a + cx_b) / 2), int((cy_a + cy_b) / 2)

                text = f"Jerk: {v_telem['jerk']:.0f} | Align: {v_telem['alignment']:.2f}"
                cv2.putText(frame, text, (mx - 50, my),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        cv2.imshow('Real-Time Webcam Pipeline', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[*] Loop broken by user.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[*] Camera feed closed.")


if __name__ == "__main__":
    main()