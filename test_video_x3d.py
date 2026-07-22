import argparse
from collections import defaultdict, deque

import cv2
import numpy as np
import supervision as sv

from action_math import detect_advanced_violence, detect_fall, extract_keypoint
from vision_models import create_pose_pipeline, match_keypoints_to_bbox
from x3d_adapter import X3DViolenceDetector, ClipBuffer
from pathlib import Path
from datetime import datetime

HISTORY_LENGTH = 15
PROXIMITY_THRESHOLD = 500.0
X3D_MIN_PEOPLE = 2
X3D_CLIP_LEN = 32

KP_NOSE = 0
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_HIP = 11
KP_RIGHT_HIP = 12
KP_LEFT_WRIST = 9
KP_RIGHT_WRIST = 10

COCO_SKELETON = [
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
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


def update_person_state(tracker_id, keypoints, state):
    head = extract_keypoint(keypoints, KP_NOSE)
    wr = extract_keypoint(keypoints, KP_RIGHT_WRIST)
    wl = extract_keypoint(keypoints, KP_LEFT_WRIST)
    sl = extract_keypoint(keypoints, KP_LEFT_SHOULDER)
    sr = extract_keypoint(keypoints, KP_RIGHT_SHOULDER)
    hl = extract_keypoint(keypoints, KP_LEFT_HIP)
    hr = extract_keypoint(keypoints, KP_RIGHT_HIP)

    state['head'].append(head if head else (0.0, 0.0))
    state['wrist_R'].append(wr if wr else (0.0, 0.0))
    state['wrist_L'].append(wl if wl else (0.0, 0.0))
    state['shoulder_L'].append(sl if sl else (0.0, 0.0))
    state['shoulder_R'].append(sr if sr else (0.0, 0.0))
    state['hip_L'].append(hl if hl else (0.0, 0.0))
    state['hip_R'].append(hr if hr else (0.0, 0.0))

    valid_points = [p for p in [head, sl, sr, hl, hr] if p is not None]
    if valid_points:
        cx = sum(p[0] for p in valid_points) / len(valid_points)
        cy = sum(p[1] for p in valid_points) / len(valid_points)
        state['centroid'] = (cx, cy)


def check_proximity_trigger(person_states):
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


def run_on_video(path, pose_model, x3d_detector):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"[!] Could not open video: {path}")
        return

      
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Create output folder
    output_dir = Path("results/videos")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Unique filename using timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"result_{timestamp}.mp4"

    out_writer = cv2.VideoWriter(
    str(output_path),
    fourcc,
    fps,
    (frame_width, frame_height)
)

    print(f"[*] Saving output to: {output_path}")   

    byte_tracker = sv.ByteTrack(
        track_activation_threshold=0.25,
        lost_track_buffer=30,
        minimum_matching_threshold=0.8,
        frame_rate=30,
    )

    person_states = defaultdict(lambda: {
        'wrist_R': deque(maxlen=HISTORY_LENGTH),
        'wrist_L': deque(maxlen=HISTORY_LENGTH),
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
    })

    clip_buffer = ClipBuffer(maxlen=X3D_CLIP_LEN)
    x3d_label, x3d_confidence = "N/A", 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        pose_results = pose_model.predict(frame, confidence=0.5)
        detections = pose_results.to_supervision()
        tracked = byte_tracker.update_with_detections(detections)

        active_ids = set()
        fall_telemetry = {}
        violence_telemetry = {}
        red_boxes = set()

        for idx in range(len(tracked)):
            tracker_id = int(tracked.tracker_id[idx])
            active_ids.add(tracker_id)
            bbox = tracked.xyxy[idx]

            width = max(1e-5, bbox[2] - bbox[0])
            height = max(1e-5, bbox[3] - bbox[1])
            person_states[tracker_id]['bbox_ar'] = width / height
            person_states[tracker_id]['bbox_height'] = height

            kps = match_keypoints_to_bbox(pose_results, bbox)
            if kps is not None:
                update_person_state(tracker_id, kps, person_states[tracker_id])
                draw_keypoints(frame, kps)

            history = {k: list(v) for k, v in person_states[tracker_id].items() if isinstance(v, deque)}
            is_fall_now, t_fall = detect_fall(history, width, height)

            if is_fall_now:
                person_states[tracker_id]['drop_streak'] += 1
            else:
                person_states[tracker_id]['drop_streak'] = 0

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
                'is_fall': person_states[tracker_id]['is_fallen']
            }
            if person_states[tracker_id]['is_fallen']:
                red_boxes.add(tracker_id)

        stale = [tid for tid in person_states if tid not in active_ids]
        for tid in stale:
            del person_states[tid]

        ids = list(person_states.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                id_a, id_b = ids[i], ids[j]
                cx_a, cy_a = person_states[id_a]['centroid']
                cx_b, cy_b = person_states[id_b]['centroid']
                dist = np.sqrt((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2)
                if dist < PROXIMITY_THRESHOLD:
                    hist_a = {k: list(v) for k, v in person_states[id_a].items() if isinstance(v, deque)}
                    hist_b = {k: list(v) for k, v in person_states[id_b].items() if isinstance(v, deque)}
                    v_a_b, telem_a_b = detect_advanced_violence(hist_a, hist_b, person_states[id_a]['bbox_height'])
                    v_b_a, telem_b_a = detect_advanced_violence(hist_b, hist_a, person_states[id_b]['bbox_height'])
                    violence_telemetry[(id_a, id_b)] = {
                        'is_violence': v_a_b or v_b_a
                    }

        trigger_active = check_proximity_trigger(person_states)
        if trigger_active:
            clip_buffer.add(frame)
            if clip_buffer.is_ready(min_frames=13):
                x3d_label, x3d_confidence = x3d_detector.predict(clip_buffer.get_clip())
        else:
            clip_buffer = ClipBuffer(maxlen=X3D_CLIP_LEN)
            x3d_label, x3d_confidence = "N/A", 0.0

        x3d_flagging_violence = (x3d_label == "Fight" and x3d_confidence >= 0.5)

        for idx in range(len(tracked)):
            tid = int(tracked.tracker_id[idx])
            x1, y1, x2, y2 = map(int, tracked.xyxy[idx])

            label = "Normal"
            f_telem = fall_telemetry.get(tid)
            if f_telem and f_telem['is_fall']:
                label = "MEDICAL EMERGENCY"
            elif x3d_flagging_violence and tid in person_states:
                label = "Violence! (X3D)"
                red_boxes.add(tid)
            else:
                for pair, v_telem in violence_telemetry.items():
                    if tid in pair and v_telem['is_violence']:
                        label = "Violence? (heuristic only)"
                        break

            color = (0, 0, 255) if tid in red_boxes else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.putText(frame, f"Headcount: {len(tracked)}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        x3d_color = (0, 0, 255) if x3d_label == "Fight" else (0, 255, 0)
        cv2.putText(frame, f"X3D: {x3d_label} ({x3d_confidence:.2f})", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, x3d_color, 2)

        out_writer.write(frame)
        cv2.imshow("Video Inference", frame)
       
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out_writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="Path to video file")
    args = parser.parse_args()

    print("[*] Loading RTMPose model...")
    pose_model = create_pose_pipeline()
    print("[*] Loading X3D-S violence detector...")
    x3d_detector = X3DViolenceDetector()

    run_on_video(args.path, pose_model, x3d_detector)