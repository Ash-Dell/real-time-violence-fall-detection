import os
import glob
import csv
import time
from collections import defaultdict, deque

import cv2
import numpy as np

from action_math import detect_advanced_violence, extract_keypoint
from behavior_helpers import should_flag_violence, update_track_state
from vision_models import create_pose_pipeline, match_keypoints_to_bbox
from x3d_adapter import X3DViolenceDetector, ClipBuffer
import supervision as sv

HISTORY_LENGTH = 15
PROXIMITY_THRESHOLD = 500
RESULTS_CSV = "eval_results.csv"
BATCH_SIZE = 25
COOLDOWN_SECONDS = 60
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

TEST_DIRS = {
    "Fight": "./data/RWF-2000/val/Fight",
    "NonFight": "./data/RWF-2000/val/NonFight",
}


def update_person_state(tracker_id, keypoints, state):
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


def run_heuristic_and_x3d_on_clip(video_path, pose_model, x3d_detector):
    byte_tracker = sv.ByteTrack(
        track_activation_threshold=0.55,
        lost_track_buffer=10,
        minimum_matching_threshold=0.8,
        frame_rate=30,
    )
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False, "N/A", 0.0

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
        'bbox_ar': 1.0,
        'bbox_height': 1.0,
    })

    heuristic_flagged = False
    clip_buffer = ClipBuffer(maxlen=32)
    track_states = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        pose_results = pose_model.predict(frame, confidence=POSE_DETECTION_CONFIDENCE, det_thr=POSE_DETECTION_CONFIDENCE)
        detections = pose_results.to_supervision()
        tracked = byte_tracker.update_with_detections(detections)

        active_ids = set()
        for idx in range(len(tracked)):
            tracker_id = int(tracked.tracker_id[idx])
            active_ids.add(tracker_id)
            track_state = track_states.setdefault(tracker_id, {})
            bbox = tracked.xyxy[idx]
            confidence_value = float(tracked.confidence[idx]) if tracked.confidence is not None else 0.0
            smoothed_bbox, _, _ = update_track_state(track_state, bbox, confidence_value)
            tracked.xyxy[idx] = smoothed_bbox

            width = max(1e-5, smoothed_bbox[2] - smoothed_bbox[0])
            height = max(1e-5, smoothed_bbox[3] - smoothed_bbox[1])
            person_states[tracker_id]['bbox_ar'] = width / height
            person_states[tracker_id]['bbox_height'] = height

            kps = match_keypoints_to_bbox(pose_results, smoothed_bbox)
            if kps is not None:
                update_person_state(tracker_id, kps, person_states[tracker_id])

        stale = [tid for tid in person_states if tid not in active_ids]
        for tid in stale:
            del person_states[tid]

        ids = list(person_states.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                cx_a, cy_a = person_states[ids[i]]['centroid']
                cx_b, cy_b = person_states[ids[j]]['centroid']
                dist = np.sqrt((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2)
                if dist < PROXIMITY_THRESHOLD:
                    hist_a = {k: list(v) for k, v in person_states[ids[i]].items() if isinstance(v, deque)}
                    hist_b = {k: list(v) for k, v in person_states[ids[j]].items() if isinstance(v, deque)}

                    v_a_b, telem_a_b = detect_advanced_violence(hist_a, hist_b, person_states[ids[i]]['bbox_height'])
                    v_b_a, telem_b_a = detect_advanced_violence(hist_b, hist_a, person_states[ids[j]]['bbox_height'])

                    telemetry = {
                        'score': max(telem_a_b['score'], telem_b_a['score']),
                        'relative_velocity': max(telem_a_b['relative_velocity'], telem_b_a['relative_velocity']),
                        'relative_acceleration': max(telem_a_b['relative_acceleration'], telem_b_a['relative_acceleration']),
                        'entropy': max(telem_a_b['entropy'], telem_b_a['entropy']),
                        'distance_norm': dist / max(person_states[ids[i]]['bbox_height'], 1e-5),
                    }
                    heuristic_flagged = should_flag_violence(telemetry, person_states[ids[i]], person_states[ids[j]]) or (v_a_b and v_b_a)

        clip_buffer.add(frame)

    cap.release()

    x3d_label, x3d_confidence = "N/A", 0.0
    if clip_buffer.is_ready(min_frames=13):
        x3d_label, x3d_confidence = x3d_detector.predict(clip_buffer.get_clip())

    return heuristic_flagged, x3d_label, x3d_confidence


def load_processed_paths():
    """Read already-processed clip paths from the CSV so we can skip them."""
    if not os.path.exists(RESULTS_CSV):
        return set()
    processed = set()
    with open(RESULTS_CSV, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            processed.add(row["path"])
    return processed


def append_result(row):
    file_exists = os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "true_label", "heuristic_pred", "x3d_pred", "x3d_conf"])
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def print_report():
    if not os.path.exists(RESULTS_CSV):
        print("No results yet.")
        return
    with open(RESULTS_CSV, "r", newline="") as f:
        rows = list(csv.DictReader(f))

    total = len(rows)
    if total == 0:
        print("No results yet.")
        return

    heuristic_correct = sum(1 for r in rows if r["heuristic_pred"] == r["true_label"])
    x3d_correct = sum(1 for r in rows if r["x3d_pred"] == r["true_label"])

    # Fusion OR: flag "Fight" if EITHER signal says Fight
    fusion_or_correct = 0
    for r in rows:
        fused_pred = "Fight" if (r["heuristic_pred"] == "Fight" or r["x3d_pred"] == "Fight") else "NonFight"
        if fused_pred == r["true_label"]:
            fusion_or_correct += 1

    # Fusion AND: flag "Fight" only if BOTH signals agree it's Fight
    fusion_and_correct = 0
    for r in rows:
        fused_pred = "Fight" if (r["heuristic_pred"] == "Fight" and r["x3d_pred"] == "Fight") else "NonFight"
        if fused_pred == r["true_label"]:
            fusion_and_correct += 1

    print("\n===== CURRENT REPORT =====")
    print(f"Total clips evaluated so far: {total}")
    print(f"Heuristic accuracy:    {heuristic_correct}/{total} = {heuristic_correct/total:.4f}")
    print(f"X3D-S accuracy:        {x3d_correct}/{total} = {x3d_correct/total:.4f}")
    #print(f"Fusion (OR) accuracy:  {fusion_or_correct}/{total} = {fusion_or_correct/total:.4f}")
    #print(f"Fusion (AND) accuracy: {fusion_and_correct}/{total} = {fusion_and_correct/total:.4f}")

def main():
    print("[*] Loading models...")
    pose_model = create_pose_pipeline()
    x3d_detector = X3DViolenceDetector()

    already_done = load_processed_paths()
    print(f"[*] Resuming — {len(already_done)} clips already processed previously.")

    all_clips = []
    for true_label, folder in TEST_DIRS.items():
        for vp in sorted(glob.glob(os.path.join(folder, "*.avi")))[:40]:
            if vp not in already_done:
                all_clips.append((true_label, vp))

    print(f"[*] {len(all_clips)} clips remaining to process.")

    for count, (true_label, vp) in enumerate(all_clips, start=1):
        heuristic_flagged, x3d_label, x3d_conf = run_heuristic_and_x3d_on_clip(vp, pose_model, x3d_detector)
        heuristic_pred = "Fight" if heuristic_flagged else "NonFight"

        append_result({
            "path": vp,
            "true_label": true_label,
            "heuristic_pred": heuristic_pred,
            "x3d_pred": x3d_label,
            "x3d_conf": x3d_conf,
        })

        print(f"  [{count}/{len(all_clips)}] {os.path.basename(vp)} | true={true_label} | "
              f"heuristic={heuristic_pred} | x3d={x3d_label} ({x3d_conf:.2f})")

        if count % BATCH_SIZE == 0:
            print(f"[*] Batch of {BATCH_SIZE} done — cooling down for {COOLDOWN_SECONDS}s...")
            time.sleep(COOLDOWN_SECONDS)

    print_report()


if __name__ == "__main__":
    main()