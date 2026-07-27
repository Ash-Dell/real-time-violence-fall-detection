import argparse
from collections import defaultdict, deque

import cv2
import numpy as np
import supervision as sv

from action_math import detect_fall, extract_keypoint
from vision_models import create_pose_pipeline, match_keypoints_to_bbox
from x3d_adapter import X3DViolenceDetector, ClipBuffer
from pathlib import Path
from datetime import datetime
from interaction_manager.interaction_manager import InteractionManager
from interaction_manager.person_node import PersonNode
from tracking_utils import (
    update_track_state,
    mark_track_lost,
)

HISTORY_LENGTH = 15
PROXIMITY_THRESHOLD = 500
X3D_MIN_PEOPLE = 2
X3D_CLIP_LEN = 32
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


"""def check_proximity_trigger(person_states):
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
"""

"""def check_motion_intensity(person_states):
    
    Returns True if there's significant motion among tracked people.
    This adds an additional gate to ensure X3D only runs when people
    are actually moving (not just standing close together).
    
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
"""

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

    byte_tracker = sv.ByteTrack()
    """track_activation_threshold=0.55,
        lost_track_buffer=10,
        minimum_matching_threshold=0.8,
        frame_rate=30,
    )
"""
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

    clip_buffer = ClipBuffer(maxlen=X3D_CLIP_LEN)
    x3d_label, x3d_confidence = "N/A", 0.0
    x3d_confidence_history = deque(maxlen=5)  # For confidence smoothing
    track_states = {}
    interaction_manager = InteractionManager()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        pose_results = pose_model.predict(frame, confidence=POSE_DETECTION_CONFIDENCE, det_thr=POSE_DETECTION_CONFIDENCE)
        detections = pose_results.to_supervision()
        print(f"Pose detections: {len(detections)}")
        print("Detection confidences:", detections.confidence)
        print("Class IDs:", detections.class_id)
        print("Boxes:", len(detections.xyxy))
        print("Confidences:", len(detections.confidence))
        tracked = byte_tracker.update_with_detections(detections)
        print(detections)
        print(f"Tracked people: {len(tracked)}")

        tracked_people = []
        active_ids = set()
        fall_telemetry = {}
        violence_telemetry = {}
        red_boxes = set()

        for idx in range(len(tracked)):
            tracker_id = int(tracked.tracker_id[idx])
            active_ids.add(tracker_id)
            track_state = track_states.setdefault(tracker_id, {})
            bbox = tracked.xyxy[idx]
            confidence_value = float(tracked.confidence[idx]) if tracked.confidence is not None else 0.0
            smoothed_bbox, track_confidence, _ = update_track_state(track_state, bbox, confidence_value)
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
                tracked_people.append({
                "tracker_id": tracker_id,
                "bbox": smoothed_bbox.copy(),
                "keypoints": kps.copy(),
                "confidence": track_confidence,
                "frame_id": int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
            })

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
        interaction_tracks = interaction_manager.update(
            tracked_people,
            int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
            frame
        )
        if frame_id % 30 == 0:
            print(f"\nFrame {frame_id}")
            print(f"Interaction tracks: {len(interaction_tracks)}")

            for track in interaction_tracks:
                print(
                    f"ID={track.interaction_id}, "
                    f"members={[p.tracker_id for p in track.members]}, "
                    f"state={track.state}, "
                    f"clip={len(track.clip_buffer.frames)}"
                )



        for interaction in interaction_tracks:

            if interaction.is_ready():

                print(f"Interaction {interaction.interaction_id} is READY")

                try:
                    clip = interaction.get_clip()

                    prediction, confidence = x3d_detector.predict(clip)

                    print(f"Prediction: {prediction} ({confidence:.3f})")

                    interaction.store_prediction(
                        prediction,
                        confidence
                    )
                except Exception as e:
                    print(f"X3D ERROR: {e}")





        stale = [tid for tid in list(person_states.keys()) if tid not in active_ids]
        for tid in stale:
            state = track_states.get(tid)
            if state is None:
                del person_states[tid]
                continue
            if not mark_track_lost(state):
                del person_states[tid]
                track_states.pop(tid, None)

        """ids = list(person_states.keys())
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
                    telemetry = {
                        'score': max(telem_a_b['score'], telem_b_a['score']),
                        'relative_velocity': max(telem_a_b['relative_velocity'], telem_b_a['relative_velocity']),
                        'relative_acceleration': max(telem_a_b['relative_acceleration'], telem_b_a['relative_acceleration']),
                        'entropy': max(telem_a_b['entropy'], telem_b_a['entropy']),
                        'distance_norm': np.sqrt((person_states[id_a]['centroid'][0] - person_states[id_b]['centroid'][0]) ** 2 + (person_states[id_a]['centroid'][1] - person_states[id_b]['centroid'][1]) ** 2) / max(person_states[id_a]['bbox_height'], 1e-5),
                    }
                    is_violence = should_flag_violence(telemetry, person_states[id_a], person_states[id_b]) or (v_a_b and v_b_a)
                    if is_violence:
                        person_states[id_a]['violence_streak'] += 1
                        person_states[id_b]['violence_streak'] += 1
                    else:
                        person_states[id_a]['violence_streak'] = max(0, person_states[id_a]['violence_streak'] - 1)
                        person_states[id_b]['violence_streak'] = max(0, person_states[id_b]['violence_streak'] - 1)

                    violence_telemetry[(id_a, id_b)] = {
                        'score': telemetry['score'],
                        'is_violence': is_violence
                    }
                 """   
                    # Disable heuristic-based red boxes - rely primarily on X3D
                    # Heuristic is too sensitive for normal settings like classrooms
                    # if person_states[id_a]['violence_streak'] >= 5 or person_states[id_b]['violence_streak'] >= 5:
                    #     red_boxes.update([id_a, id_b])

        """ trigger_active = check_proximity_trigger(person_states)
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
            clip_buffer = ClipBuffer(maxlen=X3D_CLIP_LEN)
            x3d_label, x3d_confidence = "N/A", 0.0
            x3d_confidence_history.clear()
        """
        #x3d_flagging_violence = (x3d_label == "Fight" and x3d_confidence >= 0.25)  # Lowered threshold for higher sensitivity

        for idx in range(len(tracked)):
            tid = int(tracked.tracker_id[idx])
            x1, y1, x2, y2 = map(int, tracked.xyxy[idx])

            label = "Normal"
            for interaction in interaction_tracks:

                if (
                    interaction.contains_person(tid)
                    and interaction.is_violent()
                ):
                    label = "Violence! (X3D)"
                    red_boxes.add(tid)
                    break


            
            """else:
                # Re-enable heuristic with high-confidence gating as secondary system
                for pair, v_telem in violence_telemetry.items():
                    if tid in pair and v_telem['is_violence'] and v_telem.get('score', 0) >= 3.0:
                        label = "Violence? (heuristic high-conf)"
                        red_boxes.add(tid)
                        break
            """
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