import cv2
import numpy as np

from interaction_manager.interaction_track import InteractionTrack
from interaction_manager.person_node import PersonNode


class LocalInteractionProposer:

    def __init__(self):

        self.interactions = {}
        self.next_interaction_id = 0
        self.person_registry = {}
        self.pair_history = {}
        self.MIN_CONSECUTIVE_FRAMES = 5

        self.current_frame = None

        self.SCORE_THRESHOLD = 0.8

    def update(self, tracked_people, frame_id, frame):

        self.current_frame = frame

        self._update_person_registry(tracked_people)

        groups = self._generate_groups(
            list(self.person_registry.values())
        )

        self._update_tracks(groups, frame_id)

        return list(self.interactions.values())    

    def _distance(self, a, b):
        return np.linalg.norm(np.array(a) - np.array(b))   

    def _generate_groups(self, people):

        groups = []
        assigned = set()
        valid_pairs = set()

        for i in range(len(people)):

            for j in range(i + 1, len(people)):

                p1 = people[i]
                p2 = people[j]

                if (
                    p1.tracker_id in assigned or
                    p2.tracker_id in assigned
                ):
                    continue

                d = self._distance(
                    p1.centroid,
                    p2.centroid
                )
                if d > 150:
                    continue

                key = self._pair_key(
                    p1.tracker_id,
                    p2.tracker_id
                )

                valid_pairs.add(key)

                self.pair_history[key] = (
                    self.pair_history.get(key, 0) + 1
                )

                score = self._compute_pair_score(
                    d,
                    self.pair_history[key]
                )

                if score >= self.SCORE_THRESHOLD:
                    groups.append([p1, p2])
                    assigned.add(p1.tracker_id)
                    assigned.add(p2.tracker_id)

        stale = [
            k
            for k in self.pair_history
            if k not in valid_pairs
        ]

        for k in stale:
            del self.pair_history[k]

        print(
            "Generated groups:",
            [[p.tracker_id for p in g] for g in groups]
        )

        return groups

    def _compute_pair_score(self, distance, persistence):

        score = 0

        # Distance scoring
        if distance < 80:
            score += 0.6
        elif distance < 120:
            score += 0.4
        elif distance < 150:
            score += 0.2

        # Persistence scoring
        if persistence >= self.MIN_CONSECUTIVE_FRAMES:
            score += 0.4

        return score


    def _update_tracks(self, groups, frame_id):

        updated = set()

        occupied_people = set()

        for track in self.interactions.values():

            if track.missing_frames == 0:

                for member in track.members:
                    occupied_people.add(member.tracker_id)

        for members in groups:

            member_ids = set(
                p.tracker_id for p in members
            )

            best_track = None
            best_overlap = 0

            for track in self.interactions.values():

                overlap = len(
                    member_ids &
                    set(
                        p.tracker_id
                        for p in track.members
                    )
                )

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_track = track

            if best_track is not None and best_overlap >= max(1, len(member_ids) // 2):

                if any(
                    p.tracker_id in occupied_people
                    and p.tracker_id not in
                    [m.tracker_id for m in best_track.members]
                    for p in members
                ):
                    continue

                self._refresh_track(
                    best_track,
                    members,
                    frame_id
                )

                updated.add(best_track.interaction_id)

                for p in members:
                    occupied_people.add(p.tracker_id)

            else:

                if any(
                    p.tracker_id in occupied_people
                    for p in members
                ):
                    continue

                self._create_track(
                    members,
                    frame_id
                )

                for p in members:
                    occupied_people.add(p.tracker_id)

        stale = []

        for tid, track in self.interactions.items():

            if tid not in updated:

                track.missing_frames += 1

                if track.missing_frames > 3:
                    stale.append(tid)

            else:
                track.missing_frames = 0

        for tid in stale:
            del self.interactions[tid]    


    def _create_track(self, members, frame_id):

        track = InteractionTrack(
            self.next_interaction_id,
            members,
            frame_id
        )

        self.interactions[self.next_interaction_id] = track

        self.next_interaction_id += 1        


    def _refresh_track(self, track, members, frame_id):

        track.update(
            members,
            frame_id
        )

        x1, y1, x2, y2 = map(int, track.roi)

        h, w = self.current_frame.shape[:2]

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return

        crop = self.current_frame[y1:y2, x1:x2]

        track.add_frame(crop)    


    def _update_person_registry(self, tracked_people):

        active_ids = set()

        for person in tracked_people:

            tracker_id = person["tracker_id"]
            active_ids.add(tracker_id)

            if tracker_id in self.person_registry:

                self.person_registry[tracker_id].update(
                    bbox=person["bbox"],
                    keypoints=person["keypoints"],
                    confidence=person["confidence"],
                    frame_id=person["frame_id"],
                )

            else:

                self.person_registry[tracker_id] = PersonNode(
                    tracker_id=tracker_id,
                    bbox=person["bbox"],
                    keypoints=person["keypoints"],
                    confidence=person["confidence"],
                    frame_id=person["frame_id"],
                )

        stale = [
            tid
            for tid in self.person_registry
            if tid not in active_ids
        ]

        for tid in stale:
            del self.person_registry[tid]

    def _pair_key(self, id1, id2):
        return tuple(sorted((id1, id2)))      