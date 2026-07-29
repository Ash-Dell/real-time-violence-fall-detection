from typing import Dict

import numpy as np
import cv2

from .association_graph import AssociationGraph
from .interaction_track import InteractionTrack
from .person_node import PersonNode


class InteractionManager:

    def __init__(self):

        self.person_registry: Dict[int, PersonNode] = {}

        self.graph = AssociationGraph()

        self.interactions: Dict[int, InteractionTrack] = {}

        self.next_interaction_id = 0

    def update(self, tracked_people, frame_id, frame):
        """
        Main pipeline called once every frame.
        """
        self.current_frame = frame
        self._update_person_registry(tracked_people)

        self._build_association_graph()

        components = self.graph.connected_components()

        self._update_interactions(components, frame_id)

        self._cleanup()

        return list(self.interactions.values())

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
            tid for tid in self.person_registry
            if tid not in active_ids
        ]

        for tid in stale:
            del self.person_registry[tid]

    def _build_association_graph(self):

        self.graph.clear()

        for person in self.person_registry.values():
            self.graph.add_node(person)

        people = list(self.person_registry.values())

        PROXIMITY_THRESHOLD = 500

        for i in range(len(people)):
            for j in range(i + 1, len(people)):

                if np.linalg.norm(
                    np.array(people[i].centroid)
                    - np.array(people[j].centroid)
                ) < PROXIMITY_THRESHOLD:

                    distance = np.linalg.norm(
                        np.array(people[i].centroid) -
                        np.array(people[j].centroid)
                    )

                    if distance < PROXIMITY_THRESHOLD:
                        print(
                            f"{people[i].tracker_id} <-> {people[j].tracker_id} "
                            f"distance={distance:.1f}"
                        )
                        self.graph.add_edge(people[i], people[j])

    def _member_ids(self, members):
        return {person.tracker_id for person in members}

    def _update_interactions(self, components, frame_id):

        updated = {}

        for members in components:

            if len(members) < 2:
                continue

            member_ids = self._member_ids(members)

            matched = None
            best_overlap = 0

            for interaction in self.interactions.values():

                old_ids = self._member_ids(interaction.members)

                overlap = len(old_ids & member_ids)

                if overlap > best_overlap and overlap >= min(len(old_ids), len(member_ids)) * 0.6:
                    best_overlap = overlap
                    matched = interaction

            if matched is not None:

                matched.update(
                    members=members,
                    frame_id=frame_id,
                )
                print(
                    f"[MATCH] Interaction {matched.interaction_id} | "
                    f"Members: {sorted(member_ids)} | "
                    f"Overlap: {best_overlap}"
                )

                x1, y1, x2, y2 = map(int, matched.roi)

                padding = 40

                h, w = self.current_frame.shape[:2]

                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(w, x2 + padding)
                y2 = min(h, y2 + padding)

                crop = self.current_frame[y1:y2, x1:x2]

                matched.add_frame(crop)

                cv2.imwrite(
                    f"debug_roi/frame_{frame_id}_interaction_{matched.interaction_id}.jpg",
                    crop
                )

                updated[matched.interaction_id] = matched

            else:

                interaction = InteractionTrack(
                    interaction_id=self.next_interaction_id,
                    members=members,
                    frame_id=frame_id,
                )
                print(
                    f"[NEW] Interaction {self.next_interaction_id} | "
                    f"Members: {sorted(member_ids)}"
                )

                x1, y1, x2, y2 = map(int, interaction.roi)

                padding = 40

                h, w = self.current_frame.shape[:2]

                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(w, x2 + padding)
                y2 = min(h, y2 + padding)

                crop = self.current_frame[y1:y2, x1:x2]

                interaction.add_frame(crop)
                # interaction.add_frame(self.current_frame)
                cv2.imwrite(
                    f"debug_roi/frame_{frame_id}_interaction_{interaction.interaction_id}.jpg",
                    crop
                )

                updated[self.next_interaction_id] = interaction
                self.next_interaction_id += 1

        self.interactions = updated

    def _cleanup(self):
        pass