from collections import defaultdict
from typing import Dict, List, Set

from .person_node import PersonNode


class AssociationGraph:
    """
    Stateless graph built every frame.
    Nodes = PersonNode
    Edges = Spatial association
    """

    def __init__(self):
        self.nodes: Dict[int, PersonNode] = {}
        self.edges: Dict[int, Set[int]] = defaultdict(set)

    def clear(self):
        self.nodes.clear()
        self.edges.clear()

    def add_node(self, person: PersonNode):
        self.nodes[person.tracker_id] = person

    def add_edge(self, person_a: PersonNode, person_b: PersonNode):
        self.edges[person_a.tracker_id].add(person_b.tracker_id)
        self.edges[person_b.tracker_id].add(person_a.tracker_id)

    def connected_components(self) -> List[Set[PersonNode]]:
        """
        Returns groups of associated people.
        """
        visited = set()
        components = []

        for node_id in self.nodes:

            if node_id in visited:
                continue

            stack = [node_id]
            component = []

            while stack:

                current = stack.pop()

                if current in visited:
                    continue

                visited.add(current)
                component.append(self.nodes[current])

                for neighbor in self.edges[current]:
                    if neighbor not in visited:
                        stack.append(neighbor)

            components.append(component)

        return components