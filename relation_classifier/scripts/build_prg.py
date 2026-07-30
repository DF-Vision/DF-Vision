#!/usr/bin/env python3
"""
build_prg.py
------------
Minimal Physical Relation Graph: nodes = detected objects, edges =
predicted Contact/Support/Separate relations with confidence.
No visualization — just enough structure for compute_prg_edge_error.
"""

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class PRGEdge:
    object_a: str
    object_b: str
    relation: str        # "Contact" | "Support" | "Separate"
    confidence: float    # [0, 1]


class PRG:
    def __init__(self):
        self.nodes: List[str] = []
        self.edges: List[PRGEdge] = []

    def add_node(self, name: str) -> None:
        if name not in self.nodes:
            self.nodes.append(name)

    def add_edge(self, object_a: str, object_b: str, relation: str, confidence: float) -> None:
        self.add_node(object_a)
        self.add_node(object_b)
        self.edges.append(PRGEdge(object_a, object_b, relation, confidence))

    def edge(self, object_a: str, object_b: str) -> PRGEdge | None:
        for e in self.edges:
            if {e.object_a, e.object_b} == {object_a, object_b}:
                return e
        return None

    def to_tuples(self) -> List[Tuple[str, str, str, float]]:
        return [(e.object_a, e.object_b, e.relation, e.confidence) for e in self.edges]


def build_prg(detected_objects: List[str], predictions: dict) -> PRG:
    """
    predictions: {(obj_a, obj_b): (relation_str, confidence_float)}
    keyed however your classifier-wiring code already produces pairs.
    """
    prg = PRG()
    for name in detected_objects:
        prg.add_node(name)
    for (a, b), (relation, confidence) in predictions.items():
        prg.add_edge(a, b, relation, confidence)
    return prg