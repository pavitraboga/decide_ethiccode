"""
DecisionModel: one shared TF-IDF backbone, three calibrated heads.

Artifacts are produced by `python train.py`. Each request runs exactly one head.
"""

import os
from typing import Any, Dict, List, Tuple

import joblib

import features as F
from train import approve_X, grounded_X, route_X

QUEUES: List[str] = F.QUEUES


class DecisionModel:
    def __init__(self, artifacts_dir: str = "artifacts") -> None:
        b = joblib.load(os.path.join(artifacts_dir, "decide.joblib"))
        self.vec = b["vec"]
        self.route_head = b["route"]
        self.approve_head = b["approve"]
        self.grounded_head = b["grounded"]
        self.cuts = b["cuts"]
        self.tau = b["approve_tau"]

    def route(self, state: Dict[str, Any]) -> Tuple[str, Dict[str, float], float, bool]:
        p = self.route_head.predict_proba(route_X(self.vec, [state]))[0]
        probs = {q: float(p[i]) for i, q in enumerate(QUEUES)}
        value = max(probs, key=probs.get)
        conf = probs[value]
        return value, probs, conf, conf < self.cuts["route"]

    def approve(self, state: Dict[str, Any]) -> Tuple[bool, float, bool]:
        # Policy rule: nothing above the tier cap is ever approved in the data.
        if F.over_cap(state):
            return False, 0.0, False
        p = float(self.approve_head.predict_proba(approve_X(self.vec, [state]))[0, 1])
        return p >= self.tau, p, max(p, 1 - p) < self.cuts["approve"]

    def grounded(self, state: Dict[str, Any]) -> Tuple[bool, float, bool]:
        p = float(self.grounded_head.predict_proba(grounded_X(self.vec, [state]))[0, 1])
        return p >= 0.5, p, max(p, 1 - p) < self.cuts["grounded"]
