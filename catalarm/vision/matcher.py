"""Match an embedding against the reference gallery (replaces CompareUnknownTensorToDB).

Every cat gets one score: the mean cosine distance of its ``knn_k`` closest references.
The same score is used for ranking and for the thresholds (the old code ranked by average
distance but thresholded the minimum).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from catalarm.config import Thresholds

MATCH = "match"
UNCONFIRMED = "unconfirmed"
NO_CAT = "no_cat"


@dataclass
class MatchResult:
    state: str  # MATCH / UNCONFIRMED / NO_CAT
    cat_id: str | None  # best cat (also set for UNCONFIRMED)
    distance: float  # score of the best cat
    margin: float  # score gap to the second best cat
    distances: dict[str, float] = field(default_factory=dict)  # score per cat, ascending

    @property
    def confidence(self) -> float:
        return float(max(0.0, 1.0 - self.distance))

    def to_dict(self) -> dict:
        return {
            "state": self.state,
            "cat_id": self.cat_id,
            "distance": round(self.distance, 4),
            "margin": round(self.margin, 4),
            "confidence": round(self.confidence, 4),
            "distances": {k: round(v, 4) for k, v in self.distances.items()},
        }


def cat_scores(query: np.ndarray, refs: np.ndarray, labels: list[str], k: int) -> dict[str, float]:
    if len(labels) == 0:
        return {}
    dist = 1.0 - refs @ query
    lab = np.asarray(labels)
    scores = {}
    for cat in dict.fromkeys(labels):
        d = np.sort(dist[lab == cat])
        scores[cat] = float(d[: min(k, len(d))].mean())
    return dict(sorted(scores.items(), key=lambda kv: kv[1]))


def match(query: np.ndarray, refs: np.ndarray, labels: list[str], th: Thresholds) -> MatchResult:
    scores = cat_scores(query, refs, labels, th.knn_k)
    if not scores:
        return MatchResult(NO_CAT, None, 2.0, 0.0, {})
    ranked = list(scores.items())
    best_cat, best = ranked[0]
    margin = ranked[1][1] - best if len(ranked) > 1 else 1.0
    if best <= th.match and margin >= th.margin:
        state = MATCH
    elif best <= th.unconfirmed:
        state = UNCONFIRMED
    else:
        state = NO_CAT
    return MatchResult(state, best_cat, best, margin, scores)
