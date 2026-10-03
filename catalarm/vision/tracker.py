"""Temporal voting so the reported state does not flicker frame to frame."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from catalarm.config import Thresholds
from catalarm.vision.matcher import MATCH, NO_CAT, UNCONFIRMED, MatchResult

NO_MOTION = "no_motion"


@dataclass
class TrackState:
    state: str  # cat id, "unconfirmed", "no_cat" or "no_motion"
    cat_id: str | None
    confidence: float
    changed: bool

    @property
    def is_cat(self) -> bool:
        return self.cat_id is not None and self.state == self.cat_id


class Tracker:
    def __init__(self, th: Thresholds):
        self.th = th
        self._window: deque[tuple[float, MatchResult]] = deque(maxlen=th.vote_n)
        self._state = NO_MOTION
        self._cat: str | None = None
        self._confidence = 0.0
        self._last_cat_seen = 0.0
        self._last_motion = float("-inf")

    def update_thresholds(self, th: Thresholds) -> None:
        self.th = th
        self._window = deque(self._window, maxlen=th.vote_n)

    def update(self, now: float, motion: bool, result: MatchResult | None = None) -> TrackState:
        """Feed one observation. ``result`` is None when nothing was analysed."""
        if motion:
            self._last_motion = now
        if result is not None:
            self._window.append((now, result))
        # observations older than the hold time do not vote any more
        while self._window and now - self._window[0][0] > max(self.th.hold_s, 1.0):
            self._window.popleft()

        votes: dict[str, list[MatchResult]] = {}
        for _, r in self._window:
            if r.state == MATCH and r.cat_id:
                votes.setdefault(r.cat_id, []).append(r)
        winner = max(votes.items(), key=lambda kv: len(kv[1]), default=None)

        cat, confidence = None, 0.0
        if winner and len(winner[1]) >= min(self.th.vote_k, self.th.vote_n):
            cat = winner[0]
            confidence = max(r.confidence for r in winner[1])
            self._last_cat_seen = now
            state = cat
        elif self._cat and now - self._last_cat_seen <= self.th.hold_s:
            cat, confidence, state = self._cat, self._confidence, self._cat
        elif any(r.state == UNCONFIRMED for _, r in self._window):
            # only weak matches are "unconfirmed"; strong ones still collecting votes are not,
            # otherwise every visit would start with an unconfirmed -> cat flap in HA
            state = UNCONFIRMED
        elif now - self._last_motion <= self.th.hold_s:
            state = NO_CAT
        else:
            state = NO_MOTION

        changed = state != self._state
        self._state, self._cat, self._confidence = state, cat, confidence
        return TrackState(state, cat, confidence, changed)


__all__ = ["Tracker", "TrackState", "NO_MOTION", "NO_CAT", "UNCONFIRMED"]
