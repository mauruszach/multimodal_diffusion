#!/usr/bin/env python3
"""
mask_schedule.py — randomly choose targets per batch: {'video'} or {'audio'}.
"""

from __future__ import annotations
import random


class Any2AnySchedule:
    """
    Simple Bernoulli selector using configured probabilities.
    probs: dict(video=0.5, audio=0.5)
    """
    def __init__(self, probs: dict[str, float]):
        pv = float(probs.get("video", 0.5))
        pa = float(probs.get("audio", 0.5))
        total = pv + pa
        if total <= 0:
            raise ValueError("Sum of probabilities must be > 0")
        self.pv = pv / total
        self.pa = pa / total

    def sample_target(self) -> str:
        r = random.random()
        return "video" if r < self.pv else "audio"
