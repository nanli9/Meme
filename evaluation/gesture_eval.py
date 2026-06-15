"""Shared helpers for evaluating gesture rules against labeled datasets (Milestone 2).

Maps external dataset labels to our gesture vocabulary and tallies a confusion matrix.
Used by `scripts/eval_hagrid.py` (HaGRID images, finger gestures) and
`scripts/eval_ntu.py` (NTU RGB+D skeletons, body/motion gestures).
"""

from __future__ import annotations

from collections import defaultdict

# HaGRID class -> our gesture (only the ones that map cleanly; others -> None/ignore).
HAGRID_TO_GESTURE: dict[str, str] = {
    "like": "thumbs_up",
    "one": "pointing",
    "palm": "open_palm",
    "stop": "open_palm",
}

# NTU RGB+D 60 action code (Axxx in the filename) -> our gesture.
NTU_CODE_TO_GESTURE: dict[int, str] = {
    23: "wave",          # hand waving
    10: "clap",          # clapping
    39: "clap",          # put palms together
    34: "clap",          # rub two hands together
    40: "arms_crossed",  # cross hands in front (say stop)
    31: "pointing",      # pointing to something with finger
    22: "hype",          # cheer up
}


class Confusion:
    """Accumulates expected->predicted counts and prints accuracy + a compact matrix."""

    def __init__(self) -> None:
        self.counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.total = 0
        self.correct = 0

    def add(self, expected: str, predicted: str) -> None:
        self.counts[expected][predicted] += 1
        self.total += 1
        self.correct += int(expected == predicted)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    def report(self) -> str:
        if not self.total:
            return "(no samples evaluated)"
        lines = [f"Overall accuracy: {self.accuracy:.1%}  (n={self.total})", ""]
        for expected in sorted(self.counts):
            row = self.counts[expected]
            n = sum(row.values())
            acc = row.get(expected, 0) / n if n else 0.0
            breakdown = ", ".join(f"{pred}:{cnt}" for pred, cnt in
                                  sorted(row.items(), key=lambda kv: kv[1], reverse=True))
            lines.append(f"  {expected:<13} acc {acc:5.1%}  (n={n:4d})  -> {breakdown}")
        return "\n".join(lines)
