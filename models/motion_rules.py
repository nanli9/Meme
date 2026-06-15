"""Rule-based gestures -> broad reaction categories (Milestone 2).

Hand-tuned rules over `SkeletonFeatures`. Output is a WEAK signal: a broad *visible
reaction* category (e.g. `shrug`, `hype`, `facepalm`), never an emotion claim and never a
meme ID (CLAUDE.md). Scores for every category are returned so a later fusion ranker
(M7) can use the full distribution, not just the top pick.

Thresholds are in torso-normalized "shoulder-width units" and are deliberately
hand-tuned — expect to refine them against real saved windows via
`scripts/label_session.py` (CLAUDE.md: hand-tune first, learn later).
"""

from __future__ import annotations

import math

from pydantic import BaseModel, Field

from features.skeleton_features import SkeletonFeatures, compute_features

CONFIDENCE_FLOOR = 0.45  # below this, we report `neutral`

GESTURES = (
    "shrug", "hype", "arms_crossed", "arms_wide",
    "facepalm", "thinking",
    "wave", "clap",
    "thumbs_up", "pointing", "middle_finger", "pinky", "peace", "open_palm",
)


_FINGER_ORDER = ("thumb", "index", "middle", "ring", "pinky")


class GestureEstimate(BaseModel):
    top: str = "neutral"
    confidence: float = 0.0
    intensity: float = 0.0
    valid_ratio: float = 0.0
    scores: dict[str, float] = Field(default_factory=dict)
    fingers: list[str] = Field(default_factory=list)  # extended fingers of the best hand


# --- soft threshold helpers (NaN-safe -> 0) --------------------------------
def _ramp(x: float, lo: float, hi: float) -> float:
    """0 at/below lo, 1 at/above hi, linear between. NaN -> 0."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return 0.0
    if hi == lo:
        return 1.0 if x >= hi else 0.0
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))


def _inv_ramp(x: float, lo: float, hi: float) -> float:
    """1 at/below lo, 0 at/above hi. NaN -> 0 (treated as 'not satisfied')."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return 0.0
    return 1.0 - _ramp(x, lo, hi)


def _depth_ok(dz: float) -> float:
    """Depth-consistency gate for hand-at-face: 1 when the wrist and nose are at a similar
    pose-frame depth, ramping to 0 as the gap grows. Unknown depth (NaN) -> 1 (permissive,
    so missing/unreliable z never kills a face gesture)."""
    if dz is None or (isinstance(dz, float) and math.isnan(dz)):
        return 1.0
    return _inv_ramp(dz, 0.2, 0.5)


def _band(x: float, center: float, half: float) -> float:
    """1 at center, ramping to 0 at +/- half. NaN -> 0."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return 0.0
    return max(0.0, 1.0 - abs(x - center) / half)


# --- the rules -------------------------------------------------------------
def _score_gestures(f: SkeletonFeatures) -> dict[str, float]:
    s: dict[str, float] = {}

    # Arms raised overhead.
    s["hype"] = min(_ramp(f.wrist_above_shoulder_l, 0.1, 0.6),
                    _ramp(f.wrist_above_shoulder_r, 0.1, 0.6))

    # Arms extended horizontally (T-pose): far out, ~shoulder height, elbows straight.
    s["arms_wide"] = min(
        _ramp(f.wrist_lateral_l, 1.0, 1.6), _ramp(f.wrist_lateral_r, 1.0, 1.6),
        _ramp(f.elbow_angle_l, 130, 165), _ramp(f.elbow_angle_r, 130, 165),
        _band(f.wrist_above_shoulder_l, 0.0, 0.6), _band(f.wrist_above_shoulder_r, 0.0, 0.6),
    )

    # Shrug: arms out moderately, elbows bent, around/below shoulder (the not-T-pose).
    s["shrug"] = min(
        _ramp(f.wrist_lateral_l, 0.5, 0.9), _ramp(f.wrist_lateral_r, 0.5, 0.9),
        _inv_ramp(f.elbow_angle_l, 100, 150), _inv_ramp(f.elbow_angle_r, 100, 150),
        _inv_ramp(f.wrist_lateral_l, 1.2, 1.6), _inv_ramp(f.wrist_lateral_r, 1.2, 1.6),
        _band(f.wrist_above_shoulder_l, -0.1, 0.7), _band(f.wrist_above_shoulder_r, -0.1, 0.7),
    )

    s["arms_crossed"] = _ramp(f.wrists_crossed, 0.3, 0.7)

    # Hand-to-face: nearest STILL hand to the nose. facepalm = at/above nose; thinking =
    # below (chin). A moving hand near the face is a wave, not thinking (the stillness gate).
    face, think = [], []
    for d_nose, h_height, spd, osc, dz in (
        (f.wrist_to_nose_l, f.hand_face_height_l, f.wrist_speed_l, f.wrist_x_oscillations_l, f.wrist_nose_dz_l),
        (f.wrist_to_nose_r, f.hand_face_height_r, f.wrist_speed_r, f.wrist_x_oscillations_r, f.wrist_nose_dz_r),
    ):
        still = _inv_ramp(spd, 0.05, 0.12) * _inv_ramp(osc, 1, 3)
        depth = _depth_ok(dz)  # the hand must be at the face in DEPTH too, not just on-screen
        face.append(_inv_ramp(d_nose, 0.3, 0.65) * _ramp(h_height, -0.15, 0.2) * still * depth)
        think.append(_inv_ramp(d_nose, 0.45, 0.85) * _band(h_height, -0.4, 0.35) * still * depth)
    s["facepalm"] = max(face)
    s["thinking"] = max(think)

    # Wave: a raised hand oscillating laterally (works even when the hand is near the face).
    s["wave"] = max(
        _ramp(f.wrist_x_oscillations_l, 1.5, 3) * _ramp(f.wrist_above_shoulder_l, -0.3, 0.3),
        _ramp(f.wrist_x_oscillations_r, 1.5, 3) * _ramp(f.wrist_above_shoulder_r, -0.3, 0.3),
    )

    # Clap: wrists repeatedly converge in front of the body.
    s["clap"] = _ramp(f.inter_wrist_convergences, 1.5, 3.0) * _inv_ramp(f.inter_wrist_dx, 0.9, 1.5)

    # Finger gestures on whichever hand is most present. Each single-finger gesture needs
    # ITS finger extended and the others curled (suppression uses max(), so a peace sign
    # doesn't read as pointing). Only `pointing` is face-aware: an index finger AT your own
    # head (fingertip near the nose) reads as thinking, not pointing.
    hand = f.best_hand()
    pf = _ramp(hand.present, 0.4, 0.7)
    ext = hand.extended

    def only(target: str, others: tuple[str, ...]) -> float:
        return _ramp(ext.get(target, 0.0), 0.4, 0.7) * _inv_ramp(max(ext.get(o, 0.0) for o in others), 0.4, 0.7)

    _dzs = [d for d in (f.wrist_nose_dz_l, f.wrist_nose_dz_r) if not math.isnan(d)]
    at_head = _inv_ramp(hand.to_nose, 0.15, 0.28) * _depth_ok(min(_dzs) if _dzs else float("nan"))
    point_raw = pf * only("index", ("middle", "ring", "pinky"))
    s["pointing"] = point_raw * (1.0 - at_head)
    s["thinking"] = max(s["thinking"], point_raw * at_head)  # index at own head -> thinking
    s["middle_finger"] = pf * only("middle", ("index", "ring", "pinky"))
    s["pinky"] = pf * only("pinky", ("index", "middle", "ring"))
    s["peace"] = pf * min(_ramp(ext.get("index", 0.0), 0.4, 0.7), _ramp(ext.get("middle", 0.0), 0.4, 0.7)) \
        * _inv_ramp(max(ext.get("ring", 0.0), ext.get("pinky", 0.0)), 0.4, 0.7)
    s["thumbs_up"] = pf * _ramp(ext.get("thumb", 0.0), 0.4, 0.7) * _ramp(hand.thumb_up, 0.3, 0.6) \
        * _inv_ramp(max(ext.get(o, 0.0) for o in ("index", "middle", "ring", "pinky")), 0.4, 0.7)
    s["open_palm"] = pf * _ramp(hand.extended_count, 2.5, 4.0) * _ramp(hand.spread, 0.35, 0.8)

    return {k: float(round(v, 4)) for k, v in s.items()}


def classify(features: SkeletonFeatures) -> GestureEstimate:
    """Map features to a ranked gesture estimate."""
    scores = _score_gestures(features)
    hand = features.best_hand()
    fingers = ([name for name in _FINGER_ORDER if hand.extended.get(name, 0.0) >= 0.5]
               if hand.present >= 0.4 else [])
    top = max(scores, key=scores.get) if scores else "neutral"
    top_score = scores.get(top, 0.0)
    if features.valid_ratio < 0.2 or top_score < CONFIDENCE_FLOOR:
        top = "neutral"
    return GestureEstimate(top=top, confidence=top_score, intensity=top_score,
                           valid_ratio=features.valid_ratio, scores=scores, fingers=fingers)


def estimate_gesture(window, *, visibility_threshold: float = 0.5,
                     assume_normalized: bool = False) -> GestureEstimate:
    """Convenience: features + classification from a `[T, J, 4]` window."""
    f = compute_features(window, visibility_threshold=visibility_threshold,
                         assume_normalized=assume_normalized)
    return classify(f)


class GestureStabilizer:
    """Debounce + throttle gesture labels for display (CLAUDE.md: <=1 update / 1-2s,
    no per-frame flicker). Switches the shown label only after a new candidate persists
    for `persist` evaluations AND at least `min_interval` seconds since the last switch."""

    def __init__(self, *, min_interval: float = 1.0, persist: int = 2) -> None:
        self.min_interval = min_interval
        self.persist = persist
        self.current = "neutral"
        self._cand = "neutral"
        self._cand_count = 0
        self._last_change = float("-inf")

    def update(self, estimate: GestureEstimate, now: float) -> str:
        top = estimate.top
        if top == self._cand:
            self._cand_count += 1
        else:
            self._cand, self._cand_count = top, 1
        if (self._cand != self.current
                and self._cand_count >= self.persist
                and (now - self._last_change) >= self.min_interval):
            self.current = self._cand
            self._last_change = now
        return self.current
