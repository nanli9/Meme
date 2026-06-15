"""Tests for the learned hand-gesture classifier's feature extraction + fallback."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.hand_classifier import (  # noqa: E402
    FEATURE_DIM,
    HandGestureClassifier,
    extract_hand_features,
    hand_window_features,
)


def _fake_hand(rotation=0.0, scale=1.0, tx=0.0, ty=0.0):
    """A simple deterministic 21-pt hand: fingers fanning upward from the wrist."""
    h = np.zeros((21, 4), dtype=np.float32)
    h[:, 3] = 1.0
    for i in range(21):
        h[i, 0] = (i % 5 - 2) * 0.1
        h[i, 1] = -0.1 - i * 0.04   # going "up" (image y is down)
    c, s = np.cos(rotation), np.sin(rotation)
    x, y = h[:, 0].copy(), h[:, 1].copy()
    h[:, 0] = (c * x - s * y) * scale + tx
    h[:, 1] = (s * x + c * y) * scale + ty
    return h


def test_feature_dim_and_finiteness():
    feat = extract_hand_features(_fake_hand())
    assert feat is not None
    assert feat.shape == (FEATURE_DIM,)
    assert np.isfinite(feat).all()


def test_translation_and_scale_invariance():
    a = extract_hand_features(_fake_hand())
    b = extract_hand_features(_fake_hand(scale=2.3, tx=0.5, ty=-0.4))
    assert a is not None and b is not None
    assert np.allclose(a, b, atol=1e-4)


def test_rotation_invariance():
    a = extract_hand_features(_fake_hand())
    b = extract_hand_features(_fake_hand(rotation=0.6))
    assert np.allclose(a, b, atol=1e-3)


def test_degenerate_hand_returns_none():
    assert extract_hand_features(np.zeros((21, 4), dtype=np.float32)) is None


def test_unavailable_classifier_is_graceful():
    clf = HandGestureClassifier(pipeline=None)
    assert not clf.is_available
    probs = clf.predict_proba_features(np.zeros((3, FEATURE_DIM), dtype=np.float32))
    assert set(probs) and all(v == 0.0 for v in probs.values())


def test_hand_window_features_shape():
    win = np.zeros((4, 55, 4), dtype=np.float32)
    win[:, 13:34, :] = _fake_hand()[None]  # put a hand in the left-hand slot, all frames
    feats = hand_window_features(win, 13, visibility_threshold=0.5)
    assert feats.shape == (4, FEATURE_DIM)
