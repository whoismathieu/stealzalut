"""Tests for stealzalut.core.matching — automatic zone correspondences."""

from __future__ import annotations

import numpy as np
import pytest

from stealzalut.core.matching import match_zones, sinkhorn, zone_cost
from stealzalut.core.palette import analyze_zones


def _blocks(colors: list[tuple[int, int, int]]) -> np.ndarray:
    """Vertical stripes of equal width, one per colour (RGB uint8)."""
    width = 40
    img = np.zeros((60, width * len(colors), 3), dtype=np.uint8)
    for i, c in enumerate(colors):
        img[:, i * width : (i + 1) * width] = c
    return img


class TestSinkhorn:
    def test_marginals(self) -> None:
        rng = np.random.default_rng(0)
        cost = rng.random((5, 4)) * 10.0
        a, b = np.full(5, 0.2), np.array([0.1, 0.2, 0.3, 0.4])
        plan = sinkhorn(cost, a, b, epsilon=0.01)
        np.testing.assert_allclose(plan.sum(axis=1), a, atol=1e-6)
        np.testing.assert_allclose(plan.sum(axis=0), b, atol=1e-6)

    def test_small_epsilon_does_not_underflow(self) -> None:
        cost = np.array([[0.0, 50.0], [50.0, 0.0]])
        plan = sinkhorn(cost, np.full(2, 0.5), np.full(2, 0.5), epsilon=1e-3)
        assert np.all(np.isfinite(plan))
        np.testing.assert_allclose(plan, [[0.5, 0.0], [0.0, 0.5]], atol=1e-6)


class TestMatchZones:
    def test_lightness_order_is_preserved(self) -> None:
        """Dark → dark, mid → mid, bright → bright, whatever the hues."""
        src = analyze_zones(_blocks([(20, 40, 20), (90, 140, 90), (220, 240, 220)]), 3)
        ref = analyze_zones(_blocks([(60, 20, 10), (190, 110, 60), (255, 220, 180)]), 3)
        matching = match_zones(src, ref)
        assert set(matching) == {1, 2, 3}
        for zone_id in (1, 2, 3):
            best = max(matching[zone_id], key=matching[zone_id].__getitem__)
            assert best == zone_id

    def test_rows_are_normalised(self) -> None:
        src = analyze_zones(_blocks([(20, 40, 20), (90, 140, 90), (220, 240, 220)]), 3)
        ref = analyze_zones(_blocks([(60, 20, 10), (255, 220, 180)]), 2)
        for targets in match_zones(src, ref).values():
            assert targets
            assert sum(targets.values()) == pytest.approx(1.0)
            assert all(w > 0 for w in targets.values())

    def test_cost_shape_and_zero_on_identical(self) -> None:
        palette = analyze_zones(_blocks([(20, 40, 20), (220, 240, 220)]), 2)
        cost = zone_cost(palette, palette)
        assert cost.shape == (2, 2)
        np.testing.assert_allclose(np.diag(cost), 0.0, atol=1e-12)
        assert cost[0, 1] > 0
