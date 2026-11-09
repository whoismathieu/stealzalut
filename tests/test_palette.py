"""Tests for stealzalut.core.palette — colour zone analysis."""

from __future__ import annotations

import numpy as np
import pytest

from stealzalut.core.palette import (
    ANALYSIS_MAX_SIZE,
    MIN_ZONE_AREA,
    analyze_zones,
    zone_mask,
)


def _scene() -> np.ndarray:
    """Four flat colour blocks + mild noise, RGB uint8."""
    rng = np.random.default_rng(0)
    img = np.zeros((200, 300, 3), dtype=np.float64)
    img[:100, :150] = (30, 60, 30)  # dark green   — 25 %
    img[:100, 150:] = (120, 170, 230)  # sky blue  — 25 %
    img[100:, :150] = (224, 172, 140)  # skin      — 25 %
    img[100:, 150:] = (240, 240, 235)  # near white — 25 %
    img += rng.normal(0, 3, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


class TestAnalyzeZones:
    def test_finds_the_four_blocks(self) -> None:
        palette = analyze_zones(_scene(), zone_count=4)
        assert len(palette.zones) == 4
        assert [z.id for z in palette.zones] == [1, 2, 3, 4]
        for zone in palette.zones:
            assert zone.area == pytest.approx(0.25, abs=0.01)

    def test_zones_sorted_by_lightness(self) -> None:
        palette = analyze_zones(_scene(), zone_count=4)
        lightness = [z.mean[0] for z in palette.zones]
        assert lightness == sorted(lightness)
        ranks = [z.l_rank for z in palette.zones]
        assert ranks == sorted(ranks)
        assert 0.0 < ranks[0] < ranks[-1] < 1.0

    def test_is_deterministic(self) -> None:
        a = analyze_zones(_scene(), zone_count=6)
        b = analyze_zones(_scene(), zone_count=6)
        np.testing.assert_array_equal(a.labels, b.labels)
        assert [z.hex for z in a.zones] == [z.hex for z in b.zones]

    def test_labels_cover_analysis_copy(self) -> None:
        img = np.zeros((1000, 500, 3), dtype=np.uint8)
        img[:, 250:] = 200
        palette = analyze_zones(img, zone_count=4)
        assert max(palette.labels.shape) == ANALYSIS_MAX_SIZE
        assert set(np.unique(palette.labels)) == set(palette.ids)

    def test_small_zones_are_merged(self) -> None:
        img = _scene()
        img[:3, :3] = (255, 0, 255)  # a 9-pixel magenta speck
        palette = analyze_zones(img, zone_count=8)
        assert all(z.area >= MIN_ZONE_AREA for z in palette.zones)

    def test_flat_image_gives_one_zone(self) -> None:
        palette = analyze_zones(np.full((50, 50, 3), 128, dtype=np.uint8), zone_count=8)
        assert len(palette.zones) == 1
        assert palette.zones[0].area == pytest.approx(1.0)
        assert palette.zones[0].hex == "#808080"

    def test_geometry(self) -> None:
        palette = analyze_zones(_scene(), zone_count=4)
        darkest = palette.zone(1)  # dark green, top-left block
        x0, y0, x1, y1 = darkest.bbox
        assert (x0, y0) == pytest.approx((0.0, 0.0), abs=1.0)
        assert (x1, y1) == pytest.approx((50.0, 50.0), abs=1.0)
        assert darkest.centroid_xy == pytest.approx((25.0, 25.0), abs=1.0)

    def test_unknown_zone_raises(self) -> None:
        with pytest.raises(KeyError):
            analyze_zones(_scene(), zone_count=4).zone(99)

    def test_zone_mask(self) -> None:
        palette = analyze_zones(_scene(), zone_count=4)
        mask = zone_mask(palette, 1, (60, 40))
        assert mask.shape == (40, 60)
        assert mask[:20, :30].mean() > 0.95
        assert mask[20:, 30:].mean() < 0.05
