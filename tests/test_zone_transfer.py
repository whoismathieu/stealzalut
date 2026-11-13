"""Tests for stealzalut.core.zone_transfer and correspondence tables."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from stealzalut.core.color_transfer import fit_transfer
from stealzalut.core.colorspace import oklab_to_oklch, srgb_to_oklab
from stealzalut.core.correspondence import CorrespondenceTable, Intent, Link
from stealzalut.core.grading import build_grid
from stealzalut.core.lut_engine import generate_neutral_grid
from stealzalut.core.zone_transfer import (
    ZoneTransferModel,
    bhattacharyya_coefficient,
    find_conflicts,
    fit_zone_transfer,
    harmonize_conflicts,
)


def _stripes(colors: list[tuple[int, int, int]], noise: float = 4.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    width = 50
    img = np.zeros((80, width * len(colors), 3), dtype=np.float64)
    for i, c in enumerate(colors):
        img[:, i * width : (i + 1) * width] = c
    img += rng.normal(0, noise, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


SOURCE = _stripes([(30, 70, 30), (120, 170, 230), (224, 172, 140)])  # foliage, sky, skin
REFERENCE = _stripes([(30, 110, 120), (215, 180, 120), (255, 150, 70)], seed=1)  # sea, sand, sunset


@pytest.fixture(scope="module")
def model() -> ZoneTransferModel:
    return fit_zone_transfer(SOURCE, REFERENCE, zone_count=3)


def _zone_lab(model: ZoneTransferModel, zone_id: int) -> np.ndarray:
    return model.source.zone(zone_id).mean[None, :]


def _apply_lab(model: ZoneTransferModel, lab: np.ndarray) -> np.ndarray:
    return model.apply_lab(lab)


class TestCorrespondenceTable:
    def test_from_matching_and_with_link(self) -> None:
        table = CorrespondenceTable.from_matching({1: {2: 1.0}, 2: {1: 0.5, 3: 0.5}})
        assert table.link(2).targets == {1: 0.5, 3: 0.5}
        new = table.with_link(Link(source_zone=2, locked=True, origin="user"))
        assert new.link(2).locked and new.link(2).origin == "user"
        assert not table.link(2).locked  # original untouched (immutable)
        assert new.link(1) == table.link(1)

    def test_unknown_zone_raises(self) -> None:
        table = CorrespondenceTable.from_matching({1: {1: 1.0}})
        with pytest.raises(KeyError):
            table.with_link(Link(source_zone=5))

    def test_identity_links(self) -> None:
        assert Link(source_zone=1, locked=True, targets={1: 1.0}).is_identity
        assert not Link(source_zone=1).is_identity  # no target: its tone still follows
        assert Link(source_zone=1).keeps_colours
        assert Link(source_zone=1, targets={1: 1.0}, strength=0.0).keeps_colours
        assert not Link(source_zone=1, intent=Intent(warmth="+small")).keeps_colours


class TestZoneTransfer:
    def test_fit_registers_as_method(self) -> None:
        m = fit_transfer(SOURCE, REFERENCE, "zones")
        assert isinstance(m, ZoneTransferModel)
        assert m.method == "zones"
        assert m.fold_reference() is m.global_model

    def test_each_zone_moves_towards_its_target(self, model: ZoneTransferModel) -> None:
        for link in model.table.links:
            src_ab = _zone_lab(model, link.source_zone)[0, 1:]
            out_ab = _apply_lab(model, _zone_lab(model, link.source_zone))[0, 1:]
            target_id = max(link.targets, key=link.targets.__getitem__)
            tgt_ab = model.reference.zone(target_id).mean[1:]
            assert np.linalg.norm(out_ab - tgt_ab) < np.linalg.norm(src_ab - tgt_ab)

    def test_locked_zone_follows_only_global_lightness(self, model: ZoneTransferModel) -> None:
        zone_id = 3
        locked = model.with_table(model.table.with_link(Link(source_zone=zone_id, locked=True)))
        lab = _zone_lab(model, zone_id)
        out = _apply_lab(locked, lab)
        # Fully owned by the locked zone → (almost) no displacement at all
        np.testing.assert_allclose(out, lab, atol=0.01)

    def test_retargeting_changes_only_that_zone(self, model: ZoneTransferModel) -> None:
        other_target = 1 if 1 not in model.table.link(3).targets else 3
        edited = model.with_table(
            model.table.with_link(Link(source_zone=3, targets={other_target: 1.0}))
        )
        before = _apply_lab(model, _zone_lab(model, 1))
        after = _apply_lab(edited, _zone_lab(model, 1))
        np.testing.assert_allclose(after, before, atol=1e-3)  # zone 1 unaffected
        moved = _apply_lab(edited, _zone_lab(model, 3)) - _apply_lab(model, _zone_lab(model, 3))
        assert np.linalg.norm(moved) > 0.01

    def test_lightness_follows_global_by_default(self, model: ZoneTransferModel) -> None:
        lab = srgb_to_oklab(generate_neutral_grid(17))
        zones_dl = model.displacement(lab)[:, 0]
        global_dl = model.global_model.apply_lab(lab)[:, 0] - lab[:, 0]
        np.testing.assert_allclose(zones_dl, global_dl, atol=1e-5)

    def test_strength_doses_colour_not_tone(self, model: ZoneTransferModel) -> None:
        """Regression: a strength < 1 also shrank the shared global lightness,
        so zones dosed differently swapped tones (folds, forced smoothing)."""
        table = model.table
        for link, strength in zip(table.links, (0.3, 0.6, 0.0), strict=True):
            table = table.with_link(replace(link, strength=strength))
        dosed = model.with_table(table)
        grid = generate_neutral_grid(17)
        lab = srgb_to_oklab(grid)
        global_dl = model.global_model.apply_lab(lab)[:, 0] - lab[:, 0]
        np.testing.assert_allclose(dosed.displacement(lab)[:, 0], global_dl, atol=1e-5)

        # Strength 0: the zone keeps its colours but its tone follows the image
        lab3 = _zone_lab(model, 3)
        moved = dosed.displacement(lab3)
        assert abs(moved[0, 0]) > 1e-4 or abs(global_dl).max() < 1e-4
        full = model.displacement(lab3)
        assert np.linalg.norm(moved[0, 1:]) < 0.25 * np.linalg.norm(full[0, 1:])

        result = build_grid(dosed, 33)
        assert result.fold_fraction == 0.0 and not result.auto_smoothed

    def test_intents(self, model: ZoneTransferModel) -> None:
        lab = _zone_lab(model, 2)
        base = _apply_lab(model, lab)
        warmer = model.with_table(
            model.table.with_link(
                Link(source_zone=2, targets=model.table.link(2).targets, intent=Intent(
                    warmth="+large", lightness="+medium", saturation="-large"
                ))
            )
        )
        out = _apply_lab(warmer, lab)
        assert out[0, 0] > base[0, 0]  # lighter
        assert out[0, 2] > base[0, 2]  # more yellow (b+) → warmer
        warm_shift = 0.04 * np.array([0.29, 0.96]) / np.hypot(0.29, 0.96)
        c_out = oklab_to_oklch(out - np.array([0.0, *warm_shift]))[0, 1]
        assert c_out < oklab_to_oklch(base)[0, 1]  # desaturated before the warmth shift

    def test_grid_has_no_folds(self, model: ZoneTransferModel) -> None:
        result = build_grid(model, 33)
        assert result.fold_fraction == 0.0
        assert not result.auto_smoothed
        assert result.grid.min() >= 0.0 and result.grid.max() <= 1.0


class TestConflicts:
    def test_bhattacharyya(self) -> None:
        cov = np.eye(3) * 1e-3
        mean = np.zeros(3)
        assert bhattacharyya_coefficient(mean, cov, mean, cov) == pytest.approx(1.0)
        assert bhattacharyya_coefficient(mean, cov, mean + 1.0, cov) < 1e-6

    def test_distinct_zones_do_not_conflict(self, model: ZoneTransferModel) -> None:
        assert find_conflicts(model) == []

    def test_automatic_table_has_no_conflict(self) -> None:
        """Zones split from one colour mass get the same automatic targets."""
        src = _stripes([(60, 120, 60), (64, 124, 64), (230, 230, 230)], noise=6.0)
        m = fit_zone_transfer(src, REFERENCE, zone_count=3)
        assert find_conflicts(m) == []

    def test_harmonize_keeps_manual_links(self) -> None:
        src = _stripes([(60, 120, 60), (64, 124, 64), (230, 230, 230)], noise=6.0)
        m = fit_zone_transfer(src, REFERENCE, zone_count=3)
        table = m.table.with_link(Link(source_zone=1, targets={1: 1.0}, origin="user"))
        table = table.with_link(Link(source_zone=2, targets={3: 1.0}))
        harmonized = harmonize_conflicts(m.with_table(table))
        assert harmonized.link(1).targets == {1: 1.0}  # user link untouched
        assert harmonized.link(2).targets == {3: 1.0}  # only auto pairs are merged

    def test_overlapping_zones_sent_apart_conflict(self) -> None:
        # Two very similar greens (one zone each) sent to opposite targets
        src = _stripes([(60, 120, 60), (64, 124, 64), (230, 230, 230)], noise=6.0)
        m = fit_zone_transfer(src, REFERENCE, zone_count=3)
        greens = [z.id for z in m.source.zones if z.mean[2] > 0.03 or z.mean[1] < -0.03]
        if len(greens) < 2:
            pytest.skip("k-means merged the two greens")
        a, b = greens[:2]
        table = m.table.with_link(Link(source_zone=a, targets={1: 1.0}))
        table = table.with_link(Link(source_zone=b, targets={3: 1.0}))
        conflicts = find_conflicts(m.with_table(table))
        assert any({c.zone_a, c.zone_b} == {a, b} for c in conflicts)
