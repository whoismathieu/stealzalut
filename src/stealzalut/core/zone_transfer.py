"""Zone-based colour transfer: each source zone gets its own target.

For every point ``c`` of the LUT grid (in Oklab):

1. Each source zone ``k`` proposes a displacement ``Δ_k(c)``: a 2×2 chroma
   map from the zone's statistics to those of its target reference zones,
   dosed by the link's strengths.  Lightness follows the global transfer
   unless the link opts in (see :func:`zone_delta`), which keeps the grid
   free of folds.
2. A *global* component (Oklab chroma transfer and a monotone tone curve on
   the whole images, :class:`ToneGlobalModel`) covers colours that belong to
   no source zone.
3. Soft membership weights (Mahalanobis distance to each zone, softmax)
   blend the displacements::

       G(c) = c + Σ_k softmax(logit)_k · Δ_k(c)

4. The **manual retouch** of a zone (keeping its own lightness, exposure,
   saturation, warmth, tint: :func:`zone_retouch`) is a second displacement,
   added after the global adjustments so that they never dose it.

A LUT has no spatial information: zones are regions of *colour space*.
When two source zones overlap in colour but are sent to very different
targets, the LUT can only blend them — :func:`find_conflicts` reports it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations
from typing import cast

import numpy as np
from numpy.typing import NDArray

from stealzalut.core.color_transfer import (
    OKLAB_SCALE_LIMITS,
    OklabTransferModel,
    TransferModel,
    _bound_symmetric,
    _compute_mkl_matrix,
    fit_transfer,
)
from stealzalut.core.colorspace import gamut_compress, oklab_to_srgb, srgb_to_oklab
from stealzalut.core.correspondence import CorrespondenceTable, Intent, Level, Link, Retouch
from stealzalut.core.grading import smoothstep
from stealzalut.core.look import fit_tone_curve
from stealzalut.core.matching import match_zones
from stealzalut.core.palette import (
    DEFAULT_ZONE_COUNT,
    ColourStats,
    Zone,
    ZonePalette,
    analysis_id,
    analysis_oklab,
    analyze_zones,
    colour_stats,
)

F64 = NDArray[np.float64]

#: Intent level → magnitude, per intent kind.  Levels are what recipes written
#: before the retouch sliders hold; they keep this exact effect.
INTENT_STEPS: dict[str, dict[str, float]] = {
    "lightness": {"small": 0.02, "medium": 0.05, "large": 0.10},  # ΔL
    "saturation_up": {"small": 1.08, "medium": 1.20, "large": 1.40},  # ×C
    "saturation_down": {"small": 0.92, "medium": 0.82, "large": 0.65},
    "warmth": {"small": 0.010, "medium": 0.020, "large": 0.040},  # along WARM_AXIS
    "tint": {"small": 0.008, "medium": 0.016, "large": 0.030},  # Δa
}
#: Direction of "warmer" in the Oklab (a, b) plane (towards yellow-orange).
WARM_AXIS = np.array([0.29, 0.96]) / np.hypot(0.29, 0.96)

# Full travel of the retouch sliders: twice the largest intent level.
#: Exponent of the lightness curve at full exposure: ``L → L ** EXPOSURE_GAMMA``
#: lifts by up to :data:`RETOUCH_EXPOSURE`, most in the shadows (L ≈ 0.27);
#: the inverse exponent darkens by as much, most in the mid-tones.
EXPOSURE_GAMMA = 0.55
#: Largest lightness change of a full exposure (what the old levels are compared to).
RETOUCH_EXPOSURE = 0.217
#: Lightness under which a lift fades out, so that the black point stays black.
EXPOSURE_TOE = 0.06
#: Chroma factor removed / added by a full saturation (×0.3 … ×1.8).
RETOUCH_SATURATION: tuple[float, float] = (0.7, 0.8)
#: Shift along :data:`WARM_AXIS` at full warmth.
RETOUCH_WARMTH = 0.08
#: Shift of ``a`` at full tint.
RETOUCH_TINT = 0.06
#: Lightness range over which a lifted colour stops getting chroma back.
LIFT_CHROMA_TONES: tuple[float, float] = (0.30, 0.45)
#: Share of the chroma a dark colour gets back when lifted (1 = keeps ``C / L``).
LIFT_CHROMA_SHARE = 0.5

#: Overlap (Bhattacharyya coefficient) above which two zones can't be separated.
#: One colour mass split in two by k-means overlaps at ~0.45; clearly distinct
#: zones are near 0.
CONFLICT_OVERLAP = 0.3
#: Mean displacement difference (Oklab) above which overlapping zones conflict.
CONFLICT_DISPLACEMENT = 0.05


@dataclass(frozen=True, slots=True)
class ZoneBakeParams:
    """How zones are blended in the grid.

    Attributes:
        tau: Zone separation; smaller = sharper zone boundaries in colour
            space, larger = softer blending.
        sigma0: Floor added to zone spread (Oklab units).
        beta: Weight of the zone-area prior.
        background_logit: Logit of the global component; a colour more than
            ~3σ away from every zone falls back to the global transfer.
    """

    tau: float = 1.0
    sigma0: float = 0.02
    beta: float = 0.3
    background_logit: float = -4.5


@dataclass(frozen=True, slots=True)
class Conflict:
    """Two source zones too close in colour to be treated differently."""

    zone_a: int
    zone_b: int
    overlap: float
    displacement_gap: float


@dataclass(frozen=True, slots=True, eq=False)
class ToneGlobalModel(TransferModel):
    """Global component of the zones method: affine chroma, monotone tone curve.

    The plain Oklab transfer maps lightness with a straight line; under a
    dark or contrasted reference that line sends the low mid-tones to
    black.  The quantile-matched curve of the look method cannot go below
    the reference's own shadows and its slope is bounded.

    Attributes:
        colour: The Oklab transfer, used for chroma.
        tone_x: Input lightness samples of the curve, regular in ``[0, 1]``.
        tone_y: Output lightness at these samples (non-decreasing).
    """

    colour: OklabTransferModel
    tone_x: F64
    tone_y: F64

    @property
    def method(self) -> str:
        return "oklab"

    def apply(self, colors: F64) -> F64:
        return oklab_to_srgb(gamut_compress(self.apply_lab(srgb_to_oklab(colors))))

    def apply_lab(self, lab: F64) -> F64:
        out = self.colour.apply_lab(lab)
        lightness = lab[:, 0]
        inside = np.interp(lightness, self.tone_x, self.tone_y)
        # Outside [0, 1] (colours no display can show) the curve goes on at slope 1
        out[:, 0] = inside + (lightness - np.clip(lightness, 0.0, 1.0))
        return out


@dataclass(frozen=True, slots=True)
class ZoneTransferModel(TransferModel):
    """Transfer driven by a correspondence table between colour zones."""

    source: ZonePalette
    reference: ZonePalette
    table: CorrespondenceTable
    global_model: TransferModel
    params: ZoneBakeParams = ZoneBakeParams()
    #: Fingerprint of the analysis (see :func:`~stealzalut.core.palette.analysis_id`).
    analysis_id: str = ""
    #: Colour variety of both images (a monochrome reference suits "ambiance").
    source_colours: ColourStats | None = None
    reference_colours: ColourStats | None = None

    @property
    def method(self) -> str:
        return "zones"

    def fold_reference(self) -> TransferModel:
        # Folds already present in the plain global transfer are tolerated;
        # folds added by zone blending trigger automatic smoothing.
        return self.global_model

    def with_table(self, table: CorrespondenceTable) -> ZoneTransferModel:
        """Same zones and images, different correspondences (no refit)."""
        return replace(self, table=table)

    def with_params(self, params: ZoneBakeParams) -> ZoneTransferModel:
        return replace(self, params=params)

    def apply(self, colors: F64) -> F64:
        return oklab_to_srgb(gamut_compress(self.apply_lab(srgb_to_oklab(colors))))

    def apply_lab(self, lab: F64) -> F64:
        moved = self.transfer_lab(lab)
        retouch = self.retouch_lab(lab, moved)
        return moved if retouch is None else moved + retouch

    def transfer_lab(self, lab: F64) -> F64:
        return lab + self.displacement(lab)

    def retouch_lab(self, lab_in: F64, lab_now: F64) -> F64 | None:
        """Blended manual retouch of the zones, or ``None`` when no zone has one.

        Each retouched zone weighs on its own colours without the area
        prior: a face covering 3 % of the picture answers its sliders as
        frankly as a wall covering 30 %.
        """
        zones = self.source.zones
        links = [self.table.link(zone.id) for zone in zones]
        if not any(link.is_retouched for link in links):
            return None
        weights = self._weights(lab_in, area_prior=False)
        out = np.zeros_like(lab_in)
        for i, link in enumerate(links, start=1):
            if link.is_retouched:
                out += weights[:, i : i + 1] * zone_retouch(lab_in, lab_now, link)
        return out

    # ── Internals ────────────────────────────────────────────────────────

    def displacement(self, lab: F64) -> F64:
        """Blended Oklab displacement of the transfer for colours *lab* ``(N, 3)``,
        inside sRGB or not (the manual retouch is not included)."""
        zones = self.source.zones
        deltas = np.empty((len(zones) + 1, lab.shape[0], 3))
        global_delta = self.global_model.apply_lab(lab) - lab
        deltas[0] = global_delta
        for i, zone in enumerate(zones, start=1):
            deltas[i] = zone_delta(
                lab, zone, self.table.link(zone.id), self.reference, global_delta
            )
        return cast("F64", np.einsum("nk,knd->nd", self._weights(lab), deltas))

    def _weights(self, lab: F64, area_prior: bool = True) -> F64:
        """Membership of colours *lab* ``(N, 3)``: column 0 is the global
        component, then one column per source zone; rows sum to 1."""
        zones = self.source.zones
        logits = np.empty((lab.shape[0], len(zones) + 1))
        logits[:, 0] = self.params.background_logit
        for i, zone in enumerate(zones, start=1):
            logits[:, i] = self._zone_logit(lab, zone, area_prior)
        logits -= logits.max(axis=1, keepdims=True)
        weights = np.exp(logits)
        weights /= weights.sum(axis=1, keepdims=True)
        return cast("F64", weights)

    def _zone_logit(self, lab: F64, zone: Zone, area_prior: bool = True) -> F64:
        params = self.params
        cov = zone.cov + params.sigma0**2 * np.eye(3)
        diff = lab - zone.mean
        mahalanobis2 = np.einsum("ni,ij,nj->n", diff, np.linalg.inv(cov), diff)
        prior = params.beta * np.log(zone.area) if area_prior else 0.0
        return cast("F64", -0.5 * mahalanobis2 / params.tau + prior)


# ── Per-zone displacement ────────────────────────────────────────────────────


def _target_moments(reference: ZonePalette, targets: dict[int, float]) -> tuple[F64, F64]:
    """Mean and covariance of the mixture of the target reference zones."""
    weights = np.array(list(targets.values()), dtype=np.float64)
    weights /= weights.sum()
    zones = [reference.zone(zone_id) for zone_id in targets]
    mean = sum((w * z.mean for w, z in zip(weights, zones, strict=True)), np.zeros(3))
    cov = np.zeros((3, 3))
    for w, z in zip(weights, zones, strict=True):
        d = (z.mean - mean)[:, None]
        cov += w * (z.cov + d @ d.T)
    return mean, cov


def _signed_step(level: Level, kind: str) -> float:
    if level == "0":
        return 0.0
    return (1.0 if level[0] == "+" else -1.0) * INTENT_STEPS[kind][level[1:]]


def _saturation_factor(level: Level) -> float:
    if level == "0":
        return 1.0
    table = INTENT_STEPS["saturation_up" if level[0] == "+" else "saturation_down"]
    return table[level[1:]]


def zone_delta(
    lab: F64,
    zone: Zone,
    link: Link,
    reference: ZonePalette,
    global_delta: F64,
) -> F64:
    """Displacement proposed by one source zone for colours *lab* ``(N, 3)``.

    Zones choose **colours** (chroma).  Lightness follows the global
    transfer (*global_delta*), which is monotonic: taking each zone's target
    lightness would fold the grid whenever the matching does not preserve
    the lightness order (e.g. a bright sky matched to a dark sea).  A link
    can still opt in with ``luminance_strength`` (0 = global tone,
    1 = target zone's lightness).

    ``strength`` doses what the zone *chooses* (its colour, and its opt-in
    target lightness), never the shared global tone: zones dosed differently
    must keep the same lightness mapping, or neighbours would swap tones
    (folds, then forced smoothing that blends the zones together).  Only a
    locked zone leaves its colours completely untouched.
    """
    delta = np.zeros_like(lab)
    if link.locked:
        return delta

    delta[:, 0] = global_delta[:, 0]
    if link.targets and link.strength > 0.0:
        lo, hi = OKLAB_SCALE_LIMITS
        t_mean, t_cov = _target_moments(reference, link.targets)

        sigma_s = float(np.sqrt(zone.cov[0, 0]))
        sigma_t = float(np.sqrt(t_cov[0, 0]))
        scale = float(np.clip(sigma_t / max(sigma_s, 1e-6), lo, hi))
        chroma_map = _bound_symmetric(_compute_mkl_matrix(zone.cov[1:, 1:], t_cov[1:, 1:]), lo, hi)

        new_l = t_mean[0] + scale * (lab[:, 0] - zone.mean[0])
        new_ab = t_mean[1:] + (lab[:, 1:] - zone.mean[1:]) @ chroma_map.T

        lum = 0.0 if link.luminance_strength is None else link.luminance_strength
        chroma = 1.0 if link.chroma_strength is None else link.chroma_strength
        zone_dl = new_l - lab[:, 0]
        delta[:, 0] += link.strength * lum * (zone_dl - global_delta[:, 0])
        delta[:, 1:] = link.strength * chroma * (new_ab - lab[:, 1:])

    return delta


def zone_retouch(lab_in: F64, lab_now: F64, link: Link) -> F64:
    """Manual retouch of one zone, as a displacement added to *lab_now*.

    Args:
        lab_in: Colours as they are in the source, ``(N, 3)`` Oklab.
        lab_now: The same colours after the transfer and the global
            adjustments, ``(N, 3)``.
        link: The zone's link (its ``keep_tone``, ``retouch`` and ``intent``).

    The moves, in order:

    * ``keep_tone`` brings lightness back towards the source's.
    * ``exposure`` bends lightness with a power curve, ``L ** (EXPOSURE_GAMMA
      ** e)``: nothing at black and white, up to ±:data:`RETOUCH_EXPOSURE`
      in between, and always increasing with ``L``.  A dark colour that is
      lifted gets part of its chroma back, so it does not look washed out.
    * ``saturation`` scales chroma, ``warmth`` and ``tint`` shift it.

    The discrete levels of ``intent`` (recipes written before the sliders)
    keep their original effect and add up with the retouch.
    """
    out = np.zeros_like(lab_now)
    if not link.is_retouched:
        return out
    retouch, intent = link.retouch, link.intent

    lightness = lab_now[:, 0] - link.keep_tone * (lab_now[:, 0] - lab_in[:, 0])
    bounded = np.clip(lightness, 0.0, 1.0)
    lift = _exposure_lift(bounded, retouch.exposure)
    lift = lift + _signed_step(intent.lightness, "lightness")

    dark = 1.0 - smoothstep(bounded, *LIFT_CHROMA_TONES)
    gain = 1.0 + LIFT_CHROMA_SHARE * dark * np.maximum(lift, 0.0) / np.maximum(bounded, 0.15)
    gain = gain * _retouch_saturation(retouch.saturation) * _saturation_factor(intent.saturation)

    ab = lab_now[:, 1:] * gain[:, None]
    warmth = retouch.warmth * RETOUCH_WARMTH + _signed_step(intent.warmth, "warmth")
    ab = ab + warmth * WARM_AXIS
    ab[:, 0] += retouch.tint * RETOUCH_TINT + _signed_step(intent.tint, "tint")

    out[:, 0] = lightness + lift - lab_now[:, 0]
    out[:, 1:] = ab - lab_now[:, 1:]
    return out


def _exposure_lift(lightness: F64, exposure: float) -> F64:
    """Lightness change of an exposure retouch for *lightness* in ``[0, 1]``."""
    if exposure == 0.0:
        return np.zeros_like(lightness)
    lift = lightness ** (EXPOSURE_GAMMA**exposure) - lightness
    if exposure > 0.0:
        # A power curve is infinitely steep at black: fade the lift in
        lift = lift * smoothstep(lightness, 0.0, EXPOSURE_TOE)
    return cast("F64", lift)


def _retouch_saturation(value: float) -> float:
    down, up = RETOUCH_SATURATION
    return 1.0 + (up if value > 0.0 else down) * value


def retouch_from_intent(intent: Intent) -> Retouch:
    """Slider positions that come closest to the discrete levels of *intent*."""

    def position(level: Level, kind: str, full: float) -> float:
        return float(np.clip(_signed_step(level, kind) / full, -1.0, 1.0))

    down, up = RETOUCH_SATURATION
    factor = _saturation_factor(intent.saturation) - 1.0
    return Retouch(
        exposure=position(intent.lightness, "lightness", RETOUCH_EXPOSURE),
        saturation=float(np.clip(factor / (up if factor > 0.0 else down), -1.0, 1.0)),
        warmth=position(intent.warmth, "warmth", RETOUCH_WARMTH),
        tint=position(intent.tint, "tint", RETOUCH_TINT),
    )


# ── Conflicts ────────────────────────────────────────────────────────────────


def bhattacharyya_coefficient(mean_a: F64, cov_a: F64, mean_b: F64, cov_b: F64) -> float:
    """Overlap in ``[0, 1]`` between two Gaussians (1 = identical)."""
    cov = 0.5 * (cov_a + cov_b)
    diff = mean_a - mean_b
    term_mean = 0.125 * float(diff @ np.linalg.solve(cov, diff))
    _, logdet = np.linalg.slogdet(cov)
    _, logdet_a = np.linalg.slogdet(cov_a)
    _, logdet_b = np.linalg.slogdet(cov_b)
    term_cov = 0.5 * (logdet - 0.5 * (logdet_a + logdet_b))
    return float(np.exp(-(term_mean + term_cov)))


def find_conflicts(model: ZoneTransferModel) -> list[Conflict]:
    """Pairs of source zones the LUT cannot treat differently.

    Two zones conflict when their colours overlap strongly (they occupy the
    same region of colour space) *and* the table sends them to clearly
    different results.
    """
    zones = model.source.zones
    means = np.array([z.mean for z in zones])
    global_delta = model.global_model.apply_lab(means) - means
    mean_delta = {
        z.id: zone_delta(
            z.mean[None, :], z, model.table.link(z.id), model.reference, global_delta[i : i + 1]
        )[0]
        for i, z in enumerate(zones)
    }
    conflicts: list[Conflict] = []
    for a, b in combinations(zones, 2):
        overlap = bhattacharyya_coefficient(a.mean, a.cov, b.mean, b.cov)
        gap = float(np.linalg.norm(mean_delta[a.id] - mean_delta[b.id]))
        if overlap > CONFLICT_OVERLAP and gap > CONFLICT_DISPLACEMENT:
            conflicts.append(Conflict(a.id, b.id, overlap, gap))
    return conflicts


# ── Fitting ──────────────────────────────────────────────────────────────────


def fit_zone_transfer(
    source: NDArray[np.uint8],
    reference: NDArray[np.uint8],
    zone_count: int = DEFAULT_ZONE_COUNT,
    params: ZoneBakeParams | None = None,
) -> ZoneTransferModel:
    """Analyse both images into zones and match them automatically.

    Args:
        source: Source image, **RGB** uint8.
        reference: Reference image, **RGB** uint8.
        zone_count: Requested zones per image.
        params: Blending parameters (defaults: :class:`ZoneBakeParams`).

    Returns:
        A model whose table holds the automatic correspondences.
    """
    src_palette = analyze_zones(source, zone_count)
    ref_palette = analyze_zones(reference, zone_count)
    table = CorrespondenceTable.from_matching(match_zones(src_palette, ref_palette))
    model = ZoneTransferModel(
        source=src_palette,
        reference=ref_palette,
        table=table,
        global_model=fit_global(source, reference),
        params=params or ZoneBakeParams(),
        analysis_id=analysis_id(source, reference, zone_count),
        source_colours=colour_stats(source),
        reference_colours=colour_stats(reference),
    )
    return model.with_table(harmonize_conflicts(model))


def fit_global(source: NDArray[np.uint8], reference: NDArray[np.uint8]) -> ToneGlobalModel:
    """Global component: Oklab chroma transfer and quantile-matched tone curve."""
    colour = fit_transfer(source, reference, method="oklab")
    assert isinstance(colour, OklabTransferModel)
    tone_x, tone_y = fit_tone_curve(
        analysis_oklab(source)[0][:, 0], analysis_oklab(reference)[0][:, 0]
    )
    return ToneGlobalModel(colour=colour, tone_x=tone_x, tone_y=tone_y)


def harmonize_conflicts(model: ZoneTransferModel, max_rounds: int = 4) -> CorrespondenceTable:
    """Give overlapping source zones the same targets.

    k-means often splits one colour mass (e.g. foliage) into neighbouring
    zones; the matching may then send them to different targets, which the
    LUT can only blend (a conflict).  Each conflicting pair gets the union of
    both target sets, weighted by zone area.  Only automatic links are
    changed: links set by the user or an AI are kept as they are.
    """
    table = model.table
    for _ in range(max_rounds):
        conflicts = find_conflicts(model.with_table(table))
        changed = False
        for conflict in conflicts:
            a, b = table.link(conflict.zone_a), table.link(conflict.zone_b)
            if a.origin != "auto" or b.origin != "auto" or a.locked or b.locked:
                continue
            area_a = model.source.zone(a.source_zone).area
            area_b = model.source.zone(b.source_zone).area
            merged: dict[int, float] = {}
            for link, area in ((a, area_a), (b, area_b)):
                for ref_id, weight in link.targets.items():
                    merged[ref_id] = merged.get(ref_id, 0.0) + weight * area
            total = sum(merged.values())
            targets = {k: w / total for k, w in sorted(merged.items())}
            if targets != a.targets or targets != b.targets:
                table = table.with_link(replace(a, targets=targets))
                table = table.with_link(replace(b, targets=dict(targets)))
                changed = True
        if not changed:
            break
    return table
