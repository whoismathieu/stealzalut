"""Split an image into colour zones.

Each image is clustered (k-means in Oklab) into a handful of *zones* —
groups of pixels with similar colour, which in most photographs match
distinct elements (foliage, sky, skin, shadows…).  Zones are the unit the
correspondence table works with: "source zone S3 takes its colour from
reference zone R5".

The analysis is **deterministic** (fixed downscale, fixed RNG seed), so the
same image and zone count always give the same zones and ids.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from stealzalut.core.colorspace import gamut_compress, oklab_to_srgb, srgb_to_oklab
from stealzalut.utils.image_io import create_thumbnail

F64 = NDArray[np.float64]

#: Longest edge (px) of the copy that is analysed.
ANALYSIS_MAX_SIZE = 256
DEFAULT_ZONE_COUNT = 8
MIN_ZONE_COUNT = 2
MAX_ZONE_COUNT = 16
#: Zones smaller than this fraction of the image are merged into the closest one.
MIN_ZONE_AREA = 0.01
#: Ridge added to every zone covariance (keeps it invertible).
COV_REGULARIZATION = 1e-5
#: Bump when the zone analysis changes: recipes of older analyses get flagged.
ALGORITHM_VERSION = "zones-1"

_KMEANS_ATTEMPTS = 3
_KMEANS_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)


@dataclass(frozen=True, slots=True)
class Zone:
    """Statistics of one colour zone.

    Attributes:
        id: 1-based id; zones are numbered by increasing lightness.
        mean: Mean colour, Oklab ``(3,)``.
        cov: Colour covariance, Oklab ``(3, 3)`` (regularised).
        area: Fraction of the image covered, in ``[0, 1]``.
        l_rank: Position of the zone in the image's area-weighted lightness
            distribution (0 = darkest).  Robust to exposure differences.
        centroid_xy: Centre of mass, in % of width / height.
        bbox: ``(x0, y0, x1, y1)`` bounding box, in %.
        hex: Mean colour as ``#rrggbb`` (sRGB).
    """

    id: int
    mean: F64
    cov: F64
    area: float
    l_rank: float
    centroid_xy: tuple[float, float]
    bbox: tuple[float, float, float, float]
    hex: str


@dataclass(frozen=True, slots=True)
class ZonePalette:
    """All zones of one image plus the per-pixel zone map.

    Attributes:
        zones: Zones ordered by id (``zones[i].id == i + 1``).
        labels: Zone id of every pixel of the analysed copy, ``(h, w)`` int32.
    """

    zones: tuple[Zone, ...]
    labels: NDArray[np.int32]

    def zone(self, zone_id: int) -> Zone:
        """Return the zone with *zone_id* (1-based)."""
        if not 1 <= zone_id <= len(self.zones):
            raise KeyError(f"No zone {zone_id} (zones are 1..{len(self.zones)})")
        return self.zones[zone_id - 1]

    @property
    def ids(self) -> tuple[int, ...]:
        return tuple(z.id for z in self.zones)


# ── Analysis ─────────────────────────────────────────────────────────────────


def analyze_zones(
    image_rgb: NDArray[np.uint8],
    zone_count: int = DEFAULT_ZONE_COUNT,
) -> ZonePalette:
    """Cluster *image_rgb* into colour zones.

    Args:
        image_rgb: Image, **RGB** uint8, ``(H, W, 3)``.
        zone_count: Requested number of zones (clamped to
            ``[MIN_ZONE_COUNT, MAX_ZONE_COUNT]``).  Fewer zones may be
            returned when small ones are merged or the image has few colours.

    Returns:
        The image's :class:`ZonePalette`.
    """
    k = int(np.clip(zone_count, MIN_ZONE_COUNT, MAX_ZONE_COUNT))
    small = create_thumbnail(image_rgb, ANALYSIS_MAX_SIZE)
    h, w = small.shape[:2]
    lab = srgb_to_oklab(small.reshape(-1, 3).astype(np.float64) / 255.0)

    k = min(k, lab.shape[0])
    cv2.setRNGSeed(0)
    _, raw_labels, _ = cv2.kmeans(  # type: ignore[call-overload]
        lab.astype(np.float32),
        k,
        None,
        _KMEANS_CRITERIA,
        _KMEANS_ATTEMPTS,
        cv2.KMEANS_PP_CENTERS,
    )
    labels = _merge_small_zones(lab, raw_labels.ravel().astype(np.int64), MIN_ZONE_AREA)

    # Renumber 1..K by increasing mean lightness
    present = np.unique(labels)
    mean_l = np.array([lab[labels == z, 0].mean() for z in present])
    order = present[np.argsort(mean_l, kind="stable")]
    remap = np.zeros(int(present.max()) + 1, dtype=np.int32)
    remap[order] = np.arange(1, len(order) + 1, dtype=np.int32)
    zone_map = remap[labels].reshape(h, w)

    return ZonePalette(zones=_zone_stats(lab, zone_map), labels=zone_map)


def _merge_small_zones(
    lab: F64,
    labels: NDArray[np.int64],
    min_area: float,
) -> NDArray[np.int64]:
    """Merge zones below *min_area* into the zone with the closest mean."""
    labels = labels.copy()
    n = labels.size
    while True:
        present, counts = np.unique(labels, return_counts=True)
        if len(present) <= 1:
            return labels
        smallest = int(np.argmin(counts))
        if counts[smallest] / n >= min_area:
            return labels
        means = np.array([lab[labels == z].mean(axis=0) for z in present])
        dist = np.linalg.norm(means - means[smallest], axis=1)
        dist[smallest] = np.inf
        labels[labels == present[smallest]] = present[int(np.argmin(dist))]


def _zone_stats(lab: F64, zone_map: NDArray[np.int32]) -> tuple[Zone, ...]:
    h, w = zone_map.shape
    flat = zone_map.ravel()
    n = flat.size
    count = int(flat.max())

    areas = np.array([(flat == z).sum() / n for z in range(1, count + 1)])
    # Zones are sorted by lightness: the cumulative area gives each one's rank
    ranks = np.cumsum(areas) - areas / 2.0

    zones: list[Zone] = []
    for i, zone_id in enumerate(range(1, count + 1)):
        mask = flat == zone_id
        pixels = lab[mask]
        if pixels.shape[0] > 1:
            cov = np.asarray(np.cov(pixels, rowvar=False), dtype=np.float64)
        else:
            cov = np.zeros((3, 3))
        ys, xs = np.divmod(np.flatnonzero(mask), w)
        zones.append(
            Zone(
                id=zone_id,
                mean=pixels.mean(axis=0),
                cov=cov + COV_REGULARIZATION * np.eye(3),
                area=float(areas[i]),
                l_rank=float(ranks[i]),
                centroid_xy=(float(xs.mean() / w * 100.0), float(ys.mean() / h * 100.0)),
                bbox=(
                    float(xs.min() / w * 100.0),
                    float(ys.min() / h * 100.0),
                    float((xs.max() + 1) / w * 100.0),
                    float((ys.max() + 1) / h * 100.0),
                ),
                hex=_to_hex(pixels.mean(axis=0)),
            )
        )
    return tuple(zones)


def _to_hex(lab_mean: F64) -> str:
    rgb = oklab_to_srgb(gamut_compress(lab_mean[None, :]))[0]
    r, g, b = (round(float(c) * 255.0) for c in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def upsample_mask(mask: NDArray[np.bool_], width: int, height: int) -> NDArray[np.bool_]:
    """Resize a boolean zone mask with smooth (non-blocky) edges."""
    soft = cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
    return np.asarray(soft >= 0.5, dtype=np.bool_)


#: Pixels with at least this Oklab chroma count as "coloured".
COLOURFUL_CHROMA = 0.05
#: Hue concentration above which an image reads as one dominant colour.
MONOCHROME_CONCENTRATION = 0.85


@dataclass(frozen=True, slots=True)
class ColourStats:
    """How varied an image's colours are.

    Attributes:
        concentration: Chroma-weighted mean resultant length of the hues of
            coloured pixels: 1 = one single hue (a purple-graded night),
            ~0.3–0.4 = a varied palette; 0 when the image is nearly grey.
        colourful: Fraction of coloured pixels (chroma ≥ :data:`COLOURFUL_CHROMA`).
        hue: Dominant hue, degrees (Oklch).
    """

    concentration: float
    colourful: float
    hue: float

    @property
    def is_monochrome(self) -> bool:
        """One dominant colour: the zones method would give a uniform filter."""
        return self.concentration >= MONOCHROME_CONCENTRATION


def colour_stats(image_rgb: NDArray[np.uint8]) -> ColourStats:
    """Measure how much an image's colours share a single hue."""
    small = create_thumbnail(image_rgb, ANALYSIS_MAX_SIZE)
    lab = srgb_to_oklab(small.reshape(-1, 3).astype(np.float64) / 255.0)
    chroma = np.hypot(lab[:, 1], lab[:, 2])
    coloured = chroma >= COLOURFUL_CHROMA
    fraction = float(coloured.mean())
    if fraction < 0.05:
        return ColourStats(0.0, fraction, 0.0)
    # Unit hue vectors weighted by chroma: (a, b) / C × C = (a, b)
    a_sum, b_sum = float(lab[coloured, 1].sum()), float(lab[coloured, 2].sum())
    resultant = float(np.hypot(a_sum, b_sum) / chroma[coloured].sum())
    hue = float(np.degrees(np.arctan2(b_sum, a_sum)) % 360.0)
    return ColourStats(resultant, fraction, hue)


def analysis_id(
    source_rgb: NDArray[np.uint8],
    reference_rgb: NDArray[np.uint8],
    zone_count: int,
) -> str:
    """Fingerprint of an analysis (8 hex chars).

    Same images + same zone count + same algorithm → same id, hence the same
    zones (the analysis is deterministic).  Recipes carry it so that a
    recipe written for other zones can be detected.
    """
    k = int(np.clip(zone_count, MIN_ZONE_COUNT, MAX_ZONE_COUNT))
    digest = hashlib.sha256()
    for image in (source_rgb, reference_rgb):
        small = create_thumbnail(image, ANALYSIS_MAX_SIZE)
        digest.update(str(small.shape).encode())
        digest.update(np.ascontiguousarray(small).tobytes())
    digest.update(f"{k}|{ALGORITHM_VERSION}".encode())
    return digest.hexdigest()[:8]


def zone_mask(palette: ZonePalette, zone_id: int, size: tuple[int, int]) -> NDArray[np.bool_]:
    """Boolean mask of *zone_id* resized to *size* = ``(width, height)``."""
    resized = cv2.resize(palette.labels, size, interpolation=cv2.INTER_NEAREST)
    return cast("NDArray[np.bool_]", resized == zone_id)
