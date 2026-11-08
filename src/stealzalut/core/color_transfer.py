"""Color transfer engine using Monge–Kantorovitch optimal transport.

This module provides two complementary capabilities:

1. **Image transfer** — apply the MKL color transfer between a source
   and a reference image via the ``color-matcher`` library (for preview).
2. **Fitted transfer models** — :func:`fit_transfer` reads the two images
   once and returns a :class:`TransferModel` that maps arbitrary colours
   (a 3D LUT grid of any size, in any number of chunks).  The live preview
   and the ``.cube`` export both go through the same model, so the preview
   shows exactly what is exported.

The MKL (Monge–Kantorovitch Linear) method models each image's colour
distribution as a multivariate Gaussian and finds the *optimal linear
map* (in the Wasserstein-2 sense) between the two distributions.

.. note::
    All public methods expect **RGB** images.  Convert from BGR
    (OpenCV's default) with :func:`stealzalut.utils.image_io.bgr_to_rgb`
    before calling.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import cast

import numpy as np
from color_matcher import ColorMatcher
from numpy.typing import NDArray

from stealzalut.core.lut_engine import generate_neutral_grid

# ── Data structures ──────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class AffineTransform:
    """Affine colour transform: ``y = matrix @ (x - src_mean) + ref_mean``.

    Attributes:
        matrix: The 3×3 optimal transport matrix.
        src_mean: Mean colour of the source image (3,) in [0, 1].
        ref_mean: Mean colour of the reference image (3,) in [0, 1].
    """

    matrix: NDArray[np.float64]
    src_mean: NDArray[np.float64]
    ref_mean: NDArray[np.float64]


# ── Method definitions ───────────────────────────────────────────────────────

SUPPORTED_METHODS: tuple[str, ...] = (
    "mkl",
    "reinhard",
    "hm",
    "mvgd",
    "hm-mkl-hm",
    "hm-mvgd-hm",
    "default",
)

METHOD_LABELS: dict[str, str] = {
    "mkl": "Monge–Kantorovitch (MKL)",
    "reinhard": "Reinhard (Lab Stats)",
    "hm": "Histogram Matching (HM)",
    "mvgd": "Gaussian (MVGD)",
    "hm-mkl-hm": "Hybrid (HM + MKL + HM)",
    "hm-mvgd-hm": "Hybrid (HM + MVGD + HM)",
    "default": "Default (ColorMatcher)",
}


# ── Engine ───────────────────────────────────────────────────────────────────


class ColorTransferEngine:
    """Encapsulates colour transfer pipelines and 3D LUT baking.

    Supports all transfer algorithms offered by ``color-matcher``:
    - ``mkl`` (Monge–Kantorovitch Linear optimal transport)
    - ``reinhard`` (Reinhard statistical alignment in Ruderman Lab space)
    - ``hm`` (Channel-wise histogram matching)
    - ``mvgd`` (Multivariate Gaussian distribution transfer)
    - ``hm-mkl-hm`` (Hybrid histogram matching + MKL + histogram matching)
    - ``hm-mvgd-hm`` (Hybrid histogram matching + MVGD + histogram matching)
    - ``default`` (ColorMatcher default)

    Usage::

        engine = ColorTransferEngine(method="mkl")

        # Preview — returns a transferred image
        preview = engine.transfer_image(source_rgb, reference_rgb)

        # LUT baking — bake chosen algorithm onto neutral 3D grid
        baked_grid = engine.bake_grid(neutral_grid, source_rgb, reference_rgb)
    """

    METHOD: str = "mkl"

    def __init__(self, method: str = "mkl") -> None:
        chosen = method.lower()
        if chosen not in SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported color transfer method '{method}'. "
                f"Supported: {', '.join(SUPPORTED_METHODS)}"
            )
        self.method: str = chosen

    # ── Image-level transfer (via color-matcher) ─────────────────────────

    def transfer_image(
        self,
        source: NDArray[np.uint8],
        reference: NDArray[np.uint8],
        method: str | None = None,
    ) -> NDArray[np.uint8]:
        """Apply colour transfer so *source* matches *reference*.

        Delegates to ``color-matcher`` for a battle-tested implementation
        that handles edge-cases (singular covariances, etc.).

        Args:
            source: Source image, **RGB** uint8, shape ``(H, W, 3)``.
            reference: Reference image, **RGB** uint8, shape ``(H, W, 3)``.
            method: Color transfer method name. If None, uses engine's method.

        Returns:
            Transferred image, **RGB** uint8, shape ``(H, W, 3)``.
        """
        chosen_method = (method or self.method).lower()
        if chosen_method not in SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported color transfer method '{chosen_method}'. "
                f"Supported: {', '.join(SUPPORTED_METHODS)}"
            )
        cm = ColorMatcher()
        result = cm.transfer(src=source, ref=reference, method=chosen_method)
        return cast("NDArray[np.uint8]", np.clip(result, 0, 255).astype(np.uint8))

    # ── 3D LUT Grid Baking ───────────────────────────────────────────────

    def bake_grid(
        self,
        grid: NDArray[np.float64],
        source: NDArray[np.uint8],
        reference: NDArray[np.uint8],
        method: str | None = None,
    ) -> NDArray[np.float64]:
        """Bake the chosen color transfer algorithm onto a neutral 3D LUT grid.

        Args:
            grid: Neutral grid of shape ``(N, 3)`` in [0.0, 1.0].
            source: Source image, **RGB** uint8, shape ``(H, W, 3)``.
            reference: Reference image, **RGB** uint8, shape ``(H, W, 3)``.
            method: Color transfer method name. If None, uses engine's method.

        Returns:
            Baked grid of shape ``(N, 3)`` in [0.0, 1.0], clamped.
        """
        chosen_method = (method or self.method).lower()
        if chosen_method not in SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported color transfer method '{chosen_method}'. "
                f"Supported: {', '.join(SUPPORTED_METHODS)}"
            )
        return self.fit(source, reference, method=chosen_method).apply(grid)

    def fit(
        self,
        source: NDArray[np.uint8],
        reference: NDArray[np.uint8],
        method: str | None = None,
    ) -> TransferModel:
        """Fit a reusable :class:`TransferModel` (see :func:`fit_transfer`).

        Prefer this over :meth:`bake_grid` when the same transfer is applied
        several times (chunked export, preview + export): the image
        statistics are computed only once.
        """
        return fit_transfer(source, reference, method=method or self.method)

    # ── Affine transform extraction (pure numpy) ─────────────────────────

    def compute_affine_transform(
        self,
        source: NDArray[np.uint8],
        reference: NDArray[np.uint8],
    ) -> AffineTransform:
        """Compute the closed-form MKL affine transform.

        Extracts the 3×3 transport matrix and colour means so the same
        mapping can be applied to arbitrary colour values (e.g. a neutral
        LUT grid).

        The transform maps any colour *x* (in [0, 1]) to::

            y = T @ (x − μ_src) + μ_ref

        where *T* is the Monge–Kantorovitch optimal transport matrix
        between the Gaussian fits of the two colour distributions.

        Args:
            source: Source image, **RGB** uint8, shape ``(H, W, 3)``.
            reference: Reference image, **RGB** uint8, shape ``(H, W, 3)``.

        Returns:
            An :class:`AffineTransform` holding the matrix and means.
        """
        return _fit_affine_transform(source, reference)

    # ── Apply precomputed transform ──────────────────────────────────────

    @staticmethod
    def apply_affine(
        pixels: NDArray[np.float64],
        transform: AffineTransform,
    ) -> NDArray[np.float64]:
        """Apply a precomputed affine transform to an array of colours.

        Args:
            pixels: Colour values of shape ``(N, 3)`` in [0, 1].
            transform: Previously computed :class:`AffineTransform`.

        Returns:
            Transformed colours of shape ``(N, 3)``, clamped to [0, 1].
        """
        centered = pixels - transform.src_mean
        result = centered @ transform.matrix.T + transform.ref_mean
        return cast("NDArray[np.float64]", np.clip(result, 0.0, 1.0))


# ── Private helpers ──────────────────────────────────────────────────────────


def _matrix_sqrt(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """Principal square root of a symmetric positive semi-definite matrix.

    Uses eigen-decomposition:  ``M^{1/2} = V diag(√λ) Vᵀ``.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    # Clamp negative eigenvalues arising from numerical noise
    eigenvalues = np.maximum(eigenvalues, 0.0)
    res = eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T
    return cast("NDArray[np.float64]", res)


def _matrix_sqrt_inv(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """Inverse square root of a symmetric positive-definite matrix.

    Uses eigen-decomposition:  ``M^{-1/2} = V diag(1/√λ) Vᵀ``.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    # Guard against near-zero eigenvalues
    eigenvalues = np.maximum(eigenvalues, 1e-12)
    res = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
    return cast("NDArray[np.float64]", res)


def _compute_mkl_matrix(
    cov_s: NDArray[np.float64],
    cov_r: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Compute the Monge–Kantorovitch linear transport matrix.

    Closed-form solution for Gaussian optimal transport:

    .. math::

        T = \Sigma_s^{-1/2}
            \bigl(\Sigma_s^{1/2}\,\Sigma_r\,\Sigma_s^{1/2}\bigr)^{1/2}
            \Sigma_s^{-1/2}

    Args:
        cov_s: Source covariance matrix ``(3, 3)``.
        cov_r: Reference covariance matrix ``(3, 3)``.

    Returns:
        Transport matrix ``(3, 3)``.
    """
    sqrt_s = _matrix_sqrt(cov_s)
    sqrt_s_inv = _matrix_sqrt_inv(cov_s)

    # Inner product:  Σ_s^{1/2} · Σ_r · Σ_s^{1/2}
    inner = sqrt_s @ cov_r @ sqrt_s
    sqrt_inner = _matrix_sqrt(inner)

    return sqrt_s_inv @ sqrt_inner @ sqrt_s_inv


def _to_float_pixels(image: NDArray[np.uint8]) -> NDArray[np.float64]:
    """Flatten an RGB uint8 image to ``(N, 3)`` float64 in [0, 1]."""
    return image.reshape(-1, 3).astype(np.float64) / 255.0


def _fit_affine_transform(
    source: NDArray[np.uint8],
    reference: NDArray[np.uint8],
) -> AffineTransform:
    """Fit the closed-form MKL affine transform between two images."""
    src_pixels = _to_float_pixels(source)
    ref_pixels = _to_float_pixels(reference)
    cov_s = np.asarray(np.cov(src_pixels, rowvar=False), dtype=np.float64)
    cov_r = np.asarray(np.cov(ref_pixels, rowvar=False), dtype=np.float64)
    return AffineTransform(
        matrix=_compute_mkl_matrix(cov_s, cov_r),
        src_mean=src_pixels.mean(axis=0),
        ref_mean=ref_pixels.mean(axis=0),
    )


# ── Fitted transfer models ───────────────────────────────────────────────────
#
# A model holds everything derived from the two images.  It is fitted once
# (the expensive part: statistics over every pixel) and can then be applied
# to any set of colours — a LUT grid of any size, in as many chunks as needed.


class TransferModel(ABC):
    """A colour transfer fitted on a source/reference pair."""

    __slots__ = ()

    @property
    @abstractmethod
    def method(self) -> str:
        """Name of the transfer method this model implements."""

    @abstractmethod
    def apply(self, colors: NDArray[np.float64]) -> NDArray[np.float64]:
        """Map colours ``(N, 3)`` in [0, 1] to transferred colours, clamped."""

    def bake(self, size: int) -> NDArray[np.float64]:
        """Bake the model onto a neutral grid of *size*³ points."""
        return self.apply(generate_neutral_grid(size))


@dataclass(frozen=True, slots=True)
class AffineTransferModel(TransferModel):
    """MKL optimal-transport affine map ``y = T (x − μ_s) + μ_r``."""

    transform: AffineTransform
    method_name: str = "mkl"

    @property
    def method(self) -> str:
        return self.method_name

    def apply(self, colors: NDArray[np.float64]) -> NDArray[np.float64]:
        return ColorTransferEngine.apply_affine(colors, self.transform)


@dataclass(frozen=True, slots=True)
class ReinhardTransferModel(TransferModel):
    """Reinhard statistics alignment in Ruderman's log-LMS lαβ space."""

    mean_s: NDArray[np.float64]
    mean_r: NDArray[np.float64]
    std_ratios: NDArray[np.float64]

    @property
    def method(self) -> str:
        return "reinhard"

    def apply(self, colors: NDArray[np.float64]) -> NDArray[np.float64]:
        lms_mat, lms_mat_inv = _reinhard_lms_matrices()
        p_flat = np.maximum(colors.T, 1.0 / 255.0)
        lab_p = _REINHARD_BC @ np.log10(lms_mat @ p_flat)
        lab_trans = ((lab_p.T - self.mean_s) * self.std_ratios + self.mean_r).T
        lms_res = 10.0 ** (_REINHARD_CB @ lab_trans)
        rgb_res = (lms_mat_inv @ lms_res).T
        return cast("NDArray[np.float64]", np.clip(rgb_res, 0.0, 1.0))


@dataclass(frozen=True, slots=True)
class HistogramTransferModel(TransferModel):
    """Per-channel histogram matching, stored as three 1-D lookup curves."""

    src_values: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
    mapped_values: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]

    @property
    def method(self) -> str:
        return "hm"

    def apply(self, colors: NDArray[np.float64]) -> NDArray[np.float64]:
        out = np.empty_like(colors)
        for ch in range(3):
            out[:, ch] = np.interp(colors[:, ch], self.src_values[ch], self.mapped_values[ch])
        return cast("NDArray[np.float64]", np.clip(out, 0.0, 1.0))


@dataclass(frozen=True, slots=True)
class ChainedTransferModel(TransferModel):
    """Several models applied one after another (e.g. HM → MKL → HM)."""

    steps: tuple[TransferModel, ...]
    method_name: str

    @property
    def method(self) -> str:
        return self.method_name

    def apply(self, colors: NDArray[np.float64]) -> NDArray[np.float64]:
        for step in self.steps:
            colors = step.apply(colors)
        return colors


# ── Reinhard constants ───────────────────────────────────────────────────────

_REINHARD_B = np.array(
    [
        [1.0 / np.sqrt(3), 0.0, 0.0],
        [0.0, 1.0 / np.sqrt(6), 0.0],
        [0.0, 0.0, 1.0 / np.sqrt(2)],
    ]
)
_REINHARD_C = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, -2.0], [1.0, -1.0, 0.0]])
_REINHARD_BC = _REINHARD_B @ _REINHARD_C
_REINHARD_CB = _REINHARD_C.T @ _REINHARD_B


def _reinhard_lms_matrices() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """RGB↔LMS matrices used by color-matcher's Reinhard implementation."""
    from color_matcher.reinhard_matcher import LMS_MAT, LMS_MAT_INV

    return LMS_MAT, LMS_MAT_INV


# ── Fitting ──────────────────────────────────────────────────────────────────


def _fit_reinhard(
    source: NDArray[np.uint8],
    reference: NDArray[np.uint8],
) -> ReinhardTransferModel:
    lms_mat, _ = _reinhard_lms_matrices()
    s_flat = np.maximum(_to_float_pixels(source).T, 1.0 / 255.0)
    r_flat = np.maximum(_to_float_pixels(reference).T, 1.0 / 255.0)
    lab_s = _REINHARD_BC @ np.log10(lms_mat @ s_flat)
    lab_r = _REINHARD_BC @ np.log10(lms_mat @ r_flat)
    std_s = np.std(lab_s, axis=1)
    std_r = np.std(lab_r, axis=1)
    return ReinhardTransferModel(
        mean_s=np.mean(lab_s, axis=1),
        mean_r=np.mean(lab_r, axis=1),
        std_ratios=std_r / np.maximum(std_s, 1e-12),
    )


def _fit_histogram(
    source: NDArray[np.uint8],
    reference: NDArray[np.uint8],
) -> HistogramTransferModel:
    src_values: list[NDArray[np.float64]] = []
    mapped_values: list[NDArray[np.float64]] = []
    for ch in range(3):
        s_vec = source[..., ch].ravel()
        r_vec = reference[..., ch].ravel()
        s_vals, s_cnts = np.unique(s_vec, return_counts=True)
        r_vals, r_cnts = np.unique(r_vec, return_counts=True)
        s_cdf = np.cumsum(s_cnts).astype(np.float64) / s_vec.size
        r_cdf = np.cumsum(r_cnts).astype(np.float64) / r_vec.size
        mapped_values.append(np.interp(s_cdf, r_cdf, r_vals) / 255.0)
        src_values.append(s_vals.astype(np.float64) / 255.0)
    return HistogramTransferModel(
        src_values=(src_values[0], src_values[1], src_values[2]),
        mapped_values=(mapped_values[0], mapped_values[1], mapped_values[2]),
    )


def _transform_image(model: TransferModel, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Apply *model* to every pixel of *image* (truncating back to uint8)."""
    flat = model.apply(_to_float_pixels(image))
    return np.clip(flat * 255.0, 0, 255).astype(np.uint8).reshape(image.shape)


def fit_transfer(
    source: NDArray[np.uint8],
    reference: NDArray[np.uint8],
    method: str = "mkl",
) -> TransferModel:
    """Fit a colour transfer model on a source/reference pair.

    This is the only step that reads the images.  The returned model can be
    applied to any colours (typically a LUT grid) without re-reading them.

    Args:
        source: Source image, **RGB** uint8, shape ``(H, W, 3)``.
        reference: Reference image, **RGB** uint8, shape ``(H, W, 3)``.
        method: One of :data:`SUPPORTED_METHODS`.

    Returns:
        The fitted :class:`TransferModel`.

    Raises:
        ValueError: If *method* is not supported.
    """
    chosen = method.lower()
    if chosen in ("mkl", "mvgd", "default"):
        return AffineTransferModel(_fit_affine_transform(source, reference), method_name=chosen)

    if chosen == "reinhard":
        return _fit_reinhard(source, reference)

    if chosen == "hm":
        return _fit_histogram(source, reference)

    if chosen in ("hm-mkl-hm", "hm-mvgd-hm"):
        first_hm = _fit_histogram(source, reference)
        stage1 = _transform_image(first_hm, source)
        affine = AffineTransferModel(_fit_affine_transform(stage1, reference))
        stage2 = _transform_image(affine, stage1)
        last_hm = _fit_histogram(stage2, reference)
        return ChainedTransferModel(steps=(first_hm, affine, last_hm), method_name=chosen)

    raise ValueError(
        f"Unsupported color transfer method '{method}'. "
        f"Supported: {', '.join(SUPPORTED_METHODS)}"
    )


