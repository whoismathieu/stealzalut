"""Color transfer engine using Monge–Kantorovitch optimal transport.

This module provides two complementary capabilities:

1. **Image transfer** — apply the MKL color transfer between a source
   and a reference image via the ``color-matcher`` library (for preview).
2. **Affine transform extraction** — compute the underlying 3×3 linear
   transport matrix so it can be applied to an arbitrary set of colours
   (for 3D LUT baking).

The MKL (Monge–Kantorovitch Linear) method models each image's colour
distribution as a multivariate Gaussian and finds the *optimal linear
map* (in the Wasserstein-2 sense) between the two distributions.

.. note::
    All public methods expect **RGB** images.  Convert from BGR
    (OpenCV's default) with :func:`stealzalut.utils.image_io.bgr_to_rgb`
    before calling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from color_matcher import ColorMatcher
from numpy.typing import NDArray

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
        return _bake_grid_impl(grid, source, reference, chosen_method)

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
        src_pixels = source.reshape(-1, 3).astype(np.float64) / 255.0
        ref_pixels = reference.reshape(-1, 3).astype(np.float64) / 255.0

        mu_s = src_pixels.mean(axis=0)
        mu_r = ref_pixels.mean(axis=0)

        cov_s = np.asarray(np.cov(src_pixels, rowvar=False), dtype=np.float64)
        cov_r = np.asarray(np.cov(ref_pixels, rowvar=False), dtype=np.float64)

        transport_matrix = _compute_mkl_matrix(cov_s, cov_r)

        return AffineTransform(
            matrix=transport_matrix,
            src_mean=mu_s,
            ref_mean=mu_r,
        )

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


def _bake_grid_impl(
    grid: NDArray[np.float64],
    source: NDArray[np.uint8],
    reference: NDArray[np.uint8],
    method: str,
) -> NDArray[np.float64]:
    """Execute grid baking for the specified transfer algorithm."""
    if method in ("mkl", "mvgd", "default"):
        src_pixels = source.reshape(-1, 3).astype(np.float64) / 255.0
        ref_pixels = reference.reshape(-1, 3).astype(np.float64) / 255.0
        mu_s = src_pixels.mean(axis=0)
        mu_r = ref_pixels.mean(axis=0)
        cov_s = np.asarray(np.cov(src_pixels, rowvar=False), dtype=np.float64)
        cov_r = np.asarray(np.cov(ref_pixels, rowvar=False), dtype=np.float64)
        transport_matrix = _compute_mkl_matrix(cov_s, cov_r)
        centered = grid - mu_s
        result = centered @ transport_matrix.T + mu_r
        return cast("NDArray[np.float64]", np.clip(result, 0.0, 1.0))

    if method == "reinhard":
        from color_matcher.reinhard_matcher import LMS_MAT, LMS_MAT_INV

        s_flat = np.maximum(source.reshape((-1, 3)).T.astype(np.float64) / 255.0, 1.0 / 255.0)
        r_flat = np.maximum(reference.reshape((-1, 3)).T.astype(np.float64) / 255.0, 1.0 / 255.0)

        b = np.array(
            [
                [1.0 / np.sqrt(3), 0.0, 0.0],
                [0.0, 1.0 / np.sqrt(6), 0.0],
                [0.0, 0.0, 1.0 / np.sqrt(2)],
            ]
        )
        c = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, -2.0], [1.0, -1.0, 0.0]])
        bc = b @ c
        cb = c.T @ b

        lab_s = bc @ np.log10(LMS_MAT @ s_flat)
        lab_r = bc @ np.log10(LMS_MAT @ r_flat)

        mean_s, std_s = np.mean(lab_s, axis=1), np.std(lab_s, axis=1)
        mean_r, std_r = np.mean(lab_r, axis=1), np.std(lab_r, axis=1)
        std_ratios = std_r / np.maximum(std_s, 1e-12)

        p_flat = np.maximum(grid.T, 1.0 / 255.0)
        lab_p = bc @ np.log10(LMS_MAT @ p_flat)
        lab_trans = ((lab_p.T - mean_s) * std_ratios + mean_r).T
        lms_res = 10.0 ** (cb @ lab_trans)
        rgb_res = (LMS_MAT_INV @ lms_res).T
        return cast("NDArray[np.float64]", np.clip(rgb_res, 0.0, 1.0))

    if method == "hm":
        baked = np.zeros_like(grid)
        for ch in range(3):
            s_vec = source[..., ch].ravel()
            r_vec = reference[..., ch].ravel()
            _, s_cnts = np.unique(s_vec, return_counts=True)
            r_vals, r_cnts = np.unique(r_vec, return_counts=True)
            s_cdf = np.cumsum(s_cnts).astype(np.float64) / s_vec.size
            r_cdf = np.cumsum(r_cnts).astype(np.float64) / r_vec.size
            interp_vals = np.interp(s_cdf, r_cdf, r_vals) / 255.0
            s_vals = np.unique(s_vec).astype(np.float64) / 255.0
            baked[:, ch] = np.interp(grid[:, ch], s_vals, interp_vals)
        return cast("NDArray[np.float64]", np.clip(baked, 0.0, 1.0))

    if method in ("hm-mkl-hm", "hm-mvgd-hm"):
        g1 = _bake_grid_impl(grid, source, reference, "hm")
        s1_flat = _bake_grid_impl(
            source.reshape(-1, 3).astype(np.float64) / 255.0, source, reference, "hm"
        )
        s1 = np.clip(s1_flat * 255.0, 0, 255).astype(np.uint8).reshape(source.shape)

        g2 = _bake_grid_impl(g1, s1, reference, "mkl")
        s2_flat = _bake_grid_impl(
            s1.reshape(-1, 3).astype(np.float64) / 255.0, s1, reference, "mkl"
        )
        s2 = np.clip(s2_flat * 255.0, 0, 255).astype(np.uint8).reshape(source.shape)

        return _bake_grid_impl(g2, s2, reference, "hm")

    raise ValueError(f"Unknown color transfer method: {method}")

