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


# ── Engine ───────────────────────────────────────────────────────────────────


class ColorTransferEngine:
    """Encapsulates the MKL colour transfer pipeline.

    Usage::

        engine = ColorTransferEngine()

        # Preview — returns a transferred image
        preview = engine.transfer_image(source_rgb, reference_rgb)

        # LUT baking — extract the affine, then apply to a grid
        transform = engine.compute_affine_transform(source_rgb, reference_rgb)
        baked_grid = engine.apply_affine(neutral_grid, transform)
    """

    METHOD: str = "mkl"

    # ── Image-level transfer (via color-matcher) ─────────────────────────

    def transfer_image(
        self,
        source: NDArray[np.uint8],
        reference: NDArray[np.uint8],
    ) -> NDArray[np.uint8]:
        """Apply MKL colour transfer so *source* matches *reference*.

        Delegates to ``color-matcher`` for a battle-tested implementation
        that handles edge-cases (singular covariances, etc.).

        Args:
            source: Source image, **RGB** uint8, shape ``(H, W, 3)``.
            reference: Reference image, **RGB** uint8, shape ``(H, W, 3)``.

        Returns:
            Transferred image, **RGB** uint8, shape ``(H, W, 3)``.
        """
        cm = ColorMatcher()
        result = cm.transfer(src=source, ref=reference, method=self.METHOD)
        return cast("NDArray[np.uint8]", np.clip(result, 0, 255).astype(np.uint8))

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

