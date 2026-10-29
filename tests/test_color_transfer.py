"""Tests for stealzalut.core.color_transfer — MKL optimal transport."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from stealzalut.core.color_transfer import (
    AffineTransform,
    ColorTransferEngine,
    _compute_mkl_matrix,
    _matrix_sqrt,
    _matrix_sqrt_inv,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────


def _make_synthetic_image(
    mean: tuple[float, float, float],
    std: float,
    h: int = 64,
    w: int = 64,
    seed: int = 42,
) -> np.ndarray:
    """Create a synthetic RGB uint8 image with a given mean colour."""
    rng = np.random.default_rng(seed)
    img = rng.normal(loc=mean, scale=std, size=(h, w, 3))
    return np.clip(img, 0, 255).astype(np.uint8)


@pytest.fixture()
def source_image() -> np.ndarray:
    """Warm-toned synthetic source image."""
    return _make_synthetic_image(mean=(180, 100, 80), std=30, seed=1)


@pytest.fixture()
def reference_image() -> np.ndarray:
    """Cool-toned synthetic reference image."""
    return _make_synthetic_image(mean=(80, 120, 200), std=25, seed=2)


@pytest.fixture()
def engine() -> ColorTransferEngine:
    return ColorTransferEngine()


# ── Matrix helpers ───────────────────────────────────────────────────────────


class TestMatrixSqrt:
    """Tests for _matrix_sqrt and _matrix_sqrt_inv."""

    def test_sqrt_of_identity(self) -> None:
        """√I = I."""
        identity = np.eye(3)
        result = _matrix_sqrt(identity)
        assert_array_almost_equal(result, identity)

    def test_sqrt_squared_is_original(self) -> None:
        """(M^{1/2})² ≈ M for a random SPD matrix."""
        rng = np.random.default_rng(99)
        a = rng.normal(size=(3, 3))
        m = a @ a.T + np.eye(3) * 0.1  # SPD
        sqrt_m = _matrix_sqrt(m)
        reconstructed = sqrt_m @ sqrt_m
        assert_array_almost_equal(reconstructed, m, decimal=10)

    def test_sqrt_inv_is_inverse_of_sqrt(self) -> None:
        """M^{1/2} · M^{-1/2} ≈ I."""
        rng = np.random.default_rng(77)
        a = rng.normal(size=(3, 3))
        m = a @ a.T + np.eye(3) * 0.1
        sqrt_m = _matrix_sqrt(m)
        sqrt_inv_m = _matrix_sqrt_inv(m)
        product = sqrt_m @ sqrt_inv_m
        assert_array_almost_equal(product, np.eye(3), decimal=10)


class TestComputeMKLMatrix:
    """Tests for _compute_mkl_matrix."""

    def test_identity_covariances_give_identity_transport(self) -> None:
        """When Σ_s = Σ_r = I, the transport matrix is I."""
        identity = np.eye(3)
        t = _compute_mkl_matrix(identity, identity)
        assert_array_almost_equal(t, identity)

    def test_transport_preserves_symmetry(self) -> None:
        """The transport matrix should be symmetric for Gaussian OT."""
        rng = np.random.default_rng(55)
        a = rng.normal(size=(3, 3))
        cov_s = a @ a.T + np.eye(3) * 0.1
        b = rng.normal(size=(3, 3))
        cov_r = b @ b.T + np.eye(3) * 0.1
        t = _compute_mkl_matrix(cov_s, cov_r)
        assert_array_almost_equal(t, t.T, decimal=10)


# ── Engine ───────────────────────────────────────────────────────────────────


class TestColorTransferEngine:
    """Tests for ColorTransferEngine."""

    def test_transfer_image_shape(
        self,
        engine: ColorTransferEngine,
        source_image: np.ndarray,
        reference_image: np.ndarray,
    ) -> None:
        """Transferred image must have the same shape as source."""
        result = engine.transfer_image(source_image, reference_image)
        assert result.shape == source_image.shape
        assert result.dtype == np.uint8

    def test_transfer_image_differs_from_source(
        self,
        engine: ColorTransferEngine,
        source_image: np.ndarray,
        reference_image: np.ndarray,
    ) -> None:
        """The transfer should actually change the image."""
        result = engine.transfer_image(source_image, reference_image)
        assert not np.array_equal(result, source_image)

    def test_compute_affine_shapes(
        self,
        engine: ColorTransferEngine,
        source_image: np.ndarray,
        reference_image: np.ndarray,
    ) -> None:
        """The affine transform must have the expected shapes."""
        transform = engine.compute_affine_transform(source_image, reference_image)
        assert transform.matrix.shape == (3, 3)
        assert transform.src_mean.shape == (3,)
        assert transform.ref_mean.shape == (3,)

    def test_apply_affine_identity(self, engine: ColorTransferEngine) -> None:
        """An identity transform should not change the input."""
        identity_transform = AffineTransform(
            matrix=np.eye(3),
            src_mean=np.array([0.5, 0.5, 0.5]),
            ref_mean=np.array([0.5, 0.5, 0.5]),
        )
        pixels = np.array([[0.2, 0.4, 0.6], [0.8, 0.1, 0.3]])
        result = engine.apply_affine(pixels, identity_transform)
        assert_array_almost_equal(result, pixels)

    def test_apply_affine_clamped(self, engine: ColorTransferEngine) -> None:
        """Output must be clamped to [0, 1]."""
        transform = AffineTransform(
            matrix=np.eye(3) * 5.0,  # aggressive scaling
            src_mean=np.array([0.5, 0.5, 0.5]),
            ref_mean=np.array([0.5, 0.5, 0.5]),
        )
        pixels = np.array([[1.0, 1.0, 1.0]])
        result = engine.apply_affine(pixels, transform)
        assert result.max() <= 1.0
        assert result.min() >= 0.0

    def test_affine_mean_shifts_correctly(
        self,
        engine: ColorTransferEngine,
        source_image: np.ndarray,
        reference_image: np.ndarray,
    ) -> None:
        """Source mean → reference mean when transformed."""
        transform = engine.compute_affine_transform(source_image, reference_image)
        # The source mean should map exactly to the reference mean
        src_as_row = transform.src_mean.reshape(1, 3)
        result = engine.apply_affine(src_as_row, transform)
        assert_array_almost_equal(result.ravel(), transform.ref_mean, decimal=10)

