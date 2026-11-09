"""Tests for stealzalut.core.color_transfer — MKL optimal transport."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from stealzalut.core.color_transfer import (
    METHOD_LABELS,
    SUPPORTED_METHODS,
    AffineTransform,
    ColorTransferEngine,
    OklabTransferModel,
    _compute_mkl_matrix,
    _matrix_sqrt,
    _matrix_sqrt_inv,
    fit_transfer,
)
from stealzalut.core.lut_engine import generate_neutral_grid

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

    def test_invalid_method_raises(self) -> None:
        """Initializing engine with unknown method raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported color transfer method"):
            ColorTransferEngine(method="nonexistent_algorithm")

    def test_supported_methods_constants(self) -> None:
        """SUPPORTED_METHODS and METHOD_LABELS have matching keys."""
        assert len(SUPPORTED_METHODS) == 8
        for m in SUPPORTED_METHODS:
            assert m in METHOD_LABELS

    @pytest.mark.parametrize("method", SUPPORTED_METHODS)
    def test_transfer_image_all_methods(
        self,
        method: str,
        source_image: np.ndarray,
        reference_image: np.ndarray,
    ) -> None:
        """Every supported method successfully transfers colors and keeps shape."""
        engine = ColorTransferEngine(method=method)
        result = engine.transfer_image(source_image, reference_image)
        assert result.shape == source_image.shape
        assert result.dtype == np.uint8
        assert not np.array_equal(result, source_image)

    @pytest.mark.parametrize("method", SUPPORTED_METHODS)
    def test_bake_grid_all_methods(
        self,
        method: str,
        source_image: np.ndarray,
        reference_image: np.ndarray,
    ) -> None:
        """Every supported method produces a valid [0, 1] bounded 3D LUT grid."""
        engine = ColorTransferEngine(method=method)
        grid = generate_neutral_grid(17)
        baked = engine.bake_grid(grid, source_image, reference_image)
        assert baked.shape == grid.shape
        assert baked.dtype == np.float64
        assert baked.min() >= 0.0 and baked.max() <= 1.0
        assert not np.isnan(baked).any()
        assert not np.isinf(baked).any()


# ── Fitted transfer models ───────────────────────────────────────────────────


class TestTransferModel:
    """Tests for fit_transfer / TransferModel (fit once, apply many times)."""

    @pytest.mark.parametrize("method", SUPPORTED_METHODS)
    def test_model_matches_bake_grid(
        self,
        method: str,
        source_image: np.ndarray,
        reference_image: np.ndarray,
    ) -> None:
        """Applying a fitted model gives the same grid as bake_grid."""
        engine = ColorTransferEngine(method=method)
        grid = generate_neutral_grid(17)
        model = fit_transfer(source_image, reference_image, method=method)
        assert model.method == method
        np.testing.assert_array_equal(
            model.apply(grid), engine.bake_grid(grid, source_image, reference_image)
        )

    @pytest.mark.parametrize("method", SUPPORTED_METHODS)
    def test_chunked_apply_equals_full_apply(
        self,
        method: str,
        source_image: np.ndarray,
        reference_image: np.ndarray,
    ) -> None:
        """The export applies the model in chunks: the result must not change."""
        model = fit_transfer(source_image, reference_image, method=method)
        grid = generate_neutral_grid(17)
        chunks = [model.apply(grid[i : i + 500]) for i in range(0, len(grid), 500)]
        np.testing.assert_array_equal(np.concatenate(chunks), model.apply(grid))

    def test_bake_uses_neutral_grid(
        self,
        source_image: np.ndarray,
        reference_image: np.ndarray,
    ) -> None:
        model = fit_transfer(source_image, reference_image)
        np.testing.assert_array_equal(model.bake(17), model.apply(generate_neutral_grid(17)))

    def test_engine_fit_uses_engine_method(
        self,
        source_image: np.ndarray,
        reference_image: np.ndarray,
    ) -> None:
        engine = ColorTransferEngine(method="reinhard")
        assert engine.fit(source_image, reference_image).method == "reinhard"

    def test_same_image_mkl_is_identity(self, source_image: np.ndarray) -> None:
        """Transferring an image onto itself leaves in-range colours unchanged."""
        model = fit_transfer(source_image, source_image, method="mkl")
        grid = generate_neutral_grid(17)
        assert_array_almost_equal(model.apply(grid), grid, decimal=6)

    def test_invalid_method_raises(
        self,
        source_image: np.ndarray,
        reference_image: np.ndarray,
    ) -> None:
        with pytest.raises(ValueError, match="Unsupported"):
            fit_transfer(source_image, reference_image, method="nope")



# ── Oklab decoupled transfer ─────────────────────────────────────────────────


class TestOklabTransfer:
    def test_same_image_is_near_identity(self, source_image: np.ndarray) -> None:
        model = fit_transfer(source_image, source_image, method="oklab")
        grid = generate_neutral_grid(17)
        np.testing.assert_allclose(model.apply(grid), grid, atol=1e-4)

    def test_scales_are_bounded(self) -> None:
        """A flat image must not make the transform explode."""
        flat = np.full((32, 32, 3), 128, dtype=np.uint8)
        rng = np.random.default_rng(0)
        busy = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
        model = fit_transfer(flat, busy, method="oklab")
        assert isinstance(model, OklabTransferModel)
        assert 0.5 <= model.lightness_scale <= 2.0
        eigenvalues = np.linalg.eigvalsh(model.chroma_matrix)
        assert np.all(eigenvalues >= 0.5 - 1e-9) and np.all(eigenvalues <= 2.0 + 1e-9)

    def test_moves_mean_towards_reference(
        self, source_image: np.ndarray, reference_image: np.ndarray
    ) -> None:
        from stealzalut.core.colorspace import srgb_to_oklab

        model = fit_transfer(source_image, reference_image, method="oklab")
        out = model.apply(source_image.reshape(-1, 3) / 255.0)
        mean_out = srgb_to_oklab(out).mean(axis=0)
        mean_ref = srgb_to_oklab(reference_image.reshape(-1, 3) / 255.0).mean(axis=0)
        mean_src = srgb_to_oklab(source_image.reshape(-1, 3) / 255.0).mean(axis=0)
        assert np.linalg.norm(mean_out - mean_ref) < 0.25 * np.linalg.norm(mean_src - mean_ref)

    def test_transfer_image_supports_oklab(
        self, source_image: np.ndarray, reference_image: np.ndarray
    ) -> None:
        out = ColorTransferEngine(method="oklab").transfer_image(source_image, reference_image)
        assert out.shape == source_image.shape and out.dtype == np.uint8
