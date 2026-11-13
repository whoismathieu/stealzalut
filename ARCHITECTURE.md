# StealZaLut — Technical Architecture & Reference Context

> **Single source of truth.** This document and `README.md` are the **only context files** of the project. They replace and consolidate all earlier notes (`CONTEXT.md`, `docs/approach-v3/*`, `docs/audit/*`), which remain available in the Git history.

---

## 1. Engineering Rules & Guardrails (Non-Negotiable)

Every developer or AI assistant working on this repository must follow these rules:

1. **One set of documents (no bloat).** Never create secondary `.md` files (status logs, sub-plans, session notes) to iterate. Every architectural update goes into `ARCHITECTURE.md` and `README.md` only.
2. **No AI attribution.** Never write `Co-Authored-By`, `Generated with…` or the name of an AI model in commits, comments, docstrings or any file of the repository.
3. **Git and network safety.**
   - **Never** run `git push`, `git remote -v`, `git config --list`, and never read `.git/config`: the remote URL contains a token in clear text.
   - The application is **100 % offline**: no network dependency, no AI API call, no key.
4. **Disk space.** The local drive is nearly full (over 95 %). Never write test images, `.npy` files or comparison plates in the repository. Use a temporary folder outside the repository (`mktemp -d`) and delete it right after use.
5. **Languages and validation.**
   - Interface texts and user-facing error messages are in **French**. Code, identifiers, docstrings, commit messages (`feat:`, `fix:`, `docs:`) and these two documents are in **English**.
   - Before every commit, the three commands below must pass with no error and no warning:

     ```bash
     .venv/Scripts/python.exe -m pytest -q
     .venv/Scripts/python.exe -m ruff check src tests
     .venv/Scripts/python.exe -m mypy src
     ```

---

## 2. Layers & Source Tree

The application follows a strictly decoupled **Model → Workers → View** pattern:

- **`src/stealzalut/core/`**: pure mathematics, colour science and serialisation. **Importing `PySide6` here is forbidden.**
- **`src/stealzalut/workers/`**: computations wrapped in `QThread` (`PySide6.QtCore` only), reporting to the interface through `Signal`.
- **`src/stealzalut/gui/`**: `PySide6.QtWidgets` presentation layer and immutable undo history.

```text
src/stealzalut/
├── main.py                     # QApplication entry point, loads theme.qss
├── config.py                   # .env -> frozen Settings dataclass (default: method="look")
├── core/                       # PURE LOGIC (no Qt dependency)
│   ├── colorspace.py           # sRGB <-> linear RGB <-> Oklab <-> Oklch, gamut_compress
│   ├── rgb_spaces.py           # RGBSpace (Bradford D50->D65), CUBE_SPACES (sRGB, P3, Adobe RGB, Rec.709)
│   ├── color_transfer.py       # Abstract TransferModel, fit_transfer, SUPPORTED_METHODS (11)
│   ├── look.py                 # Default "look" engine (monotone tone curve + per-band chroma transport)
│   ├── palette.py              # Deterministic Oklab k-means (256 px), ColourStats, analysis_id
│   ├── matching.py             # Entropic optimal matching (Sinkhorn) between ZonePalettes
│   ├── correspondence.py       # Immutable CorrespondenceTable, Link, Retouch, Intent, pure edit operations
│   ├── zone_transfer.py        # "zones" engine, soft Mahalanobis blend, per-zone retouch, Bhattacharyya conflicts
│   ├── ambiance.py             # "ambiance" engine (tint curve along L for monochrome references)
│   ├── grading.py              # GradingParams, masks (neutrals/skin/vivid), unified build_grid
│   ├── regularize.py           # Separable 3D Gaussian blur (smooth_field), fold measure (fold_fraction)
│   ├── finishing.py            # Finishing: 3 Bernstein wheels, PCHIP curve, 8 HSL bands, sat/vibrance
│   ├── lut_engine.py           # Neutral grid [0,1]^3 (R fastest, G, B slowest), VALID_LUT_SIZES={17,33,64}
│   ├── lut_apply.py            # Trilinear interpolation (apply_lut, apply_lut_colors, render_lut)
│   ├── lut_export.py           # Reads and writes the standard Iridas .cube format
│   ├── recipe.py               # Parser and serialiser of .szl v2 recipes (look, zones, ambiance)
│   ├── look_pack.py            # AI pack of the "look" mode (with 3_current_render.jpg and the gap table)
│   └── ai_pack.py              # AI pack of the "zones" mode (isolated zone plates and overviews)
├── workers/                    # ASYNCHRONOUS THREADS (QtCore)
│   ├── transfer_worker.py      # Fits the model, bakes the preview grid, renders a WideImage
│   └── export_worker.py        # Bakes at the target size and writes the .cube file
├── gui/                        # PRESENTATION (QtWidgets)
│   ├── main_window.py          # UI orchestration, model cache (_model_cache), stale workers
│   ├── history.py              # Immutable undo/redo stack (EditState, 1.0 s merge window)
│   ├── overlay.py              # Highlights and outlines zones on the thumbnails
│   ├── zone_text.py            # French labels for intents, origins and dominant colours
│   ├── styles/theme.qss        # Dark Qt theme (font sizes in px)
│   └── widgets/                # image_drop_zone, preview_panel, correspondence_view, zone_inspector,
│                               # adjustments_panel, finishing_panel, paste_recipe_dialog
├── utils/
│   ├── image_io.py             # WideImage, native ICC decoding, RAW/DNG (embedded preview / rawpy), HEIC
│   └── paths.py                # Path resolution compatible with PyInstaller (sys._MEIPASS)
└── devtools/
    ├── ground_truth.py         # Known-truth bench in Display P3 (not imported by the application)
    └── smoke_gui.py            # Captures the real window on one pair, in "look" or "zones" mode (visual check)
```

---

## 3. Wide-Gamut Colour Pipeline (`WideImage` → Oklab → `.cube`)

### 3.1. The three colour spaces

1. **Reading space (`utils/image_io.py:WideImage`)**
   - `load_image_rgb` returns a `WideImage`, a `uint8` subclass of `np.ndarray`.
   - The `uint8` array itself is the 8-bit sRGB reduction, used only for display in Qt.
   - When the file carries a wide-gamut ICC profile recognised by substring (`Display P3`, `Adobe RGB` / `ClayRGB` / `opRGB`, `ProPhoto` / `ROMM`), `WideImage` carries two more things:
     - `analysis_lab`: an `(h, w, 3)` `float64` array of the **unclipped Oklab** colours of the analysis copy (256 px), computed from the file's own values with the exact matrices of `core/rgb_spaces.py`.
     - `native` and `native_space`: the full-size pixels as encoded in the file, and their `RGBSpace`.
   - **`WideImage` invariant.** Any slicing, copy or resize (`create_thumbnail`, `np.asarray`, indexing) goes through `__array_finalize__` and resets `analysis_lab` and `native` to `None`. The engine's single entry point for reading the analysis is `core/palette.py:analysis_oklab(image)`.
2. **Engine space (`core/colorspace.py`)**
   - The whole engine works in **unbounded Oklab**. The conversion to LMS uses a signed cube root (`np.cbrt(lms)`), so coordinates outside the sRGB gamut are kept without clipping.
   - Every Oklab model (`LookTransferModel`, `ZoneTransferModel`, `AmbianceTransferModel`, `OklabTransferModel`) implements `apply_lab(self, lab: F64) -> F64` natively, without going through sRGB.
3. **Export and LUT application space (`core/rgb_spaces.py`, `core/lut_apply.py`)**
   - Spaces available in `CUBE_SPACES`: `SRGB`, `DISPLAY_P3`, `ADOBE_RGB` (gamma 563/256), `REC709` (gamma 2.4).
   - `PROPHOTO_RGB` (adapted from D50 to D65 with a Bradford matrix) is defined for reading files but **excluded from `CUBE_SPACES`**: its imaginary primaries put corners of the `[0, 1]³` cube where LMS responses are negative, which breaks the monotonicity of `L` (0.6 % of folds on the identity grid alone).
   - In `render_lut`, the source's `native` pixels are converted to the space chosen for the `.cube`, interpolated trilinearly in the 3D grid of that space, and only then converted and gamut-compressed to 8-bit sRGB for the screen or the image export.

### 3.2. Unified baking chain (`core/grading.py:build_grid`)

For a size `N` in {17, 33, 64} and a target `space: RGBSpace`, there is one sequence:

1. **Input.** The neutral grid `[0, 1]³`, shape `(N³, 3)`, is converted to Oklab: `lab_in = space.to_oklab(grid)`.
2. **Raw transfer.** `lab_raw = model.apply_lab(lab_in)`.
3. **Strength and masks (`adjust_displacement`).** On the displacement field `Δ = lab_raw − lab_in`:
   - `ΔL` and `Δab` are scaled separately (`luminance_strength`, `chroma_strength`).
   - `Δab` only is attenuated (never `ΔL`, to avoid folds) by three `smoothstep` masks evaluated in Oklch on the input:
     - `neutral_mask` protects greys (C from 0.02 to 0.06);
     - `skin_mask` is a window on hue 27.5°–87.5° (centre 57.5°, full weight on 37.5°–77.5°), C from 0.010 to 0.22, L from 0.12 to 0.98 (widened towards dark and less saturated tones, so dark brown and black skin are covered). It protects chroma only: holding skin lightness with a global mask folds 1.6 to 5 % of the grid on 3 real pairs out of 8, even after the strongest automatic smoothing. A skin that came out too dark is lifted per zone (`keep_tone`, `exposure`);
     - `vivid_mask` protects saturated colours (C from 0.04 to 0.12).
   - The whole displacement is multiplied by `mix`.
4. **Regularisation and fold control (`core/regularize.py`).**
   - Separable 3D Gaussian blur `smooth_field`, with a radius in cells of `smoothing × (N − 1) / 32`.
   - `fold_fraction` measures the share of grid edges along which `L` drops by more than 0.001 while R, G or B increases. If the settings add more than 0.5 % of folds (`MAX_EXTRA_FOLDS`) over `model.fold_reference()`, the smoothing is doubled automatically (starting at 0.5, up to 3 steps).
5. **Manual retouch (`model.retouch_lab`).** A model may carry a retouch set by hand (the per-zone sliders of the zones method). It is added here, after the adjustments and the fold control: strengths, protections and mix never dose it, and it never raises the automatic smoothing. It is smoothed once, lightly (`RETOUCH_SMOOTHING = 0.5`, or the user's smoothing when larger). `model.transfer_lab` is the part steps 2 to 4 work on; `apply_lab` is both.
6. **Finishing (`core/finishing.py:apply_finishing`).**
   - Monotone cubic tone curve on `L` (Fritsch–Carlson / PCHIP, no SciPy dependency).
   - 3 colour wheels (shadows, midtones, highlights) weighted by the Bernstein basis `(1−L)²`, `2L(1−L)`, `L²`.
   - 8 Gaussian colour bands (σ = 22°, weighted by `smoothstep(C, 0.01, 0.04)` so greys are ignored), plus global saturation and vibrance.
7. **Output and gamut.** Gamut compression at constant hue and lightness, by bisection (12 iterations) towards the linear cube of the target space (`gamut_compress(lab_out, to_linear=space.oklab_to_linear)`), then encoding with `space.from_oklab`.

---

## 4. Mathematics of the Transfer Engines

### 4.1. `look` engine (« Rendu » in the interface — `core/look.py`)

Designed to reproduce the behaviour of an analog print without tonal inversions and without losing saturation at the gamut boundary.

1. **Monotone tone curve `T(L)`**
   - Matches the lightness quantiles of the source and of the reference (weighted by `reference_weights`) at levels 0.01, 0.05, 0.15, 0.30, 0.50, 0.70, 0.85 (the bulk of the scene) and at 0.999 (the white point).
   - The jump from 85 % to 99.9 % matters: it aligns the level of light sources (signs, neons) without forcing them to cover the same area as in the reference.
   - The slope between two consecutive knots is bounded to [0.35, 3.0] (`TONE_SLOPE`), then the curve is interpolated by a monotone cubic on 129 samples.
2. **Tone-conditional chroma transport (12 bands)**
   - For each band `m` in 0…11, centred on the source quantile `ℓ_m` and sent to the output tone `ℓ'_m = T(ℓ_m)`, 3 000 points `(a, b)` are drawn in a Gaussian window around `ℓ_m` (source) and around `ℓ'_m` (reference).
   - The source cloud `P` is carried onto the target cloud `Q` by **sliced optimal transport** (`sliced_transport`, 48 directions following the golden-ratio sequence), weighted by the trust `trust_m = min(1, share_m / 0.004)²`.
   - The displacement `D = trust_m · (moved − P)` is split into three terms:

     ```math
     \Delta ab(L, a, b) = \mathrm{cast}(L) + \mathrm{separation} \cdot \Big( \big(ab - c(L)\big)\,\mathrm{stretch}(L)^\top + \mathrm{field}(L, p_a, p_b) \Big)
     ```

     - **`cast_m`**: the mean displacement of the band (the colour cast).
     - **`stretch_m`**: a 2×2 linear map `B` fitted by ridge least squares (`STRETCH_PRIOR = 0.005`) on the displacement minus its mean, with the singular values of `I + B` bounded to [0.5, 2.0]. It carries a saturation gain or a hue rotation exactly, even outside the support of the source.
     - **`field_m`**: the non-linear residual, splatted bilinearly on a 49×49 lattice in companded coordinates `p = ab / (‖ab‖ + 0.05)`, combining a fine blur (σ = 1.5 cells) and a far field (σ = 6 cells, weight 2.0).
3. **Hue-selective protection (`protect_hue`)**
   - If the reference is detected as monochrome (`concentration >= 0.85`), `protect_vivid` is set to 0.8 automatically, but only for source hues more than 15° to 45° (`PROTECT_HUE_RANGE`) away from the dominant hue of the reference.
4. **Locality (« Fidélité à ma photo » in the interface; `locality` in [0, 1])**
   - The slider value `t` in (0, 1] becomes a maximum reach `r = 0.30 · (0.02 / 0.30)^t` for chroma and `r_L = 2r` for tone. At `t = 0` there is no limit.
   - A soft limit is applied with a hyperbolic tangent, without refitting anything:

     ```math
     L_{\mathrm{out}} = L + r_L \tanh\!\left(\frac{T(L) - L}{r_L}\right), \qquad \Delta ab_{\mathrm{out}} = \Delta ab \cdot \frac{\tanh\left(\lVert \Delta ab \rVert / r\right)}{\lVert \Delta ab \rVert / r}
     ```

   - The derivative of `L_out` stays between 1 and `T'(L)`, so the curve remains increasing: no fold is possible.

### 4.2. `zones` engine (`core/palette.py`, `matching.py`, `zone_transfer.py`)

- **Segmentation.** `cv2.kmeans` (K = 8, RNG seed fixed to 0) on the 256 px Oklab copy. Zones under 1 % of the area are merged; zones are sorted by increasing `L` (1…K).
- **Sinkhorn matching.** Entropic regularisation ε = 0.05 in the log domain, with the cost matrix:

  ```math
  C_{ij} = 2.0\,(\ell_i - \ell_j)^2 + 0.5\,\frac{\lVert ab_i - ab_j \rVert^2}{s_{ab}^2}
  ```

- **Global component (`ToneGlobalModel`).** Chroma follows the affine Oklab transfer; lightness follows the monotone quantile curve of the look engine (`fit_tone_curve`), not a straight line: a line sends the low mid-tones under the reference's own shadows, the curve cannot. On the bench this alone takes `zones` on same content from 0.17 / 0.13 to 0.12 / 0.08 (tuning / control lot).
- **Baking.** Each zone applies a 2×2 MKL map in chroma towards the blend of its targets; lightness follows the monotone global transfer by default (`luminance_strength = None`) to avoid folds. Zones are blended with soft weights (Mahalanobis distance regularised by σ₀ = 0.02, temperature τ, area prior β = 0.3, global fallback logit −4.5).
- **Manual retouch of a zone (`zone_retouch`, `Link.keep_tone`, `Link.retouch`).** A second displacement, separate from the transfer:
  - `keep_tone` (0 to 1) brings the zone's lightness back towards the one it has in the source;
  - `retouch.exposure` (−1 to 1) bends lightness with a power curve `L ** (0.55 ** e)`: nothing at black and white, up to ±0.217 in between, always increasing with `L`; a lift fades out under L = 0.06, and a dark colour that is lifted gets half of its `C / L` back;
  - `retouch.saturation` scales chroma from ×0.3 to ×1.8, `retouch.warmth` shifts it by up to 0.08 along the warm axis, `retouch.tint` by up to 0.06 on `a`.
  - The retouch of each zone is blended with the same soft weights **without the area prior**, so a zone covering 3 % of the picture owns its colours as frankly as a large one.
  - The discrete levels of `Intent` (`-large` … `+large`, recipes written before the sliders) keep their exact effect and add up with the retouch; the inspector shows them on the sliders (`retouch_from_intent`) and replaces a level when its slider is moved.
- **Conflicts.** `find_conflicts` reports pairs of zones whose Bhattacharyya coefficient exceeds 0.3 and whose displacements differ by more than 0.05. `harmonize_conflicts` merges the targets of conflicting `"auto"` links when the pair is loaded.

---

## 5. Offline AI Bridge & `.szl` Format (v2)

- **Tolerant extraction (`core/recipe.py:extract_json`).** Takes the **last** fenced `json` code block of a free-text answer, so the AI can reason before concluding.
- **Invariants of `parse_recipe` and `apply_recipe`.**
  - Supports `RECIPE_MODES = ("zones", "ambiance", "look")`.
  - In `"look"` mode, the `"links"` block is ignored silently, even when present or malformed.
  - If `recipe.analysis_id` differs from the SHA-256 fingerprint of the loaded pair: in `"look"` mode every setting (`separation`, `protect_vivid`, `locality`, `global`, `finishing`) is applied **except `look.focus`** (the ids `R1..Rn` no longer name the same regions), with a note in the status bar.
  - `merge_finishing` only overwrites the keys explicitly present in `recipe.finishing_keys`, so the user's wheels and curve survive when the AI only adjusts `vibrance` or `saturation`.
- **Zone retouch in a recipe.** A link may carry `"keep_tone"` (0 to 1) and `"retouch": {"exposure", "saturation", "warmth", "tint"}` (−1 to 1 each). `"intent"` with levels is still read and keeps its effect; both may be given together.
- **Zones pack (`core/ai_pack.py`).** When a render is on screen, `write_pack` adds `7_current_render.jpg` and, in `analysis.json` and in the prompt, one row per source zone: lightness and chroma in the source, in the current render and in its target, plus the current settings of its link (`zone_measures`). The prompt asks the AI to correct that render and explains `keep_tone`, `retouch` and the per-link `luminance_strength`.
- **Prompt guardrails (`core/look_pack.py`, `core/ai_pack.py`).**
  - Never advise the AI to lower `luminance_strength` or `chroma_strength` out of caution: the strengths multiply (`strength × chroma_strength × mix`).
  - The JSON example generated in `_PROMPT` (`example_recipe`) must always be checked by a unit test through `parse_recipe`.

---

## 6. Known-Truth Bench & Measured Findings

The module `src/stealzalut/devtools/ground_truth.py` evaluates the methods on 8 known synthetic grades `G` by measuring the **residual**:

```math
\mathrm{residual} = \frac{\lVert \mathrm{LUT}_{Q \to G(P)}(Q) - G(Q) \rVert_{\mathrm{Oklab}}}{\lVert Q - G(Q) \rVert_{\mathrm{Oklab}}}
```

0 means the grade is recovered exactly, 1 means no better than doing nothing, above 1 means worse than doing nothing.

Measured on 25 neutral photographs, 33³ LUT in Display P3. Each cell gives the range over the two lots (tuning and control).

| Regime | `look` (default) | `zones` | `oklab` | Finding |
|---|---|---|---|---|
| **Same content** (`Q = P`) | **0.06 – 0.07** | 0.08 – 0.12 | 0.18 – 0.20 | `look` recovers the tone curve and the chroma stretches almost exactly, with no fold on any pair. |
| **Other content** (`Q ≠ P`), locality 0 % | 4.1 – 6.3 | 4.1 – 6.3 | 4.2 – 5.8 | Two unrelated photographs differ about 3 times more from each other (0.156 Oklab) than the grade moves colours (0.056). Any alignment of distributions therefore copies the subject. |
| **Other content** (`Q ≠ P`), locality 100 % | **1.45 – 1.60** | — | — | The soft limit (`with_locality`) divides the over-transfer of the subject by 3 to 4, with no computing cost and no fold. |

> **Calibration rule.** Never retune the internal constants of `core/look.py` to lower the "other content" score at zero locality: it is a fundamental limit of statistical identifiability, not a defect of the constants. Any algorithmic change in `core/` must keep the "same content" residual and satisfy the oracles of `tests/test_look.py`.

---

## 7. Technical Pitfalls & Code Invariants

1. **Method counters in the tests.** `tests/test_color_transfer.py` and `tests/test_gui.py` check `len(SUPPORTED_METHODS) == 11` and `window._algo_combo.count() == 11`.
2. **Positional constructor of `EditState` (`gui/history.py`).** `EditState` is built positionally in several places. Any new field must be **added at the end, with a default value**.
3. **Qt warning `QFont::setPointSize <= 0`.** `theme.qss` sets font sizes in `px`. Never use `QPushButton.setMenu()` or `QFont.setPointSizeF()` without checking `pointSizeF() > 0`; open each `QMenu` by hand with `menu.popup(...)`, as done for `importRecipeButton`, `focusButton` and `lookRecipeButton`.
4. **The local `.env` file takes precedence.** `load_settings()` loads `.env`, which overrides the default in the code (`COLOR_TRANSFER_METHOD`). Interface tests that depend on a given algorithm must select it explicitly.
5. **Lifetime of `QThread` objects (`MainWindow._retire_transfer_worker`).** A `TransferWorker` replaced by a new request while it is running must never be destroyed by Python's garbage collector before it finishes (fatal Qt crash). Its signals are disconnected and a reference is kept in `_stale_transfer_workers` until the `finished` signal.
6. **A dark colour holds little chroma.** At L ≈ 0.25 a display shows a chroma of about 0.06 on skin hues. A saturation or warmth retouch on a dark zone is computed in full by the engine and then reduced by the final gamut compression: lift the zone first (`keep_tone`, `exposure`) to give its colour room.
