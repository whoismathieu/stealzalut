# StealZaLut

> **3D LUT (`.cube`) generator based on conditional optimal transport and perceptual analysis in unbounded Oklab.**
>
> A desktop application (Python 3.10+ / PySide6) that transfers the colour rendering of a reference photograph (in particular medium-format analog night photography) to a source photograph, while respecting the physics of colour spaces and the pipelines of the host software (Lightroom Classic, Adobe Camera Raw, DaVinci Resolve).

The interface is in French; the labels quoted below appear as they do on screen.

---

## 1. What a 3D LUT Can and Cannot Do

A 3D lookup table (`.cube`) is a purely colorimetric function from `[0, 1]³` to `[0, 1]³`, evaluated pixel by pixel with trilinear interpolation and **with no spatial information**: two regions with the same input colour always get the same output colour.

To avoid the "monochrome filter" effect and the crushed highlights typical of global statistical transfers, **StealZaLut** separates the analysis of the picture from its display rendition:

1. **Analysis in the native, unclipped gamut (`WideImage`).** Embedded ICC profiles (Display P3, Adobe RGB, ProPhoto) are decoded straight to the perceptual **Oklab** space in floating point (`float64`), which keeps the vivid colours (neons, signs) that an early 8-bit sRGB conversion would destroy.
2. **Tone and chroma handled separately (`look`).** Lightness `L` follows a monotone cubic curve (no tonal inversion is possible), while chroma `(a, b)` is transported tone band by tone band, towards the lightness the colour will actually have at the output.
3. **One baking chain (`build_grid`).** The model, the strengths, the protection of skin tones, neutrals and vivid colours, the fold control and the finishing are all evaluated in unbounded Oklab. Gamut compression at constant hue is applied only at the very last step, towards the target space of the `.cube`.

---

## 2. Key Features

### Colour transfer engines (11 methods)

- **Rendu (`look`, the default method)** reads the reference as a photographic print, in four parts:
  - *Monotone tone curve*: Fritsch–Carlson interpolation on the 1 % to 85 % quantiles for the bulk of the picture, then a jump to the white point at 99.9 %, so light sources keep their brightness whatever area they cover.
  - *Cast per tone band*: the mean displacement of `(a, b)` on 12 lightness bands.
  - *2×2 linear stretch (`stretch`)* and a *residual field on a companded lattice* (49×49, with `p = ab / (‖ab‖ + 0.05)`, which gives the finest resolution near the neutral axis), both computed by sliced optimal transport (48 projections).
  - *Dedicated controls*: **Séparation** (0 to 1.5); **Fidélité à ma photo**, the locality (0 to 1), a soft `tanh` limit that reduces the transfer of the reference's subject when source and reference show unrelated scenes; and **weighting by reference zone** (`Référence ▾`: ignore a frame, favour a sky).
- **Zones (`zones`)**: deterministic k-means segmentation in Oklab (K from 2 to 16, default 8), automatic matching by entropic optimal transport (Sinkhorn), and an interactive **correspondence table** editor (weighted links, per-zone locks, qualitative intents, detection of Bhattacharyya conflicts).
- **Ambiance (`ambiance`)**: for nearly monochrome references (hue concentration of 0.85 or more). Extracts a continuous tint curve along `L` and adds it to the source colours, while protecting vivid colours at 80 %.
- **Decoupled Oklab (`oklab`)**: linear Gaussian optimal transport, with a separate scale for lightness `L` and a 2×2 Monge–Kantorovich matrix on `(a, b)`.
- **Classic global methods (`color-matcher`)**: `mkl`, `reinhard`, `hm`, `mvgd`, `hm-mkl-hm`, `hm-mvgd-hm`, `default`.

### `.cube` export spaces and preview

- **Grid sizes**: 17³ (4 913 points), 33³ (35 937 points) or 64³ (262 144 points).
- **Colour spaces of the `.cube`**: `sRGB`, `Display P3`, `Adobe RGB (1998)`, `Rec.709 (gamma 2.4)`. ProPhoto RGB is excluded from the export: its imaginary primaries give negative LMS cone responses at corners of the `[0, 1]³` cube, which folds lightness on the neutral grid alone.
- **Preview faithful to the export**: the preview applies the 3D grid, computed in the chosen space, to the native pixels of the source picture by trilinear interpolation, and converts to sRGB for the screen only afterwards.
- **Full-size image export**: the graded source photograph can be exported directly as JPEG (quality 95), TIFF or PNG.

### Adjustment and finishing tools (Oklab / Oklch)

- **Global adjustments**: luminance strength, colour strength, neutral protection, skin protection (an Oklch window with full weight on hues 37.5°–77.5°, which never blocks `L`), vivid colour protection, 3D smoothing and mix.
- **Finishing panel**: 3 colour wheels (shadows, midtones, highlights, weighted by Bernstein polynomials on `L`), a monotone cubic tone curve on `L`, 8 Gaussian HSL hue bands, saturation and vibrance.

### Offline AI bridge and `.szl` recipes (v2)

- **No API, no network.** The application writes an analysis folder (`_stealzalut_look_pack` or `_stealzalut_pack`, plus a `.zip` archive) containing the photographs, the **current render** (`3_current_render.jpg` in Rendu mode), the zone plates, the measured gaps (`analysis.json`), and copies a ready-made prompt to the clipboard.
- **Tolerant import.** Paste the full answer of your AI assistant (Claude, ChatGPT, Gemini): StealZaLut extracts and validates the last `.szl` JSON block, merges the finishing, and detects over-cautious strengths (« Doser à 100 % »).

---

## 3. Lightroom Classic / Camera Raw and RAW Files

When a RAW/DNG file is loaded directly, StealZaLut reads the JPEG preview embedded by the camera (often in Display P3, with the manufacturer's tone curve already applied). In Adobe Camera Raw and Lightroom, however:

1. An `.xmp` profile created from a `.cube` is built on the **Adobe Standard** base (not *Adobe Color* or *Apple ProRAW*).
2. The 3D LUT is evaluated **after** the base profile and the tone curve, in the colour space selected when the profile was created.

**For an exact match between the StealZaLut preview and Lightroom:**

1. In Lightroom or Camera Raw, apply the **Adobe Standard** profile to your RAW file with every slider at zero.
2. Export an intermediate copy (JPEG or TIFF, ideally in **Display P3**).
3. Use this export as the **source image** in StealZaLut and export your `.cube` in the **Display P3** space.
4. In Camera Raw, create your profile with **Display P3** selected as the space of the colour table, then apply that profile directly to your RAW files.

---

## 4. Installation & Quick Start

### Requirements

- **Python 3.10+** and the [`uv`](https://docs.astral.sh/uv/) package manager.

### Installation (Windows PowerShell)

```powershell
uv sync --all-extras
if (!(Test-Path .env)) { Copy-Item .env.example .env }
```

### Running the application

```powershell
# From the local virtual environment
.venv\Scripts\python.exe -m stealzalut.main

# Or with uv, or with the launch script
uv run stealzalut
.\launch.bat
```

---

## 5. Tests, Code Quality & Bench

Every change must keep the three checks below passing:

```bash
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m ruff check src tests
.venv/Scripts/python.exe -m mypy src
```

To measure the transfer algorithms on a folder of pictures with the **known-truth bench** (floating-point Display P3, lots `reglage` for tuning and `controle` for control):

```bash
PYTHONPATH=src .venv/Scripts/python.exe -m stealzalut.devtools.ground_truth <picture_folder> --methods look,zones,oklab
```

---

## 6. Technical Documentation

The internal architecture, the equations, the handling of colour spaces and the development invariants are described in **[ARCHITECTURE.md](ARCHITECTURE.md)**.
