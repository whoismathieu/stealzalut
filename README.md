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
- **Zones (`zones`)**: deterministic k-means segmentation in Oklab (K from 2 to 16, default 8), automatic matching by entropic optimal transport (Sinkhorn), and an interactive **correspondence table** editor (weighted links, per-zone locks, detection of Bhattacharyya conflicts). Each zone also has its own **retouch sliders**, applied on top of its transfer and never attenuated by the global adjustments: **Garder ma lumière** (the zone keeps the lightness it has in your photo, for a skin or a subject that came out too dark), **Exposition**, **Saturation**, **Température** and **Teinte**.
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
- **Images from the web or the clipboard**: a picture can be dragged straight from a browser onto the source or the reference, or pasted with Ctrl+V (or « Coller une image » in the right-click menu of a zone) after « Copy image » or « Copy image address ». Ctrl+V fills the zone under the pointer, else the first empty one. The original file is fetched from its address when there is one, so that its colour profile is kept; otherwise the copied picture itself is used. It is written to the system's temporary folder and deleted when the window closes. This is the only moment the application uses the network, and only for the address you dropped or pasted; a site that refuses is reported in the status bar (save the picture and drop the file instead). JPEG, PNG, WebP, TIFF, BMP and HEIC are accepted, AVIF is not.
- **Window**: the three images (source, reference, preview) zoom with the mouse wheel, move with a drag and come back to the whole picture with a click on the wheel; source and preview stay on the same area. A right click on an image opens a menu with the grey behind the three images, the six levels Lightroom offers (Blanc, Gris clair, Gris moyen, Gris foncé, Gris très foncé, Noir). The Zones section and the Ajustements / Finition tabs fold down to their header with the ▾ arrow and are resized by dragging the separators. The toolbars wrap on two rows when the window is narrow.
- **Appearance**: the « Apparence ▾ » button at the bottom right chooses the theme (Graphite by default, Noir, Darkroom) and the grey behind the images; both choices are remembered. A theme is one small JSON file that gives a value to the colours and radii of the single stylesheet:

  ```json
  {"name": "Mon thème", "tokens": {"accent": "#e8a04c", "bg": "#1a1816"}}
  ```

  It only lists what it changes; everything else comes from Graphite (`src/stealzalut/gui/styles/themes/graphite.json` holds the full list of tokens). Drop the file in `src/stealzalut/gui/styles/themes/`, or in the `themes` folder of the application's configuration directory (`%LOCALAPPDATA%\StealZaLut\StealZaLut\themes` on Windows) to add one without touching the code: it appears in the menu at the next opening. `APP_THEME` in `.env` sets the theme used before any choice in the window. The zone diagram, the colour wheels and the tone curve are drawn with dark colours of their own: a light theme would need them adapted.

### Offline AI bridge and `.szl` recipes (v2)

- **No API, no network.** The application writes an analysis folder (`_stealzalut_look_pack` or `_stealzalut_pack`, plus a `.zip` archive) containing the photographs, the **current render** (`3_current_render.jpg` in Rendu mode), the zone plates, the measured gaps (`analysis.json`), and copies a ready-made prompt to the clipboard. In Zones mode the pack also holds the current render (`7_current_render.jpg`) and, per zone, its lightness and chroma in the source, in the render and in its target, so the AI corrects what you see.
- **Tolerant import.** Paste the full answer of your AI assistant (Claude, ChatGPT, Gemini): StealZaLut extracts and validates the last `.szl` JSON block, merges the finishing, and detects over-cautious strengths (« Doser à 100 % »).

---

## 3. Lightroom Classic / Camera Raw and RAW Files

A RAW/DNG file dropped as the source is **developed by StealZaLut the way Lightroom develops it on Adobe Standard with every slider at zero**. For a DNG whose camera Adobe knows, the application reads Adobe's own « Adobe Standard » profile of that camera, the `.dcp` file Lightroom or Camera Raw installed on the machine (`C:\ProgramData\Adobe\CameraRaw\CameraProfiles\Adobe Standard` on Windows, `/Library/Application Support/Adobe/CameraRaw/CameraProfiles/Adobe Standard` on macOS), and applies the steps of Adobe's DNG SDK in its order:

1. demosaicing by LibRaw into the camera's own channels, 16-bit linear, no white balance, no automatic brightening, and the white level of the file left where it is;
2. the profile's matrices, mixed for the white balance of the shot (the temperature found is the one Lightroom displays for « As Shot »);
3. the profile's hue and saturation table;
4. the file's `BaselineExposure`, as a plain gain clipped at white, then a fixed black point (0.43 % of the range) on each channel;
5. the profile's « look » table;
6. Adobe's default base tone curve (or the profile's own, when it has one), applied at constant hue.

Nothing is measured on the picture: a night scene stays a night scene. The profile is read where Adobe put it and is never copied. **Without it** (another RAW format, a camera Adobe has no profile for, a machine without Lightroom or Camera Raw) the file is developed with its own matrices and the same exposure, black point and curve, which is close but less exact; the banner of the window says which of the two was used.

The result enters the engine unclipped and is handed to the LUT in Display P3. The JPEG embedded by the camera (with the manufacturer's look: local tone mapping, lifted shadows, extra saturation) is only used when the file cannot be demosaiced.

**How close this is to Lightroom.** Measured on two iPhone 15 Pro Max ProRAW night scenes (`BaselineExposure` +1.6 and +2.4 EV), 400 000 pixels each, against Lightroom Classic 15.6 on Adobe Standard with every slider at zero (16-bit TIFF, Display P3). The first is compared with its plain export; the second with its export through a profile made from a `.cube`, the application's development going through the same `.cube`:

| Source read by StealZaLut | File | Mean lightness gap | Typical colour gap (median ΔE Oklab) | RMS ΔE | Saturation, by lightness band |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Adobe's profile (current)** | first | −0.001 | 0.005 | 0.016 | 1.01 to 1.04 × Lightroom |
| | second, through the `.cube` | +0.004 | 0.006 | 0.016 | 0.86 to 1.07 × |
| The file's own matrices (fallback) | first | +0.001 | 0.005 | 0.022 | 0.93 to 1.60 × |
| | second, through the `.cube` | +0.001 | 0.007 | 0.017 | 0.87 to 1.28 × |
| Same, before the white level was fixed | second, through the `.cube` | **+0.055** | 0.052 | 0.068 | 0.74 to 1.65 × |
| JPEG embedded by the phone | first | +0.007 | 0.039 | 0.077 | 1.24 to 1.39 × |

The fifth row is a defect that was corrected: LibRaw moved the white level of each file to its brightest pixel, which brightened the second photo by 0.21 EV and the first by nothing. A steep `.cube` turns such a gap into a visible one.

What remains (about 0.014 of the RMS gap) is what Lightroom does area by area and what no development of a pixel alone, hence no `.cube`, reproduces: sharpening, noise reduction, and the way it holds the brightest highlights. The gain table map of a ProRAW file is ignored, as Lightroom does outside Apple's own profile. Other cameras have not been measured.

In Adobe Camera Raw and Lightroom:

1. An `.xmp` profile created from a `.cube` is built on the **Adobe Standard** base (not *Adobe Color* or *Apple ProRAW*).
2. The 3D LUT is evaluated **after** the base profile and the tone curve, in the colour space selected when the profile was created.

**The « Créer un profil » window of Camera Raw, field by field.** Open it from the Presets tab with Alt + click on the new-preset button, on a photo set to **Adobe Standard** with every slider at zero. The application shows these values in the banner of a RAW source and again when the export ends, with the space of the `.cube` you chose:

| Field | Value | Why |
| :--- | :--- | :--- |
| Checkboxes of the upper list | none | The profile must hold the colour table only |
| Intensité de mappage de la tonalité | **Faible (normal)** | The other value is meant for HDR pictures |
| Table de correspondance | unchecked | Not used |
| Table de recherche des couleurs | checked, your `.cube` | |
| Espace | **the space of the `.cube`** (« Espace du .cube » in StealZaLut) | In another space the colours are not those of the preview |
| Gamut | **Écrêtage** | Colours outside the space are brought into it before the table, as in the preview |
| Échantillons | **32** | The maximum |
| Min / Niveau / Max | **0 / 100 / 200 %** | 100 % is the `.cube` as exported; Min and Max bound the profile's amount slider |

Camera Raw offers sRGB, Display P3, Adobe RGB, ProPhoto RGB and Rec. 2020 there: a `.cube` exported in Rec.709 gamma 2.4 has no matching entry. For a RAW source, export in **Display P3**. The space is what the result depends on; the Gamut value is the one consistent with how the `.cube` is built, and has not been compared with the other one in Camera Raw.

**Direct workflow (RAW in, `.cube` out):** drop the RAW/DNG as the source, export the `.cube` in **Display P3**, create the profile in Camera Raw with **Display P3** as the space of the colour table, and apply it on Adobe Standard with the sliders at zero.

**For an exact match between the StealZaLut preview and Lightroom** (the reference workflow, which removes the uncertainty above):

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
