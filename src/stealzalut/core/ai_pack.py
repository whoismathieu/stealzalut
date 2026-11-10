"""Analysis pack for the user's own AI (Claude, ChatGPT, Gemini…).

The app measures; the user's AI reasons about *what* each zone is and
which reference zone should colour it.  The pack gives the AI everything
it needs, in a folder (plus a ``.zip`` of it):

* ``1_source_zones.png`` / ``2_reference_zones.png`` — one tile per zone,
  the zone in colour, the rest greyed, labelled ``S1``… / ``R1``…;
* ``3_source_overview.png`` / ``4_reference_overview.png`` — whole image,
  zone outlines and labels;
* ``analysis.json`` — exact measurements and the automatic matching;
* ``PROMPT.md`` — the prompt to paste, asking for a ``.szl`` recipe.

The app never calls any AI service: no key, no network.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from stealzalut.core.colorspace import oklab_to_oklch
from stealzalut.core.palette import ZonePalette, upsample_mask
from stealzalut.core.recipe import RECIPE_FORMAT, RECIPE_VERSION
from stealzalut.core.zone_transfer import ZoneTransferModel
from stealzalut.utils.image_io import create_thumbnail

ANALYSIS_FORMAT = "stealzalut.analysis"
ANALYSIS_VERSION = 1
#: Longest edge of every image of the pack.
PACK_MAX_SIZE = 1024
TILE_SIZE = 240
_TILE_GAP = 8
_LABEL_H = 34
_SHEET_BG = (22, 22, 26)
_FONT = cv2.FONT_HERSHEY_SIMPLEX

PACK_FILES = (
    "1_source_zones.png",
    "2_reference_zones.png",
    "3_source_overview.png",
    "4_reference_overview.png",
    "analysis.json",
    "PROMPT.md",
)


# ── analysis.json ────────────────────────────────────────────────────────────


def _lightness_band(lightness: float) -> str:
    if lightness < 0.45:
        return "shadows"
    if lightness < 0.72:
        return "midtones"
    return "highlights"


def _zones_json(palette: ZonePalette) -> list[dict[str, Any]]:
    zones = []
    for zone in palette.zones:
        lightness, chroma, hue = (float(v) for v in oklab_to_oklch(zone.mean[None, :])[0])
        zones.append(
            {
                "id": zone.id,
                "hex": zone.hex,
                "area_pct": round(zone.area * 100.0, 1),
                "oklch": [round(lightness, 3), round(chroma, 3), round(hue, 1)],
                "lightness_band": _lightness_band(float(zone.mean[0])),
                "centroid_pct": [round(v, 1) for v in zone.centroid_xy],
                "bbox_pct": [round(v, 1) for v in zone.bbox],
            }
        )
    return zones


def analysis_dict(
    model: ZoneTransferModel,
    source_size: tuple[int, int],
    reference_size: tuple[int, int],
    zone_count: int,
) -> dict[str, Any]:
    """Content of ``analysis.json``.

    Args:
        model: Freshly fitted zones model (its table is the automatic matching).
        source_size: ``(width, height)`` of the source image.
        reference_size: ``(width, height)`` of the reference image.
        zone_count: Zone count requested for the analysis.
    """
    return {
        "format": ANALYSIS_FORMAT,
        "version": ANALYSIS_VERSION,
        "analysis_id": model.analysis_id,
        "zone_count": zone_count,
        "source": {"size": list(source_size), "zones": _zones_json(model.source)},
        "reference": {"size": list(reference_size), "zones": _zones_json(model.reference)},
        "auto_matching": [
            {
                "source_zone": link.source_zone,
                "targets": {str(k): round(w, 3) for k, w in sorted(link.targets.items())},
            }
            for link in model.table.links
        ],
    }


# ── Images ───────────────────────────────────────────────────────────────────


def _hex_rgb(hex_color: str) -> tuple[int, int, int]:
    value = hex_color.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def _ink(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    luma = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    return (20, 20, 24) if luma > 150 else (245, 245, 248)


def _fit(image: NDArray[np.uint8], max_size: int = PACK_MAX_SIZE) -> NDArray[np.uint8]:
    h, w = image.shape[:2]
    scale = max_size / max(h, w)
    if scale >= 1.0:
        return image
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return np.asarray(cv2.resize(image, size, interpolation=cv2.INTER_AREA), dtype=np.uint8)


def _display_labels(palette: ZonePalette, width: int, height: int) -> NDArray[np.uint8]:
    """Zone map at display size, median-filtered so outlines follow shapes, not noise."""
    labels = cv2.resize(
        palette.labels.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
    )
    ksize = max(3, (min(height, width) // 60) | 1)
    return np.asarray(cv2.medianBlur(labels, ksize), dtype=np.uint8)


def _outline(image: NDArray[np.uint8], mask: NDArray[np.uint8], thickness: int = 1) -> None:
    """Draw the outline of *mask* on *image*, skipping specks."""
    h, w = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 0.0015 * h * w
    kept = [c for c in contours if cv2.contourArea(c) >= min_area]
    cv2.drawContours(image, kept, -1, (240, 240, 244), thickness, lineType=cv2.LINE_AA)


def zone_sheet(
    image_rgb: NDArray[np.uint8],
    palette: ZonePalette,
    prefix: str,
    columns: int = 4,
) -> NDArray[np.uint8]:
    """Contact sheet: one tile per zone, the zone in colour, the rest greyed.

    Args:
        image_rgb: The analysed image, RGB uint8.
        palette: Its zones.
        prefix: ``"S"`` or ``"R"``.
        columns: Tiles per row.

    Returns:
        RGB uint8 image, longest edge at most :data:`PACK_MAX_SIZE`.
    """
    small = create_thumbnail(image_rgb, TILE_SIZE)
    h, w = small.shape[:2]
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32)
    greyed = np.repeat((gray * 0.35 + 18.0)[..., None], 3, axis=2).astype(np.uint8)

    labels = _display_labels(palette, w, h)
    tiles: list[NDArray[np.uint8]] = []
    for zone in palette.zones:
        # Colour follows the exact zone; the outline follows its smoothed shape
        exact = upsample_mask(palette.labels == zone.id, w, h)
        tile = np.where(exact[..., None], small, greyed).astype(np.uint8)
        _outline(tile, (labels == zone.id).astype(np.uint8))

        bar = np.full((_LABEL_H, w, 3), 34, dtype=np.uint8)
        color = _hex_rgb(zone.hex)
        cv2.rectangle(bar, (6, 6), (28, 28), color, thickness=-1)
        cv2.rectangle(bar, (6, 6), (28, 28), (200, 200, 205), thickness=1)
        text = f"{prefix}{zone.id}   {zone.area * 100:.0f}%"
        cv2.putText(bar, text, (38, 24), _FONT, 0.6, (240, 240, 244), 1, cv2.LINE_AA)
        tiles.append(np.vstack([tile, bar]))

    columns = max(1, min(columns, len(tiles)))
    rows = -(-len(tiles) // columns)
    tile_h, tile_w = h + _LABEL_H, w
    sheet = np.full(
        (rows * tile_h + (rows + 1) * _TILE_GAP, columns * tile_w + (columns + 1) * _TILE_GAP, 3),
        _SHEET_BG,
        dtype=np.uint8,
    )
    for i, tile in enumerate(tiles):
        r, c = divmod(i, columns)
        y = _TILE_GAP + r * (tile_h + _TILE_GAP)
        x = _TILE_GAP + c * (tile_w + _TILE_GAP)
        sheet[y : y + tile_h, x : x + tile_w] = tile
    return _fit(sheet)


def zone_overview(
    image_rgb: NDArray[np.uint8],
    palette: ZonePalette,
    prefix: str,
) -> NDArray[np.uint8]:
    """Whole image, darkened, with every zone outlined and labelled.

    Each label sits at the most interior point of the zone's largest
    connected part, so it is always inside the zone.
    """
    image = create_thumbnail(image_rgb, PACK_MAX_SIZE)
    h, w = image.shape[:2]
    labels = _display_labels(palette, w, h)
    out = (image.astype(np.float32) * 0.6).astype(np.uint8)

    placements: list[tuple[str, tuple[int, int], tuple[int, int, int]]] = []
    for zone in palette.zones:
        mask = (labels == zone.id).astype(np.uint8)
        if not mask.any():
            continue
        _outline(out, mask, thickness=2 if max(h, w) > 600 else 1)
        count, components, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if count < 2:
            continue
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        part = (components == largest).astype(np.uint8)
        distance = cv2.distanceTransform(part, cv2.DIST_L2, 5)
        row, col = np.unravel_index(int(np.argmax(distance)), distance.shape)
        placements.append((f"{prefix}{zone.id}", (int(col), int(row)), _hex_rgb(zone.hex)))

    scale = 0.55 if max(h, w) <= 600 else 0.7
    for text, (x, y), color in placements:
        (tw, th), base = cv2.getTextSize(text, _FONT, scale, 2)
        x0 = int(np.clip(x - tw // 2 - 5, 0, max(0, w - tw - 10)))
        y0 = int(np.clip(y - th // 2 - 5, 0, max(0, h - th - base - 10)))
        cv2.rectangle(out, (x0, y0), (x0 + tw + 10, y0 + th + base + 10), color, thickness=-1)
        cv2.rectangle(out, (x0, y0), (x0 + tw + 10, y0 + th + base + 10), (255, 255, 255), 1)
        cv2.putText(
            out, text, (x0 + 5, y0 + th + 5), _FONT, scale, _ink(color), 2, cv2.LINE_AA
        )
    return out


# ── Prompt ───────────────────────────────────────────────────────────────────


_PROMPT = """\
# StealZaLut — Correspondance de couleurs entre deux images

Tu es étalonneur (coloriste) cinéma. Je veux donner à ma photo SOURCE l'ambiance
colorimétrique de ma photo RÉFÉRENCE, même si elles ne montrent pas la même chose.

Mon logiciel a découpé chaque image en zones de couleur. Fichiers joints :
- 1_source_zones.png : chaque zone de la source isolée, nommée {source_ids}.
- 2_reference_zones.png : chaque zone de la référence, nommée {reference_ids}.
- 3_source_overview.png et 4_reference_overview.png : les images entières, zones entourées.
- analysis.json : mesures exactes et proposition automatique (auto_matching).

## Ta mission
1. Décris brièvement ce que représente chaque zone (ex. « S3 : feuillage en contre-jour »).
2. Pour chaque zone source, choisis quelle(s) zone(s) de la référence doivent lui
   donner sa couleur, en raisonnant sur le RÔLE et l'AMBIANCE (lumière principale,
   ombres, matière, sujet), pas sur la ressemblance de couleur. Tu peux répartir sur
   plusieurs zones avec des poids, ou garder la zone telle quelle (`"keep": true`).
3. Protège ce qui doit rester crédible : peau, éléments de marque, blancs.
4. Ajuste avec des intentions qualitatives si besoin (plus chaud, moins saturé…).
5. Corrige l'auto_matching seulement là où il est mauvais.

## Dosage : les forces se MULTIPLIENT
La couleur reçue par une zone = strength × global.chroma_strength × global.mix,
puis preserve_neutrals en retire encore une part sur les tons peu saturés.
Exemple : "medium" (60 %) × chroma_strength 0.75 × mix 0.85 ≈ 38 % seulement :
le look disparaît presque. Donc :
- La valeur normale est le PLEIN effet : strength "full", chroma_strength 1.0,
  mix 1.0. Ne baisse que là où c'est voulu (une zone précise à adoucir), jamais
  partout « par prudence ».
- Barème de strength : off = 0 %, subtle = 30 %, medium = 60 %, strong = 85 %,
  full = 100 %.
- preserve_neutrals : laisse-le à 0 en général. Sur une photo peu saturée il
  bride presque tout l'effet. Pour protéger des blancs ou des gris précis,
  utilise plutôt "keep": true sur leur zone.

## Luminosité et intensité du look
- global.luminance_strength = part de la luminosité et du contraste de la
  référence que prend la photo (1.0 = toute). C'est une grande partie de
  l'intensité d'un look : une référence sombre et dense ne se retrouve que si
  la photo s'assombrit avec elle. Garde 1.0 par défaut ; baisse-le seulement si
  les éclairages sont incompatibles (jour vs nuit, intérieur vs extérieur, flash
  vs lumière naturelle), pour ne garder que la couleur.
- Le transfert par zones n'atteint en général qu'environ 75–80 % de la
  saturation de la référence (les tons très sombres portent peu de couleur).
  Si la référence est nettement plus intense ou plus saturée que la source,
  compense dans "finishing" : vibrance (ravive surtout les couleurs ternes)
  et/ou saturation (toutes les couleurs), de -1.0 à 1.0, 0 = neutre.
  Ordres de grandeur : référence intense → vibrance 0.3 à 0.6 et saturation
  0.1 à 0.2 ; référence douce ou délavée → valeurs négatives.
- Dans "finishing", ne donne QUE "saturation" et "vibrance" (les roues et la
  courbe restent à l'utilisateur). Mets "finishing": null si rien n'est à compenser.

## Règles
- N'invente AUCUNE valeur de couleur (pas de RGB, hex, lift, gamma…).
- Utilise uniquement les identifiants présents dans analysis.json :
  sources {source_ids}, références {reference_ids}.
- Échelles : strength ∈ off | subtle | medium | strong | full ;
  intentions possibles : lightness (plus clair / sombre), saturation, warmth
  (+ plus chaud, - plus froid), tint (+ magenta, - vert), chacune ∈
  -large | -medium | -small | 0 | +small | +medium | +large.
- Raisonne d'abord librement, puis termine par UN SEUL bloc ```json respectant
  exactement ce format (garde analysis_id tel quel) :

```json
{{ "format": "{recipe_format}", "version": {recipe_version}, "analysis_id": "{analysis_id}",
  "name": "...", "description": "...",
  "global": {{ "luminance_strength": 1.0, "chroma_strength": 1.0,
              "preserve_neutrals": 0.0, "skin_protection": 0.0-1.0,
              "zone_separation": "sharp|medium|soft", "mix": 1.0 }},
  "links": [ {{ "source_zone": 1, "targets": {{ "2": 0.7, "4": 0.3 }},
               "strength": "full", "intent": {{ "warmth": "+small" }},
               "rationale": "..." }},
             {{ "source_zone": 2, "keep": true, "rationale": "..." }} ],
  "finishing": {{ "saturation": 0.0, "vibrance": 0.0 }} }}
```
"""


def _ids(prefix: str, palette: ZonePalette) -> str:
    return ", ".join(f"{prefix}{zone.id}" for zone in palette.zones)


def build_prompt(model: ZoneTransferModel) -> str:
    """``PROMPT.md`` for this analysis (ids and analysis id filled in)."""
    return _PROMPT.format(
        source_ids=_ids("S", model.source),
        reference_ids=_ids("R", model.reference),
        recipe_format=RECIPE_FORMAT,
        recipe_version=RECIPE_VERSION,
        analysis_id=model.analysis_id,
    )


# ── Writing ──────────────────────────────────────────────────────────────────


def _write_png(path: Path, image_rgb: NDArray[np.uint8]) -> None:
    # cv2.imwrite can't write to non-ASCII paths on Windows: encode, then write bytes
    ok, data = cv2.imencode(".png", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise OSError(f"Impossible d'encoder {path.name}")
    path.write_bytes(data.tobytes())


def write_pack(
    destination: Path,
    name: str,
    source_rgb: NDArray[np.uint8],
    reference_rgb: NDArray[np.uint8],
    model: ZoneTransferModel,
    zone_count: int,
) -> Path:
    """Write the pack folder ``<destination>/<name>_stealzalut_pack/`` and its ``.zip``.

    Args:
        destination: Existing folder to write into.
        name: Base name of the pack.
        source_rgb: Source image, RGB uint8.
        reference_rgb: Reference image, RGB uint8.
        model: Freshly fitted zones model (automatic table).
        zone_count: Zone count requested for the analysis.

    Returns:
        The pack folder (the zip sits next to it, same name + ``.zip``).
    """
    folder = destination / f"{name}_stealzalut_pack"
    folder.mkdir(parents=True, exist_ok=True)

    _write_png(folder / PACK_FILES[0], zone_sheet(source_rgb, model.source, "S"))
    _write_png(folder / PACK_FILES[1], zone_sheet(reference_rgb, model.reference, "R"))
    _write_png(folder / PACK_FILES[2], zone_overview(source_rgb, model.source, "S"))
    _write_png(folder / PACK_FILES[3], zone_overview(reference_rgb, model.reference, "R"))

    analysis = analysis_dict(
        model,
        (int(source_rgb.shape[1]), int(source_rgb.shape[0])),
        (int(reference_rgb.shape[1]), int(reference_rgb.shape[0])),
        zone_count,
    )
    (folder / PACK_FILES[4]).write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (folder / PACK_FILES[5]).write_text(build_prompt(model), encoding="utf-8")

    shutil.make_archive(str(folder), "zip", root_dir=folder)
    return folder
