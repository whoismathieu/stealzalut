"""Pack for the user's AI, look method: the AI sees the current render.

With the zones method the AI writes its recipe blind.  Here the engine has
already copied the reference's tone, casts and amount of colour; the pack
shows the AI **the render as it is now** next to the reference, with a few
measured gaps, and asks for corrections in a colourist's words.  Its answer
is a ``.szl`` recipe with ``"mode": "look"`` (see
:mod:`stealzalut.core.recipe`); a new pack can then be made from the new
render, for another round.

Still no network and no API: files and clipboard only.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from stealzalut.core.ai_pack import (
    ANALYSIS_FORMAT,
    ANALYSIS_VERSION,
    ORIGINAL_MAX_SIZE,
    _colours_json,
    _write_jpeg,
    _write_png,
    _zones_json,
    zone_overview,
    zone_sheet,
)
from stealzalut.core.colorspace import gamut_compress, oklab_to_srgb, srgb_to_oklab
from stealzalut.core.palette import COLOURFUL_CHROMA, ZonePalette, colour_stats
from stealzalut.core.recipe import RECIPE_FORMAT, RECIPE_VERSION
from stealzalut.utils.image_io import create_thumbnail

F64 = NDArray[np.float64]

LOOK_PACK_FILES = (
    "1_source.jpg",
    "2_reference.jpg",
    "3_current_render.jpg",
    "4_reference_zones.png",
    "5_reference_overview.png",
    "analysis.json",
    "PROMPT.md",
)

#: Longest edge of the copies the gaps are measured on.
GAPS_SIZE = 640
#: Tone ranges (Oklab L) whose mean colour is compared.
TONE_RANGES: tuple[tuple[str, str, float, float], ...] = (
    ("shadows", "ombres", 0.0, 0.35),
    ("midtones", "tons moyens", 0.35, 0.65),
    ("highlights", "hautes lumières", 0.65, 1.01),
)
#: Share of pixels below which a tone range is too thin to describe.
MIN_RANGE_SHARE = 0.01


def _measure(image_rgb: NDArray[np.uint8]) -> dict[str, Any]:
    """Tone and colour of one image: means, coloured share, mean colour per range."""
    small = create_thumbnail(image_rgb, GAPS_SIZE)
    lab = srgb_to_oklab(small.reshape(-1, 3).astype(np.float64) / 255.0)
    chroma = np.hypot(lab[:, 1], lab[:, 2])
    out: dict[str, Any] = {
        "lightness_mean": round(float(lab[:, 0].mean()), 3),
        "chroma_mean": round(float(chroma.mean()), 3),
        "colourful_share": round(float(np.mean(chroma >= COLOURFUL_CHROMA)), 3),
    }
    for key, _label, lo, hi in TONE_RANGES:
        inside = (lab[:, 0] >= lo) & (lab[:, 0] < hi)
        if inside.mean() < MIN_RANGE_SHARE:
            out[key] = None
            continue
        mean = lab[inside].mean(axis=0)
        rgb = oklab_to_srgb(gamut_compress(mean[None, :]))[0]
        red, green, blue = (round(float(c) * 255.0) for c in rgb)
        out[key] = {
            "share": round(float(inside.mean()), 3),
            "hex": f"#{red:02x}{green:02x}{blue:02x}",
            "ab": [round(float(mean[1]), 3), round(float(mean[2]), 3)],
        }
    return out


def look_gaps(render_rgb: NDArray[np.uint8], reference_rgb: NDArray[np.uint8]) -> dict[str, Any]:
    """The render and the reference measured the same way, side by side.

    Returns:
        ``{"render": {...}, "reference": {...}}``; each side holds
        ``lightness_mean``, ``chroma_mean``, ``colourful_share`` and, per
        tone range (``shadows``, ``midtones``, ``highlights``), its share of
        the image and its mean colour (``hex`` and Oklab ``ab``), or ``None``
        when the image has almost no pixel there.
    """
    return {"render": _measure(render_rgb), "reference": _measure(reference_rgb)}


def gaps_table(gaps: dict[str, Any]) -> str:
    """The gaps as a Markdown table for the prompt."""

    def colour(side: dict[str, Any], key: str) -> str:
        entry = side[key]
        if entry is None:
            return "presque absents"
        return f"{entry['hex']} ({round(entry['share'] * 100)} % de l'image)"

    render, reference = gaps["render"], gaps["reference"]
    rows = [
        "| Mesure | Rendu actuel | Référence |",
        "|---|---|---|",
        f"| Luminosité moyenne (0 à 1) | {render['lightness_mean']:.3f} "
        f"| {reference['lightness_mean']:.3f} |",
        f"| Quantité de couleur (chroma moyenne) | {render['chroma_mean']:.3f} "
        f"| {reference['chroma_mean']:.3f} |",
        f"| Part de pixels colorés | {round(render['colourful_share'] * 100)} % "
        f"| {round(reference['colourful_share'] * 100)} % |",
    ]
    for key, label, _lo, _hi in TONE_RANGES:
        rows.append(
            f"| Couleur moyenne des {label} | {colour(render, key)} | {colour(reference, key)} |"
        )
    return "\n".join(rows)


def look_analysis_dict(
    reference_rgb: NDArray[np.uint8],
    reference: ZonePalette,
    analysis_id: str,
    separation: float,
    protect_vivid: float,
    focus: tuple[tuple[int, float], ...],
    gaps: dict[str, Any],
) -> dict[str, Any]:
    """Content of ``analysis.json`` for the look method."""
    return {
        "format": ANALYSIS_FORMAT,
        "version": ANALYSIS_VERSION,
        "analysis_id": analysis_id,
        "mode": "look",
        "reference": {
            "size": [int(reference_rgb.shape[1]), int(reference_rgb.shape[0])],
            **_colours_json(colour_stats(reference_rgb)),
            "zones": _zones_json(reference),
        },
        "look": {
            "separation": separation,
            "protect_vivid": protect_vivid,
            "focus": {f"R{zone}": weight for zone, weight in focus},
            "gaps": gaps,
        },
    }


_PROMPT = """\
# StealZaLut — Corriger un rendu (mode « Rendu »)

Tu es étalonneur (coloriste). Je veux donner à ma photo l'ambiance colorimétrique
d'une photo de référence, même si elles ne montrent pas la même chose.

Fichiers joints :
- 1_source.jpg : ma photo d'origine.
- 2_reference.jpg : l'ambiance à atteindre.
- 3_current_render.jpg : ce que mon logiciel produit **maintenant** à partir de
  ma photo. C'est cette image que tu corriges.
- 4_reference_zones.png et 5_reference_overview.png : la référence découpée en
  zones de couleur, nommées {reference_ids}.
- analysis.json : les mêmes mesures que ci-dessous, en détail.

Le logiciel a déjà copié automatiquement la courbe de tons de la référence, la
dominante de chaque tranche de tons et sa quantité de couleur. Ton rôle : comparer
3_current_render.jpg à 2_reference.jpg **à l'œil** et dire quoi corriger.

Mesures (elles aident, mais ton œil prime : les deux photos n'ont pas le même sujet) :

{gaps}

## Ce que tu peux régler

- `look.separation` (0 à 1.5 ; actuellement {separation:g}) : écarte les couleurs
  proches du gris vers les couleurs de la référence. Plus haut : couleurs plus
  tranchées. Plus bas : seulement la dominante de chaque ton.
- `look.protect_vivid` (0 à 1 ; actuellement {protect_vivid:g}) : garde leur teinte
  aux néons, enseignes et couleurs vives de ma photo. Plus haut : ma photo garde
  ses propres couleurs vives. Plus bas : elles prennent celles de la référence.
- `look.focus` : poids des zones de la référence ({reference_ids}) dans ce que le
  logiciel y lit. 0 : zone ignorée (un cadre, un sujet sans rapport avec
  l'ambiance). 2 : zone privilégiée. Ne cite que les zones dont tu changes le poids.
  Actuellement : {focus}.
- `global.luminance_strength`, `global.chroma_strength`, `global.mix` (0 à 1) :
  dose du ton, de la couleur, de l'ensemble.
- `finishing.saturation`, `finishing.vibrance` (-1 à 1).
- `finishing.shadows`, `finishing.midtones`, `finishing.highlights` : teinter une
  tranche de tons. `hue` est la teinte en degrés (0 rose, 30 rouge, 55 orange,
  110 jaune, 140 vert, 195 cyan, 265 bleu, 295 violet, 330 magenta), `amount` la
  force (0 à 1), `lightness` plus clair ou plus sombre (-1 à 1).

## Règles

- Ne corrige que ce que tu **vois** en comparant les deux images.
- Ne cite que les réglages que tu changes : ceux que tu omets restent comme ils sont.
- Mon logiciel applique une seule transformation de couleur à toute l'image : deux
  régions de même couleur ne peuvent pas être traitées différemment. Si c'est ce
  qui manque, dis-le dans `description` plutôt que de forcer un réglage.
- Raisonne d'abord librement, puis termine par UN SEUL bloc ```json de cette forme
  (garde `format`, `version`, `analysis_id` et `mode` tels quels). Les valeurs de
  cet exemple sont arbitraires : elles montrent la forme, ne les recopie pas.

```json
{example}
```
"""


def example_recipe(analysis_id: str, first_zone: int) -> dict[str, Any]:
    """The answer shown to the AI as a model; it is a valid recipe."""
    return {
        "format": RECIPE_FORMAT,
        "version": RECIPE_VERSION,
        "analysis_id": analysis_id,
        "mode": "look",
        "name": "…",
        "description": "…",
        "look": {"separation": 1.2, "protect_vivid": 0.5, "focus": {f"R{first_zone}": 0}},
        "global": {"chroma_strength": 1.0, "mix": 1.0},
        "finishing": {
            "vibrance": 0.2,
            "shadows": {"hue": 140, "amount": 0.3, "lightness": 0.0},
        },
    }


def build_look_prompt(
    reference: ZonePalette,
    analysis_id: str,
    separation: float,
    protect_vivid: float,
    focus: tuple[tuple[int, float], ...],
    gaps: dict[str, Any],
) -> str:
    """``PROMPT.md`` of a look pack (measures and current settings filled in)."""
    ids = reference.ids
    reference_ids = f"R{ids[0]} à R{ids[-1]}" if len(ids) > 1 else f"R{ids[0]}"
    focus_text = ", ".join(f"R{zone} = {weight:g}" for zone, weight in focus) or "aucun"
    example = json.dumps(example_recipe(analysis_id, ids[0]), ensure_ascii=False, indent=2)
    return _PROMPT.format(
        reference_ids=reference_ids,
        gaps=gaps_table(gaps),
        separation=separation,
        protect_vivid=protect_vivid,
        focus=focus_text,
        example=example,
    )


def write_look_pack(
    destination: Path,
    name: str,
    source_rgb: NDArray[np.uint8],
    reference_rgb: NDArray[np.uint8],
    render_rgb: NDArray[np.uint8],
    reference: ZonePalette,
    analysis_id: str,
    separation: float,
    protect_vivid: float,
    focus: tuple[tuple[int, float], ...] = (),
) -> tuple[Path, str]:
    """Write ``<destination>/<name>_stealzalut_look_pack/`` and its ``.zip``.

    Args:
        destination: Existing folder to write into.
        name: Base name of the pack.
        source_rgb: Source image, RGB uint8.
        reference_rgb: Reference image, RGB uint8.
        render_rgb: The current render of the source, RGB uint8.
        reference: Zones of the reference (the ids ``look.focus`` uses).
        analysis_id: Fingerprint of the pair the zone ids belong to.
        separation: Current separation.
        protect_vivid: Current protection of vivid colours (the effective one).
        focus: Current weights of the reference's zones.

    Returns:
        The pack folder (the zip sits next to it) and the prompt text.
    """
    folder = destination / f"{name}_stealzalut_look_pack"
    folder.mkdir(parents=True, exist_ok=True)

    gaps = look_gaps(render_rgb, reference_rgb)
    prompt = build_look_prompt(reference, analysis_id, separation, protect_vivid, focus, gaps)
    analysis = look_analysis_dict(
        reference_rgb, reference, analysis_id, separation, protect_vivid, focus, gaps
    )

    _write_jpeg(folder / LOOK_PACK_FILES[0], create_thumbnail(source_rgb, ORIGINAL_MAX_SIZE))
    _write_jpeg(folder / LOOK_PACK_FILES[1], create_thumbnail(reference_rgb, ORIGINAL_MAX_SIZE))
    _write_jpeg(folder / LOOK_PACK_FILES[2], create_thumbnail(render_rgb, ORIGINAL_MAX_SIZE))
    _write_png(folder / LOOK_PACK_FILES[3], zone_sheet(reference_rgb, reference, "R"))
    _write_png(folder / LOOK_PACK_FILES[4], zone_overview(reference_rgb, reference, "R"))
    (folder / LOOK_PACK_FILES[5]).write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (folder / LOOK_PACK_FILES[6]).write_text(prompt, encoding="utf-8")

    shutil.make_archive(str(folder), "zip", root_dir=folder)
    return folder, prompt
