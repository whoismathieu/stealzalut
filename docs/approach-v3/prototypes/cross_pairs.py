"""Real behaviour: source x analog reference, by category, look vs zones.

No ground truth here: how close each render's statistics come to the
reference's, and plates to judge by eye.  Nothing is written in the repository.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from stealzalut.core.color_transfer import fit_transfer
from stealzalut.core.colorspace import srgb_to_oklab
from stealzalut.core.grading import build_grid
from stealzalut.core.lut_apply import apply_lut
from stealzalut.core.look import look_diagnostics
from stealzalut.core.palette import analysis_oklab, colour_stats
from stealzalut.utils.image_io import create_thumbnail, load_image_rgb

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
T = Path(r"C:\Users\mathi\Pictures\test_images")
SRC, REF = T / "benchmark_photos_source", T / "benchmark_photos_analog"
PLATES_PER_CATEGORY = 3


def by_category(folder: Path) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for entry in json.loads((folder / "manifest.json").read_text(encoding="utf-8")):
        groups.setdefault(entry["category"][:2], []).append(entry["file"])
    return {key: sorted(files) for key, files in groups.items()}


def stats(lab: np.ndarray) -> dict[str, float]:
    chroma = np.hypot(lab[:, 1], lab[:, 2])
    dark = lab[:, 0] < 0.35
    return {
        "L": float(lab[:, 0].mean()),
        "C": float(chroma.mean()),
        "col": float(np.mean(chroma >= 0.05)),
        "sa": float(lab[dark, 1].mean()) if dark.any() else 0.0,
        "sb": float(lab[dark, 2].mean()) if dark.any() else 0.0,
    }


def gap(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    return {
        "dL": abs(a["L"] - b["L"]),
        "dC": abs(a["C"] - b["C"]),
        "dcol": abs(a["col"] - b["col"]),
        "dshadow": float(np.hypot(a["sa"] - b["sa"], a["sb"] - b["sb"])),
    }


def label(img: np.ndarray, text: str) -> np.ndarray:
    img = img.copy()
    cv2.rectangle(img, (0, 0), (14 + 10 * len(text), 26), (0, 0, 0), -1)
    cv2.putText(img, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def fit_h(img: np.ndarray, height: int) -> np.ndarray:
    return cv2.resize(img, (round(img.shape[1] * height / img.shape[0]), height), interpolation=cv2.INTER_AREA)


sources, references = by_category(SRC), by_category(REF)
rows = []
for category in sorted(sources):
    plate_rows = []
    refs = references[category]
    for index, s_name in enumerate(sources[category]):
        r_name = refs[index % len(refs)]
        src, ref = load_image_rgb(SRC / s_name), load_image_rgb(REF / r_name)
        view = create_thumbnail(np.asarray(src), 640)
        ref_stats = stats(analysis_oklab(ref)[0])
        src_stats = stats(analysis_oklab(src)[0])
        row = {"category": category, "source": s_name[:40], "reference": r_name[:40],
               "mono": colour_stats(ref).is_monochrome, "nothing": gap(src_stats, ref_stats)}
        tiles = [("SOURCE", view), ("REFERENCE", create_thumbnail(np.asarray(ref), 640))]
        for method in ("zones", "look"):
            model = fit_transfer(src, ref, method=method)
            result = build_grid(model, 33)
            img = apply_lut(view, result.grid, 33)
            row[method] = gap(stats(srgb_to_oklab(img.reshape(-1, 3) / 255.0)), ref_stats)
            row[method]["folds"] = result.fold_fraction
            row[method]["auto_smooth"] = result.auto_smoothed
            if method == "look":
                row["low_colour"] = look_diagnostics(model, src).low_colour
            tiles.append((method.upper(), img))
        rows.append(row)
        if index < PLATES_PER_CATEGORY:
            plate_rows.append(np.hstack([label(fit_h(im, 300), name) for name, im in tiles]))
    width = max(r.shape[1] for r in plate_rows)
    plate = np.vstack([np.hstack([r, np.full((300, width - r.shape[1], 3), 20, np.uint8)]) for r in plate_rows])
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(plate, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    (out / f"cross_{category}.jpg").write_bytes(buf.tobytes())

(out / "cross.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
print("Ecart moyen des statistiques du rendu a celles de la reference (plus petit = plus proche)")
print(f"{'cat':4s} {'n':>2s} | {'':8s} {'dL':>6s} {'dC':>6s} {'dcol':>6s} {'dombre':>7s} | replis>0 lissage_auto")
for category in sorted(sources):
    chosen = [r for r in rows if r["category"] == category]
    for key in ("nothing", "zones", "look"):
        m = {k: float(np.mean([r[key][k] for r in chosen])) for k in ("dL", "dC", "dcol", "dshadow")}
        extra = ""
        if key != "nothing":
            extra = f"{sum(r[key]['folds'] > 0 for r in chosen):>3d}      {sum(r[key]['auto_smooth'] for r in chosen)}"
        print(f"{category:4s} {len(chosen):2d} | {key:8s} {m['dL']:6.3f} {m['dC']:6.3f} {m['dcol']:6.3f} {m['dshadow']:7.3f} | {extra}")
print("references monochromes:", sum(r["mono"] for r in rows), "| sources peu colorees:", sum(r["low_colour"] for r in rows), "sur", len(rows))
