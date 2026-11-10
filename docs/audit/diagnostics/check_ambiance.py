import sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, r"C:\Users\mathi\Documents\StealZaLut\src")
from stealzalut.core.color_transfer import fit_transfer
from stealzalut.core.colorspace import oklab_to_oklch
from stealzalut.core.grading import GradingParams, build_grid
from stealzalut.core.lut_apply import apply_lut
from stealzalut.core.palette import hue_concentration
from stealzalut.utils.image_io import bgr_to_rgb, load_image_bgr
S = Path(sys.argv[1]); P = Path(r"C:\Users\mathi\Pictures")
src = bgr_to_rgb(load_image_bgr(P / "IMG_6295.DNG"))
small = cv2.resize(src, (640, round(640 * src.shape[0] / src.shape[1])), interpolation=cv2.INTER_AREA)
print("source concentration", hue_concentration(src))
rows = []
for label, name in (("VIOLET", "841121141_18631647988026044_4768591964563047386_n.jpg"), ("TOKYO", "ben-blennerhassett-CptmwIhxfo4-unsplash.jpg")):
    ref = bgr_to_rgb(load_image_bgr(P / name))
    print(label, "concentration, colores:", tuple(round(v, 2) for v in hue_concentration(ref)))
    m = fit_transfer(src, ref, "ambiance")
    for L in (0.1, 0.25, 0.4, 0.7, 0.9):
        t = m.tint(np.array([L]))[0]; nm = f"L={L}"
        _, c, h = oklab_to_oklch(np.array([[0.5, *t]]))[0]; print(f"   teinte {nm}: C={c:.3f} h={h:.0f}")
    r = build_grid(m, 33, GradingParams())
    img = apply_lut(small, r.grid, 33)
    print("   folds", r.fold_fraction)
    refs = cv2.resize(ref, (round(ref.shape[1] * small.shape[0] / ref.shape[0]), small.shape[0]), interpolation=cv2.INTER_AREA)
    rows.append(np.hstack([refs, np.full((small.shape[0], 6, 3), 20, np.uint8), img]))
w = max(r.shape[1] for r in rows)
rows = [np.hstack([r, np.full((r.shape[0], w - r.shape[1], 3), 20, np.uint8)]) for r in rows]
cv2.imwrite(str(S / "ambiance_lut.jpg"), cv2.cvtColor(np.vstack([rows[0], np.full((6, w, 3), 20, np.uint8), rows[1]]), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 86])
