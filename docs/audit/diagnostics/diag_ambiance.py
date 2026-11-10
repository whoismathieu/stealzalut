"""Prototype 'ambiance' mode: the reference read as a tint per tone (split toning),
added on top of the source's own colours instead of replacing them."""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, r"C:\Users\mathi\Documents\StealZaLut\src")
from stealzalut.core.color_transfer import fit_transfer
from stealzalut.core.colorspace import gamut_compress, oklab_to_oklch, oklab_to_srgb, srgb_to_oklab
from stealzalut.core.grading import smoothstep
from stealzalut.utils.image_io import bgr_to_rgb, load_image_bgr

S = Path(sys.argv[1])
PICS = Path(r"C:\Users\mathi\Pictures")


def bernstein(L):
    L = np.clip(L, 0, 1)
    return np.stack([(1 - L) ** 2, 2 * L * (1 - L), L**2], axis=1)


def ref_tints(ref_lab):
    w = bernstein(ref_lab[:, 0])
    tints = (w[:, :, None] * ref_lab[:, None, 1:]).sum(0) / w.sum(0)[:, None]
    return tints  # (3, 2): shadows, mids, highlights


def ambiance(src_rgb, ref_rgb, cast=1.0, mute=0.35, keep_vivid=0.8):
    lab = srgb_to_oklab(src_rgb.reshape(-1, 3) / 255.0)
    ref_lab = srgb_to_oklab(ref_rgb.reshape(-1, 3) / 255.0)
    tints = ref_tints(ref_lab)
    glob = fit_transfer(src_rgb, ref_rgb, "oklab")
    L_new = srgb_to_oklab(glob.apply(src_rgb.reshape(-1, 3) / 255.0))[:, 0]
    chroma = oklab_to_oklch(lab)[:, 1]
    vivid = keep_vivid * smoothstep(chroma, 0.04, 0.12)
    w = bernstein(L_new)
    tint = (w[:, :, None] * tints[None]).sum(1)
    ab = lab[:, 1:] * (1 - mute * (1 - vivid))[:, None] + cast * (1 - vivid)[:, None] * tint
    out = np.column_stack([L_new, ab])
    return (oklab_to_srgb(gamut_compress(out)).reshape(src_rgb.shape) * 255 + 0.5).astype(np.uint8), tints


def conc(img):
    lch = oklab_to_oklch(srgb_to_oklab(img.reshape(-1, 3) / 255.0))
    col = lch[:, 1] > 0.05
    h, w = np.radians(lch[col, 2]), lch[col, 1]
    r = np.hypot((w * np.cos(h)).sum(), (w * np.sin(h)).sum()) / max(w.sum(), 1e-9)
    return f"C={lch[:, 1].mean():.3f} colores={col.mean()*100:.0f}% concentration R={r:.2f}"


src = bgr_to_rgb(load_image_bgr(PICS / "IMG_6295.DNG"))
small = cv2.resize(src, (640, round(640 * src.shape[0] / src.shape[1])), interpolation=cv2.INTER_AREA)
rows = []
for label, name in (("VIOLET", "841121141_18631647988026044_4768591964563047386_n.jpg"),
                    ("TOKYO VERT", "ben-blennerhassett-CptmwIhxfo4-unsplash.jpg")):
    ref = bgr_to_rgb(load_image_bgr(PICS / name))
    refs = cv2.resize(ref, (round(ref.shape[1] * small.shape[0] / ref.shape[0]), small.shape[0]), interpolation=cv2.INTER_AREA)
    img, tints = ambiance(small, refs)
    lch = [oklab_to_oklch(np.array([[0.5, *t]]))[0] for t in tints]
    print(label, "teintes ombres/moyens/hautes:", " | ".join(f"C={c:.3f} h={h:.0f}" for _, c, h in lch))
    print("   ambiance:", conc(img), "  reference:", conc(refs))
    tiles = []
    for t, im in (("REFERENCE " + label, refs), ("AMBIANCE (prototype)", img)):
        im = im.copy()
        cv2.rectangle(im, (0, 0), (12 + 11 * len(t), 26), (0, 0, 0), -1)
        cv2.putText(im, t, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(im)
    rows.append(np.hstack([tiles[0], np.full((small.shape[0], 6, 3), 20, np.uint8), tiles[1]]))
w = max(r.shape[1] for r in rows)
rows = [np.hstack([r, np.full((r.shape[0], w - r.shape[1], 3), 20, np.uint8)]) for r in rows]
sheet = np.vstack([rows[0], np.full((6, w, 3), 20, np.uint8), rows[1]])
cv2.imwrite(str(S / "ambiance.jpg"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 86])
