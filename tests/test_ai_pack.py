"""Tests for stealzalut.core.ai_pack — the analysis pack for the user's AI."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from stealzalut.core.ai_pack import (
    PACK_FILES,
    PACK_MAX_SIZE,
    analysis_dict,
    build_prompt,
    write_pack,
    zone_overview,
    zone_sheet,
)
from stealzalut.core.zone_transfer import ZoneTransferModel, fit_zone_transfer


def _stripes(colors: list[tuple[int, int, int]], height: int = 300, width: int = 160) -> np.ndarray:
    img = np.zeros((height, width * len(colors), 3), dtype=np.uint8)
    for i, c in enumerate(colors):
        img[:, i * width : (i + 1) * width] = c
    return img


SOURCE = _stripes([(30, 70, 30), (120, 170, 230), (224, 172, 140)])
REFERENCE = _stripes([(30, 110, 120), (215, 180, 120), (255, 150, 70), (90, 40, 20)], 400, 500)


@pytest.fixture(scope="module")
def model() -> ZoneTransferModel:
    return fit_zone_transfer(SOURCE, REFERENCE, zone_count=4)


def test_analysis_json(model: ZoneTransferModel) -> None:
    data = analysis_dict(model, (480, 300), (2000, 400), 4)
    assert data["format"] == "stealzalut.analysis"
    assert data["analysis_id"] == model.analysis_id and len(model.analysis_id) == 8
    assert data["source"]["size"] == [480, 300]
    zones = data["source"]["zones"]
    assert [z["id"] for z in zones] == [1, 2, 3]
    assert sum(z["area_pct"] for z in zones) == pytest.approx(100.0, abs=0.5)
    for zone in zones:
        assert zone["hex"].startswith("#") and len(zone["oklch"]) == 3
        assert zone["lightness_band"] in ("shadows", "midtones", "highlights")
    assert len(data["auto_matching"]) == len(zones)
    json.dumps(data)  # serialisable as-is


def test_sheets_and_overviews(model: ZoneTransferModel) -> None:
    sheet = zone_sheet(REFERENCE, model.reference, "R")
    assert sheet.dtype == np.uint8 and sheet.ndim == 3
    assert max(sheet.shape[:2]) <= PACK_MAX_SIZE
    overview = zone_overview(REFERENCE, model.reference, "R")
    assert max(overview.shape[:2]) == PACK_MAX_SIZE  # 2000 px wide → fitted
    assert overview.mean() < REFERENCE.mean()  # darkened under the labels


def test_prompt_names_ids_and_analysis(model: ZoneTransferModel) -> None:
    prompt = build_prompt(model)
    assert model.analysis_id in prompt
    assert "S1, S2, S3" in prompt
    assert "R1" in prompt and "N'invente AUCUNE valeur de couleur" in prompt
    assert "```json" in prompt
    # The AI must know that doses multiply and that full is the normal value
    assert "MULTIPLIENT" in prompt and "PLEIN effet" in prompt
    assert '"strength": "full"' in prompt
    # Luminance is full by default; intensity can be compensated in finishing
    assert "Garde 1.0 par défaut" in prompt and "préfère ≤ 0.5" not in prompt
    assert '"luminance_strength": 1.0' in prompt
    assert '"finishing": { "saturation": 0.0, "vibrance": 0.0 }' in prompt


def test_write_pack_with_non_ascii_path(tmp_path: Path, model: ZoneTransferModel) -> None:
    destination = tmp_path / "Été"
    destination.mkdir()
    folder = write_pack(destination, "forêt_x_plage", SOURCE, REFERENCE, model, 4)
    assert folder.name == "forêt_x_plage_stealzalut_pack"
    assert sorted(p.name for p in folder.iterdir()) == sorted(PACK_FILES)
    image = cv2.imdecode(np.fromfile(folder / PACK_FILES[0], np.uint8), cv2.IMREAD_COLOR)
    assert image is not None and image.shape[:2] == SOURCE.shape[:2]  # untouched size
    original = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(int)
    assert np.abs(original - SOURCE.astype(int)).mean() < 3  # untouched colours
    sheet = cv2.imdecode(np.fromfile(folder / PACK_FILES[2], np.uint8), cv2.IMREAD_COLOR)
    assert sheet is not None
    analysis = json.loads((folder / "analysis.json").read_text(encoding="utf-8"))
    assert analysis["analysis_id"] == model.analysis_id

    with zipfile.ZipFile(folder.with_suffix(".zip")) as archive:
        assert sorted(archive.namelist()) == sorted(PACK_FILES)

    # Writing again (same analysis) overwrites cleanly, old file names included
    (folder / "1_source_zones.png").write_bytes(b"old pack")
    assert write_pack(destination, "forêt_x_plage", SOURCE, REFERENCE, model, 4) == folder
    assert sorted(p.name for p in folder.iterdir()) == sorted(PACK_FILES)


def test_large_originals_are_resized(tmp_path: Path, model: ZoneTransferModel) -> None:
    from stealzalut.core.ai_pack import ORIGINAL_MAX_SIZE

    big = cv2.resize(REFERENCE, (3000, 600), interpolation=cv2.INTER_NEAREST)
    folder = write_pack(tmp_path, "grand", SOURCE, big, model, 4)
    image = cv2.imdecode(np.fromfile(folder / PACK_FILES[1], np.uint8), cv2.IMREAD_COLOR)
    assert max(image.shape[:2]) == ORIGINAL_MAX_SIZE
