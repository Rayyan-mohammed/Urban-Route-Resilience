"""Tests for real-dataset ingest (roadmap §3.2).

Uses synthetic fixtures (no real DeepGlobe/SpaceNet download needed): a fake
folder of `*_sat.jpg` / `*_mask.png` and a generic image/mask dir. Verifies the
core promise — real image tiles are written and `image_path` is populated so
training reads actual pixels instead of the synthetic stand-in.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PIL")
pytest.importorskip("rasterio")

from PIL import Image  # noqa: E402

from route_resilience.config import load_config  # noqa: E402
from route_resilience.data import ingest  # noqa: E402


def _cfg(tile=64):
    # small tiles + low road threshold so a tiny fixture yields tiles
    return load_config("base.yaml", "data.yaml", overrides=[
        f"data.tile_size={tile}", "data.overlap=0.0", "data.min_road_frac=0.001",
    ])


def _road_image_mask(h=128, w=128):
    img = np.random.randint(0, 255, (h, w, 3), np.uint8)
    mask = np.zeros((h, w), np.uint8)
    mask[h // 2 - 2 : h // 2 + 2, :] = 255   # a horizontal road across every column
    return img, mask


def test_ingest_pairs_writes_real_images(tmp_path):
    cfg = _cfg(tile=64)
    img, mask = _road_image_mask(128, 128)
    df = ingest.ingest_pairs(
        [(img, mask, "chip0")], cfg, source="folder", terrain="test",
        gsd_m=0.5, out_dir=tmp_path,
    )
    assert len(df) >= 2                       # 128/64 -> tiles produced
    assert (df["image_path"] != "").all()     # real imagery wired, not synth
    assert (df["terrain"] == "test").all()
    # every referenced file exists and image tiles are 3-band
    import rasterio
    for _, row in df.iterrows():
        assert (tmp_path / "images" / f"{row['tile_id']}.tif").exists()
        with rasterio.open(row["image_path"]) as ds:
            assert ds.count == 3 and ds.width == 64 and ds.height == 64


def test_ingest_skips_empty_tiles(tmp_path):
    cfg = _cfg(tile=64)
    img = np.random.randint(0, 255, (128, 128, 3), np.uint8)
    empty = np.zeros((128, 128), np.uint8)     # no road anywhere
    df = ingest.ingest_pairs([(img, empty, "blank")], cfg, source="folder",
                             terrain="t", out_dir=tmp_path)
    assert df.empty                            # road_frac filter drops all tiles


def test_iter_deepglobe_pairs(tmp_path):
    # fake DeepGlobe layout: 105_sat.jpg + 105_mask.png
    img, mask = _road_image_mask(96, 96)
    Image.fromarray(img).save(tmp_path / "105_sat.jpg")
    Image.fromarray(mask).save(tmp_path / "105_mask.png")
    # an image with no mask (test-split style) must be skipped, not crash
    Image.fromarray(img).save(tmp_path / "999_sat.jpg")

    pairs = list(ingest.iter_deepglobe(tmp_path))
    assert len(pairs) == 1
    im, mk, base = pairs[0]
    assert base == "105" and im.shape == (96, 96, 3) and mk.max() > 0


def test_ingest_source_deepglobe_end_to_end(tmp_path):
    img, mask = _road_image_mask(128, 128)
    Image.fromarray(img).save(tmp_path / "a_sat.jpg")
    Image.fromarray(mask).save(tmp_path / "a_mask.png")
    cfg = _cfg(tile=64)
    df = ingest.ingest_source("deepglobe", cfg, terrain="deepglobe",
                              root=tmp_path, out_dir=tmp_path / "out")
    assert not df.empty
    assert (df["place"] == "deepglobe").all()
    assert (df["image_path"] != "").all()


def test_iter_folder_matches_by_stem(tmp_path):
    imgs, masks = tmp_path / "img", tmp_path / "msk"
    imgs.mkdir()
    masks.mkdir()
    img, mask = _road_image_mask(64, 64)
    Image.fromarray(img).save(imgs / "tile1.png")
    Image.fromarray(mask).save(masks / "tile1.png")
    Image.fromarray(img).save(imgs / "orphan.png")   # no matching mask -> skipped
    pairs = list(ingest.iter_folder(imgs, masks))
    assert len(pairs) == 1 and pairs[0][2] == "tile1"
