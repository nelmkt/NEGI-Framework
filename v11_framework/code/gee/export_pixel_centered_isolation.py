"""Export an exact 30 m pixel-centred 300 m external-greening screen.

Run from package root in an authenticated Earth Engine session:
    python code/gee/export_pixel_centered_isolation.py --project PROJECT \
        --out code/gee/pixel_centered_isolation_raw_v7.csv

For each classified greened 90 m cell, count other endpoint-greened 30 m
pixels within 300 m of *each own greened pixel*, excluding all own nine
30 m pixels. The saved value is the maximum external count over the own
greened pixels: zero means every own greened pixel passes the screen.
This is endpoint neighborhood geometry, not evidence of physical spillover
or applied irrigation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import export_panel as source

ROOT = Path(__file__).resolve().parents[2]
REVISION = "pixel-centered-native30-v7"


def build(ee, panel_image, cell, greened):
    landsat = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
               .filterBounds(ee.Geometry.Rectangle(source.RECT))
               .filter(ee.Filter.lt("CLOUD_COVER", 20)))
    proj30 = landsat.first().select("SR_B4").projection()
    green30 = greened.unmask(0).toFloat().rename("green30").reproject(proj30)
    own90 = (green30.reduceResolution(ee.Reducer.mean(), maxPixels=64)
             .reproject(cell).multiply(9).rename("green_count_cell"))
    own30 = own90.reproject(proj30)
    total30 = (green30.reduceNeighborhood(
        reducer=ee.Reducer.sum(),
        kernel=ee.Kernel.circle(radius=300, units="meters", normalize=False),
        skipMasked=False).reproject(proj30))
    external30 = total30.subtract(own30).updateMask(green30)
    negative30 = external30.lt(-0.25).toFloat().updateMask(green30)
    max_external90 = (external30.max(0).reduceResolution(ee.Reducer.max(), maxPixels=64)
                      .reproject(cell).rename("max_external_green_within300_any_own30"))
    bad90 = (negative30.reduceResolution(ee.Reducer.max(), maxPixels=64)
             .reproject(cell).unmask(0).rename("negative_external_flag"))
    classified = panel_image.select("cls").eq(1)
    image = (ee.Image.cat([own90, max_external90, bad90])
             .updateMask(classified).clip(ee.Geometry.Rectangle(source.RECT)))
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "code/gee/pixel_centered_isolation_raw_v7.csv")
    args = parser.parse_args()
    print(f"NEGI Earth Engine pixel-centred recipe: {REVISION}", flush=True)
    if args.out.exists() or args.out.with_suffix(".json").exists():
        raise SystemExit(f"Refusing to overwrite {args.out} or its metadata")
    ee = source._init(args.project)
    panel_image, land, cell, _, greened = source.build(ee)
    diagnostic = build(ee, panel_image, cell, greened)
    rows = []
    for index, box in enumerate(source.tiles(*source.TILES), 1):
        part = source.fetch(ee, diagnostic, land, cell, box)
        rows.extend(part)
        print(f"  tile {index}/{source.TILES[0] * source.TILES[1]}: {len(part)} cells", flush=True)
    data = pd.DataFrame(rows).drop_duplicates(["lon", "lat"])
    if data.empty:
        raise SystemExit("No classified greened cells returned; no output written.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.out, index=False, float_format="%.6f")
    original = pd.read_csv(ROOT / "code/gee/panel.csv")
    original = original.loc[original.cls.eq(1), ["lon", "lat", "greened_frac"]].copy()
    for frame in (data, original):
        frame["lon_key"] = frame.lon.round(6)
        frame["lat_key"] = frame.lat.round(6)
    check = data.merge(original[["lon_key", "lat_key", "greened_frac"]],
                       on=["lon_key", "lat_key"], how="left", validate="one_to_one")
    mismatched = int((check.green_count_cell - (check.greened_frac * 9).round()).abs().gt(.25).sum())
    unjoined = int(check.greened_frac.isna().sum())
    negative = int(check.negative_external_flag.gt(.25).sum())
    missing_screen = int(check.max_external_green_within300_any_own30.isna().sum())
    metadata = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "recipe_revision": REVISION,
        "source_panel_sha256": hashlib.sha256((ROOT / "code/gee/panel.csv").read_bytes()).hexdigest(),
        "n_exported_cells": int(len(data)),
        "dose_count_disagreements": mismatched,
        "unjoined_cells": unjoined,
        "negative_external_flags": negative,
        "missing_pixel_centered_screens": missing_screen,
        "distance_definition": "300 m from each own greened 30 m pixel centre to other greened 30 m pixel centres outside the own 90 m cell",
        "pixel_centered_screen": "max external 300 m count over all own greened pixels equals zero",
    }
    args.out.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if (len(data) != len(original) or mismatched or unjoined or negative or missing_screen):
        raise SystemExit("Export saved for diagnosis but failed count/alignment checks: " + str(metadata))
    print(f"Wrote {args.out}: {len(data)} classified greened cells; all checks zero")


if __name__ == "__main__":
    main()
