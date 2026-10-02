"""Fetch the land-cover labels straight from Earth Engine — no asset upload, no Drive export.

One-time setup (in PowerShell):
    pip install earthengine-api
    earthengine authenticate          # opens your browser; sign in with your Google account

Then, from this package root:
    python code\\gee\\fetch_landcover.py --project YOUR-CLOUD-PROJECT-ID

It writes code\\gee\\Jeddah_landcover.csv (lon, lat, worldcover, wc_built_share, wc_bare_share,
ghsl_built_frac) for all pixels in code\\gee\\Jeddah_points_for_gee.csv. Then run the pipeline with
    --landcover code\\gee\\Jeddah_landcover.csv
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True, help="your Google Cloud project ID registered for Earth Engine")
    ap.add_argument("--points", default=str(HERE / "Jeddah_points_for_gee.csv"))
    ap.add_argument("--out", default=str(HERE / "Jeddah_landcover.csv"))
    ap.add_argument("--chunk", type=int, default=2000)
    a = ap.parse_args()
    try:
        import ee
    except ImportError:
        sys.exit("earthengine-api is not installed: run  pip install earthengine-api")
    try:
        ee.Initialize(project=a.project)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=a.project)

    wc = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
    shares = (wc.eq(50).addBands(wc.eq(60))
              .reduceNeighborhood(reducer=ee.Reducer.mean(), kernel=ee.Kernel.square(15, "meters"))
              .rename(["wc_built_share", "wc_bare_share"]))
    ghsl = (ee.Image("JRC/GHSL/P2023A/GHS_BUILT_S/2020").select("built_surface")
            .divide(10000).clamp(0, 1).rename("ghsl_built_frac"))
    stack = wc.rename("worldcover").addBands(shares).addBands(ghsl)

    pts = pd.read_csv(a.points)
    rows = []
    n = len(pts)
    for i0 in range(0, n, a.chunk):
        part = pts.iloc[i0:i0 + a.chunk]
        fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]),
                                              {"i": int(i)}) for i, r in part.iterrows()])
        for attempt in range(3):
            try:
                res = stack.reduceRegions(collection=fc, reducer=ee.Reducer.first(), scale=10).getInfo()
                break
            except Exception as e:
                if attempt == 2:
                    raise
                print(f"  retrying chunk {i0} ({e})"); time.sleep(5)
        for f in res["features"]:
            p = f["properties"]
            rows.append({"i": p["i"], "worldcover": p.get("worldcover"), "wc_built_share": p.get("wc_built_share"),
                         "wc_bare_share": p.get("wc_bare_share"), "ghsl_built_frac": p.get("ghsl_built_frac")})
        print(f"  {min(i0 + a.chunk, n):,} / {n:,} points")
    lc = pd.DataFrame(rows).set_index("i").sort_index()
    out = pts.join(lc)
    out.to_csv(a.out, index=False, float_format="%.10f")
    print(f"Wrote {a.out} ({out['worldcover'].notna().mean():.0%} of points labelled)")


if __name__ == "__main__":
    main()
