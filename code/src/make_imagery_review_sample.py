"""Create a blinded, reproducible high-resolution imagery review queue.

Sample 25 primary-matched outside cells in each dose-by-southern-belt
stratum. This selects locations only; it does not classify land use or infer
irrigation. Preserve queue and sampling key as separate versioned files.
"""
from pathlib import Path
import numpy as np
import pandas as pd

from fw_config import Config
import fw_panel as panel
import fw_matching as matching

ROOT = Path(__file__).resolve().parents[2]
QUEUE = ROOT / "tables" / "imagery_review_queue_v7.csv"
KEY = ROOT / "tables" / "imagery_sampling_key_v7.csv"
SEED = 20261001


def main():
    if QUEUE.exists() or KEY.exists():
        raise SystemExit("Refusing to overwrite imagery review queue or key.")
    cfg = Config(panel_path=ROOT / "code/gee/panel.csv")
    df, _ = panel.load(cfg)
    treated = df.group.eq("greened") & df.setting.eq("outside the built-up area")
    controls = df.group.eq("control") & df.setting.eq("outside the built-up area") & df.d_own_built.abs().lt(cfg.stable_surface_max)
    strata = panel.strata(df, cfg, baseline_only=True)
    matched = treated & matching.control_weights(df, treated, controls, strata, cfg).gt(0)
    frame = df.loc[matched, ["lon", "lat", "n_greened_px", "block"]].copy()
    frame["dose_class"] = pd.cut(frame.n_greened_px, bins=[0, 2, 4, 8, 9],
                                 labels=["1–2", "3–4", "5–8", "9/9"])
    frame["southern_belt"] = np.where(frame.lat.ge(21.3) & frame.lat.lt(21.4), "yes", "no")
    rng = np.random.default_rng(SEED)
    chosen = []
    for (dose, belt), group in frame.groupby(["dose_class", "southern_belt"], observed=True, sort=True):
        group = group.sort_values(["lon", "lat"])
        n = min(25, len(group))
        indices = rng.choice(group.index.to_numpy(), size=n, replace=False)
        part = group.loc[indices].copy().sort_values(["lon", "lat"])
        part["stratum_n"] = len(group)
        part["sample_n"] = n
        part["sampling_weight"] = len(group) / n
        chosen.append(part)
    sample = pd.concat(chosen).reset_index(drop=True)
    sample["sample_id"] = [f"NEGI-{i:03d}" for i in range(1, len(sample) + 1)]
    key_cols = ["sample_id", "dose_class", "southern_belt", "n_greened_px", "block",
                "stratum_n", "sample_n", "sampling_weight"]
    sample[key_cols].to_csv(KEY, index=False)
    review = sample[["sample_id", "lon", "lat"]].copy()
    review["target_period"] = "2024–2025"
    review["review_status"] = "pending"
    for col in ("image_date", "land_use_label", "confidence", "evidence_notes"):
        review[col] = ""
    review.to_csv(QUEUE, index=False)
    print(f"Wrote {len(review)} unlabelled review locations in {len(chosen)} dose-by-belt strata")
    print(sample.groupby(["dose_class", "southern_belt"], observed=True)
          .agg(population=("stratum_n", "first"), sample=("sample_n", "first")).to_string())


if __name__ == "__main__":
    main()
