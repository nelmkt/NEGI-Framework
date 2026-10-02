"""Recompute paired mixing/coastal diagnostics without refitting the LST models."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from fw_config import Config, SETTINGS
import fw_panel as panel, fw_contrast_checks as contrast_checks


def main():
    cfg = Config()
    df, _ = panel.load(cfg)
    checks = contrast_checks.run(df, cfg)
    tables = cfg.out_dir / "tables"
    original = pd.read_csv(tables / "measured_dose.csv")
    original = original[original.setting == SETTINGS[0]].reset_index(drop=True)
    reproduced = checks["dose_reproduction"]
    for field in ["estimate_C", "lo_C", "hi_C"]:
        np.testing.assert_allclose(reproduced[field], original[field], atol=1e-10, rtol=0)
    support = pd.read_csv(tables / "test_supported_by_dose.csv")
    doses = pd.read_csv(tables / "test_by_dose.csv")
    denom = doses[doses.test == "B"][["setting", "pixels", "n_cells"]].rename(columns={"n_cells":"n_matched"})
    support = support.merge(denom, on=["setting", "pixels"], validate="one_to_one")
    support["supported_share"] = support.n_cells/support.n_matched
    support["interpretation"] = np.where(support.n_cells == 0, "no supported comparison available",
        np.where((support.n_cells < cfg.min_validation_cells) | (support.n_blocks < cfg.min_validation_blocks) |
                 (support.n_joint_valid < cfg.min_validation_refits), "descriptive only; insufficient supported replication",
                 "supported subset only; tolerance not supplied"))
    checks["dose_support"] = support
    for name, data in checks.items():
        data.to_csv(tables / f"contrast_{name}.csv", index=False)
        print(name)
        print(data.to_string(index=False))
    path = cfg.out_dir / "tables" / "results.json"
    res = json.loads(path.read_text(encoding="utf-8"))
    res["contrasts"] = {k: json.loads(v.to_json(orient="records", double_precision=15)) for k, v in checks.items()}
    res["contrast_provenance"] = dict(seed=cfg.seed, measured_draws=cfg.n_boot, model_refits_added=0,
        matched_dose_reproduction="point and interval endpoints agree within 1e-10",
        status="exploratory paired percentile intervals; not multiplicity-adjusted")
    path.write_text(json.dumps(res, indent=1, ensure_ascii=False, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
