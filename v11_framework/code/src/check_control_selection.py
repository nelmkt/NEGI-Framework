"""Check post-period stable-surface control selection in the primary match.

This changes only control eligibility; treated cells, pre-treatment strata,
outcome, spatial blocks and the prespecified 2,000 bootstrap draws are fixed.
"""
from pathlib import Path

import pandas as pd

from fw_config import Config, SETTINGS
import fw_panel as panel
import fw_matching as matching


def main():
    cfg = Config()
    df, _ = panel.load(cfg)
    key = panel.strata(df, cfg, baseline_only=True)
    boot = matching.Bootstrap(df.block, cfg.n_boot, cfg.seed)
    rows = []
    for setting in SETTINGS:
        in_setting = df.setting.eq(setting)
        treated = df.group.eq("greened") & in_setting
        controls_all = df.group.eq("control") & in_setting
        stable = df.d_own_built.abs().lt(cfg.stable_surface_max)
        for label, controls in (("stable-surface controls", controls_all & stable),
                                ("all eligible controls", controls_all)):
            estimate = matching.att(df, treated, controls, "d_lst", key, cfg,
                                    boot, dose="n_greened_px")
            rows.append(dict(setting=setting, control_rule=label, **estimate))
    out = Path("tables/followup_control_selection_primary.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(pd.DataFrame(rows)[["setting", "control_rule", "estimate_C",
                              "lo_C", "hi_C", "n_treated_matched",
                              "n_controls_matched"]].to_string(index=False))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
