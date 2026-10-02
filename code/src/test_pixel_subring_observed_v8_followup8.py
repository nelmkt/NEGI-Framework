"""Independent saved-output reconciliation for the observed 30/60 m export."""
import hashlib
import json
import unittest

import numpy as np
import pandas as pd

from fw_config import Config, SETTINGS
from analyze_isolation_reweighting_v8 import ROOT, PANEL, RAW as PRIOR
from analyze_pixel_subrings_v8_followup6 import DEFAULT_RAW, load_new, subring_design

COEF = ROOT / "tables/pixel_subring_regression_v8_followup6.csv"
COLLIN = ROOT / "tables/pixel_subring_collinearity_v8_followup6.csv"
PAIRED = ROOT / "tables/pixel_subring_pair_differences_v8_followup7.csv"


class ObservedPixelSubringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = Config(panel_path=PANEL)
        cls.raw = pd.read_csv(DEFAULT_RAW)
        cls.df = load_new(DEFAULT_RAW, cls.cfg)
        cls.coef = pd.read_csv(COEF)
        cls.collin = pd.read_csv(COLLIN)
        cls.paired = pd.read_csv(PAIRED)
        controls = (cls.df.cls.eq(3) & cls.df.setting.eq(SETTINGS[0]) &
                    cls.df.d_own_built.abs().lt(cls.cfg.stable_surface_max))
        n_ctrl = cls.df.loc[controls, "key"].value_counts()
        cls.small = (cls.df.cls.eq(1) & cls.df.setting.eq(SETTINGS[0]) &
                     cls.df.key.isin(n_ctrl[n_ctrl >= cls.cfg.min_controls].index) &
                     cls.df.n_greened_px.le(2))

    def test_export_metadata_and_nested_counts(self):
        meta = json.loads(DEFAULT_RAW.with_suffix(".json").read_text(encoding="utf-8"))
        self.assertEqual(meta["n_exported_cells"], len(self.raw))
        self.assertEqual(len(self.raw), 1333)
        self.assertTrue(all(value == 0 for value in meta["checks"].values()))
        self.assertEqual(meta["source_panel_sha256"], hashlib.sha256(PANEL.read_bytes()).hexdigest())
        self.assertEqual(meta["source_prior_external_counts_sha256"],
                         hashlib.sha256(PRIOR.read_bytes()).hexdigest())
        self.assertTrue((self.raw.external_ring_0_30m + self.raw.external_ring_30_60m +
                         self.raw.external_ring_60_90m).eq(self.raw.external_green_px_90m).all())
        for term in ("external_ring_0_30m", "external_ring_30_60m", "external_ring_60_90m"):
            self.assertAlmostEqual(meta["zero_count_share_by_subring"][term],
                                   self.raw[term].eq(0).mean())

    def test_primary_zero_shares_and_design_rank_reconcile(self):
        self.assertEqual((int(self.small.sum()), int(self.df.loc[self.small, "block"].nunique())), (419, 32))
        selected = self.df.loc[self.small]
        fine = subring_design(selected.n_greened_px,
                              selected.external_green_px_30m,
                              selected.external_green_px_60m,
                              selected.external_green_px_90m,
                              selected.external_green_px_180m,
                              selected.external_green_px_300m)["fine_three_inner_rings"]
        self.assertEqual(np.linalg.matrix_rank(fine), 6)
        terms = ("own", "external_0_30m", "external_30_60m", "external_60_90m",
                 "external_90_180m", "external_180_300m")
        rows = self.coef.loc[self.coef.model.eq("fine_three_inner_rings")].set_index("term")
        for j, term in enumerate(terms):
            self.assertAlmostEqual(rows.loc[term, "zero_count_share"], np.mean(fine[:, j] == 0))
        a, b = fine[:, 1], fine[:, 2]
        cosine = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
        row = self.collin.loc[self.collin.term_a.eq("external_0_30m") &
                              self.collin.term_b.eq("external_30_60m")].iloc[0]
        self.assertAlmostEqual(cosine, row.uncentred_cosine)

    def test_paired_intervals_and_point_differences_reconcile(self):
        fine = self.coef.loc[self.coef.model.eq("fine_three_inner_rings")].set_index("term")
        self.assertEqual(len(self.paired), 3)
        for row in self.paired.itertuples():
            self.assertAlmostEqual(row.beta_a_minus_beta_b_C_per_pixel,
                                   fine.loc[row.term_a, "beta_C_per_pixel"] -
                                   fine.loc[row.term_b, "beta_C_per_pixel"])
            self.assertEqual(row.valid_paired_block_draws, 2000)
            self.assertLess(row.lo_C, 0)
            self.assertGreater(row.hi_C, 0)


if __name__ == "__main__":
    unittest.main()
