"""Check whether the user-labelled raw-v3 file changes any exported values."""
from hashlib import sha256
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "code/gee/monthly_isolation_raw_v2.csv"
USER = ROOT / "code/gee/monthly_isolation_raw_v3_oldrecipe.csv"
OUT = ROOT / "tables/gee_raw_version_comparison.csv"


def main():
    if OUT.exists():
        raise SystemExit(f"Refusing to overwrite {OUT}")
    a = pd.read_csv(OLD).sort_values(["lon", "lat"]).reset_index(drop=True)
    b = pd.read_csv(USER).sort_values(["lon", "lat"]).reset_index(drop=True)
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        raise SystemExit("The two raw exports differ in columns or row count.")
    changed = int(a.ne(b).any(axis=1).sum())
    row = dict(old_file=OLD.name, user_file=USER.name, n_rows=len(a),
               n_changed_rows=changed,
               old_sha256=sha256(OLD.read_bytes()).hexdigest(),
               user_sha256=sha256(USER.read_bytes()).hexdigest())
    pd.DataFrame([row]).to_csv(OUT, index=False)
    print(pd.DataFrame([row]).to_string(index=False))


if __name__ == "__main__":
    main()
