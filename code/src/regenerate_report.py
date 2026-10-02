"""Regenerate the narrative report from saved observed-panel results and current claim rules."""
import json
from pathlib import Path

from fw_config import Config
from fw_report import write_report
from run_followup import restore


if __name__ == "__main__":
    result = restore(json.loads(Path("tables/results.json").read_text(encoding="utf-8")))
    write_report(result, Config(), Path("manuscript/REPORT.md"))
    print("Report regenerated; saved numerical results were not refitted.")
