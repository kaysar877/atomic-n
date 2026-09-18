"""Package the small text artifacts (logs, metrics, summaries) from archived
legs into the repo's experiments/ tree. Checkpoints (487MB/leg) stay local and
are documented as available-on-request — never pushed to GitHub.

Only copies: run.log / diag.log / metrics.csv / summary.txt / phase.log
Output layout:
    experiments/
      arena/       lr_*/      (atomic vs vanilla matched-pair matrix)
      bisect/      lr_*/
      opt/         proof, attribution, recipe-repeat, solution, conf
      ascent/      leg_*/  (if present)
      multiseed/   seed_*/ (fills after runs)
"""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEST = ROOT / "experiments"

SMALL = {"run.log", "diag.log", "metrics.csv", "summary.txt", "phase.log",
         "cmd.log", "summary.csv"}

# (source dir pattern, dest subfolder)
BUCKETS = [
    ("sweep",          "arena/atomic"),
    ("sweep_baseline", "arena/vanilla"),
    ("sweep_bisect",   "bisect"),
    ("sweep_opt",      "opt/proof"),
    ("sweep_attrib_wd", "opt/attrib_wd"),
    ("sweep_attrib_cos", "opt/attrib_cos"),
    ("sweep_recipe",   "opt/recipe"),
    ("sweep_solution", "opt/solution"),
    ("sweep_conf",     "opt/conf"),
    ("sweep_v2probe",  "v2_rejected"),
    ("sweep_v2_radial", "v2_rejected"),
    ("sweep_ascent",   "ascent"),
    ("sweep_multiseed", "multiseed"),
]


def should_copy(name: str) -> bool:
    return any(name == s or (s.endswith(".log") and name.endswith(".log"))
               for s in SMALL) or name.endswith(".csv")


def main() -> int:
    copied = 0
    skipped_mb = 0
    for src_name, dest_rel in BUCKETS:
        src = ROOT / src_name
        if not src.exists():
            continue
        for leg in src.rglob("*"):
            if not leg.is_file():
                continue
            rel = leg.relative_to(src)
            if not should_copy(leg.name):
                skipped_mb += leg.stat().st_size / 1e6
                continue
            if "checkpoints" in str(rel) or "rng_state" in leg.name:
                skipped_mb += leg.stat().st_size / 1e6
                continue
            target = DEST / dest_rel / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(leg), str(target))
            copied += 1
    if copied == 0:
        # still write a marker so the folder exists in git
        DEST.mkdir(parents=True, exist_ok=True)
        (DEST / "README.txt").write_text(
            "experiments/ filled by package_experiments.py after legs complete.",
            encoding="utf-8")
    print(f"copied {copied} text artifacts; skipped {skipped_mb:.0f} MB "
          "of checkpoints (on-request only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())