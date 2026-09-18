"""Atomic LR micro-sweep driver (low-and-slow, crash-isolated per leg).

For each LR: archives any existing checkpoints/metrics (fresh init),
runs main.py for SWEEP_STEPS with SELFLEARN_LR set, then archives the
leg's outputs and parses its validation trajectory into summary.txt.

Run detached:  Start-Process python -ArgumentList sweep.py
Watch:         Get-Content logs/sweep.log -Tail 5 -Wait
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LEGS = [7.5e-5, 1.5e-4, 3e-4, 6e-4]   # 0.25x, 0.5x, 1x, 2x baseline LR
SWEEP_STEPS = 6000                     # 3 val evals per leg (every 2000)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s",
    handlers=[logging.FileHandler(ROOT / "logs" / "sweep.log", encoding="utf-8"),
              logging.StreamHandler(sys.stdout)])
log = logging.getLogger("sweep")


def _archive_leg(tag: str) -> None:
    dest = ROOT / "sweep" / tag
    dest.mkdir(parents=True, exist_ok=True)
    for p in (ROOT / "checkpoints").glob("*.pt"):
        shutil.move(str(p), dest / p.name)
    for name in ("metrics.csv",):
        p = ROOT / "logs" / name
        if p.exists():
            shutil.move(str(p), dest / name)


def _parse_vals(run_log: Path) -> list[tuple[int, float]]:
    out = []
    try:
        for line in run_log.read_text(encoding="utf-8",
                                      errors="ignore").splitlines():
            m = re.search(r"M step=(\d+)\s+train=([\d.]+)", line)
            if m:
                out.append((int(m.group(1)), float(m.group(2))))
    except OSError:
        pass
    return out


def run_leg(lr: float) -> dict:
    tag = f"lr_{lr:g}"
    log.info("=== leg %s (%d steps) ===", tag, SWEEP_STEPS)
    _archive_leg(tag + "_prev")          # clear the decks -> fresh init
    env = dict(os.environ)
    env["SELFLEARN_LR"] = str(lr)
    run_log = ROOT / "sweep" / tag / "run.log"
    run_log.parent.mkdir(parents=True, exist_ok=True)
    with run_log.open("w", encoding="utf-8") as fh:
        proc = subprocess.run(
            [sys.executable, "-u", "main.py", "--steps", str(SWEEP_STEPS),
             "--use-pytorch"],
            cwd=str(ROOT), env=env, stdout=fh, stderr=subprocess.STDOUT)
    log.info("leg %s exited rc=%d", tag, proc.returncode)
    leg_dir = ROOT / "sweep" / tag
    for p in (ROOT / "checkpoints").glob("*.pt"):
        shutil.move(str(p), leg_dir / p.name)
    m = ROOT / "logs" / "metrics.csv"
    if m.exists():
        shutil.move(str(m), leg_dir / "metrics.csv")
    vals, trains = [], _parse_vals(run_log)
    try:
        for line in run_log.read_text(encoding="utf-8",
                                      errors="ignore").splitlines():
            vm = re.search(r"VAL loss=([\d.]+) best=([\d.]+)", line)
            if vm:
                vals.append((float(vm.group(1)), float(vm.group(2))))
    except OSError:
        pass
    return {"lr": lr, "rc": proc.returncode, "vals": vals,
            "final_train": trains[-1][1] if trains else float("nan")}


def main() -> int:
    (ROOT / "sweep").mkdir(exist_ok=True)
    # archive whatever is currently lying around (interrupted challenger etc.)
    _archive_leg("prev_state")
    results = []
    for lr in LEGS:
        try:
            results.append(run_leg(lr))
        except Exception as exc:  # noqa: BLE001 - next leg must still run
            log.error("leg lr=%g crashed: %r", lr, exc)
            results.append({"lr": lr, "rc": -1, "vals": [], "final_train": float("nan")})
        time.sleep(5)
    lines = ["lr,final_train,val_trajectory(best per eval)"]
    for r in results:
        traj = ";".join(f"{v:.4f}/{b:.4f}" for v, b in r["vals"]) or "no-val"
        lines.append(f"{r['lr']:g},{r['final_train']:.4f},{traj}")
    (ROOT / "sweep" / "summary.txt").write_text("\n".join(lines) + "\n",
                                                encoding="utf-8")
    for line in lines:
        log.info("SWEEP %s", line)
    log.info("sweep complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
