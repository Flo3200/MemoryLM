"""Run the whole experiment in order, resuming correctly after any reset:

    python -m memorylm.run_all --preset quick

  0. tests (CPU, ~1 min), data + tokenizer, speed test, time plan
  1. phase 1: baseline, then MemoryLM (identical tokens, order, seed)
  2. phase 2: MemoryLM recitation episodes
  3. evaluation, plots, report

Every stage checks what already exists on Drive and skips it, and every
training continues from its newest checkpoint. So after a disconnect the
same command simply continues.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from . import data as D
from .checkpoint import RunLog
from .config import load_config
from .train import Trainer, get_device, load_or_make_plan, run_dir

DRY_RUN = [  # ~10 GPU minutes end to end: proves the pipeline works before the real run
    "run_name=dryrun", "train.phase1.target_minutes_memorylm=1.5", "train.phase2.minutes=1",
    "train.speed_test_minutes=1", "train.eval_interval_minutes=0.5", "checkpoint.every_minutes=1",
    "train.eval_minutes=2", "train.total_gpu_hours=0.2", "eval.passages_per_length=32",
]


def run_tests() -> None:
    root = Path(__file__).resolve().parent.parent
    print("== required tests (CPU)")
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", str(root / "tests")], cwd=root)
    if r.returncode != 0:
        raise SystemExit("tests failed: training is NOT started (fix the bug first, GPU time is scarce)")


def summarize_plan(plan: dict) -> str:
    lines = [f"batch size {plan['batch_size']} on {plan['device']}",
             f"phase-1 budget {plan['phase1_tokens'] / 1e6:.0f} M tokens = {plan['phase1_steps']} steps per model",
             f"phase-2 {plan['phase2_steps']} steps"]
    est = plan.get("estimate_minutes")
    if est:
        lines.append("estimate (min): " + ", ".join(f"{k} {v:.0f}" for k, v in est.items()))
        if plan.get("shrunk"):
            lines.append("NOTE: the token budget was shrunk automatically to fit the GPU time limit")
    return "\n  ".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run all phases for both models (resumable).")
    ap.add_argument("--preset", default="quick")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides")
    ap.add_argument("--dry-run", action="store_true", help="~10-minute end-to-end rehearsal")
    ap.add_argument("--skip-tests", action="store_true")
    ap.add_argument("--plan-only", action="store_true", help="stop after phase 0 and print the plan")
    a = ap.parse_args(argv)
    cfg = load_config(a.preset, (DRY_RUN if a.dry_run else []) + a.set)
    log = RunLog(run_dir(cfg))
    t0 = time.time()

    if not a.skip_tests:
        run_tests()

    # ---- phase 0
    log.update_status(stage="phase0")
    print("== phase 0: data and tokenizer")
    # The whole train file is tokenized (~10-15 min on Colab) unless the run
    # is a dry run or has a fixed small budget; tokenizing is resumable.
    fixed = cfg.train.phase1.tokens
    need = 30_000_000 if a.dry_run else (None if fixed == "auto" else int(fixed) + 1_000_000)
    D.prepare_data(cfg, min_train_tokens=need)
    D.localize(cfg)
    print("== phase 0: speed test and time plan")
    plan = load_or_make_plan(cfg, get_device(), phase0_seconds=time.time() - t0)
    print("  " + summarize_plan(plan))
    log.update_status(stage="phase0_done", plan=plan)
    if a.plan_only:
        return

    # ---- phases 1 and 2
    for kind in ("baseline", "memorylm"):
        log.update_status(stage=f"training_{kind}")
        print(f"== training {kind}")
        Trainer(cfg, kind, plan).train()

    # ---- phase 3
    log.update_status(stage="phase3_eval")
    print("== phase 3: evaluation")
    try:
        from .evaluate import run_evaluation
    except ImportError:
        print("  evaluation module not available yet")
    else:
        run_evaluation(cfg)
    log.update_status(stage="done")
    print(f"== all done ({(time.time() - t0) / 60:.0f} min in this session)")


if __name__ == "__main__":
    main()
