"""Phase 3: evaluation, plots and a plain-language report.

Measures
  1. validation loss of both models (after phase 1: identical training tokens)
  2. recall accuracy with a cleared context, by passage length (16/32/64/128)
     under four conditions:
       baseline_full      passage still in the context  -> upper bound
       baseline_cleared   context cleared, no memory     -> lower bound
       memorylm_cleared   context cleared, memory on     -> the experiment
       memorylm_disabled  context cleared, memory off    -> must equal the lower bound
  3. compression: correctly recalled tokens per KB of memory written.

Accuracy is measured in two ways:
  teacher-forced  the model sees the TRUE previous passage tokens and predicts
                  the next one (how much the model "knows" at each position);
  greedy          the model generates the whole passage itself, errors
                  propagate (strict recall, like reciting a poem).
Even without any memory, teacher-forced accuracy is well above 0, because
the language model can guess many tokens from the preceding ones ("Once upon
a" -> "time"). That is exactly what the lower bound measures; the memory's
contribution is the gap above it.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from . import data as D
from .checkpoint import RunLog, atomic_write_json
from .memory import MemoryLM, build_model
from .model_core import KVCache
from .train import amp_dtype, get_device, lm_loss, recitation_forward, run_dir


# ======================================================================
# loading and generation (shared with the local app)
# ======================================================================
def load_model(cfg, kind: str, path: Path, device=None):
    device = device or get_device()
    model = build_model(cfg, kind)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    return model.to(device).eval()


@torch.no_grad()
def read_text(model: MemoryLM, ids: torch.Tensor, state=None):
    """Let MemoryLM read a (B, T) text of any length: blocks of `context`
    tokens, each with a fresh attention cache, while the MEMORY persists.
    That is the whole point: what is written survives the cleared context."""
    state = state or model.new_state()
    T = model.cfg.model.context
    for s in range(0, ids.shape[1], T):
        model.process(ids[:, s:s + T], state, write=True)
    return state


@torch.no_grad()
def generate(model, prompt: torch.Tensor, max_new: int, state=None, use_memory: bool = True,
             temperature: float = 0.0, top_k: int = 0, stop_id: Optional[int] = None,
             generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Continue `prompt` (B, P) by up to `max_new` tokens inside one context
    window. temperature 0 = greedy (always the most likely token)."""
    ctx = model.cfg.model.context
    max_new = min(max_new, ctx - prompt.shape[1])
    cache = KVCache(model.cfg.model.n_layers)
    core = model.core

    def step(idx, pos0):
        if isinstance(model, MemoryLM):
            st = state if state is not None else model.new_state()
            return model.process(idx, st, cache=cache, pos0=pos0, write=False, use_memory=use_memory)["logits"]
        return core(idx, pos0=pos0, cache=cache)["logits"]

    logits = step(prompt, 0)[:, -1]
    out = []
    done = torch.zeros(prompt.shape[0], dtype=torch.bool, device=prompt.device)
    for i in range(max_new):
        if temperature <= 0:
            nxt = logits.argmax(-1)
        else:
            lg = logits.float() / temperature
            if top_k:
                kth = lg.topk(top_k, -1).values[:, -1:]
                lg = lg.masked_fill(lg < kth, -float("inf"))
            nxt = torch.multinomial(torch.softmax(lg, -1), 1, generator=generator).squeeze(-1)
        out.append(nxt)
        if stop_id is not None:
            done |= nxt == stop_id
            if done.all():
                break
        if i + 1 < max_new:
            logits = step(nxt[:, None], prompt.shape[1] + i)[:, -1]
    return torch.stack(out, 1) if out else prompt[:, :0]


# ======================================================================
# the four recall conditions
# ======================================================================
@torch.no_grad()
def full_context_logits(model, passages: torch.Tensor, recite_id: int) -> torch.Tensor:
    """Upper bound: [passage, <recite>, passage] with the passage still in the
    context window. If 2N > context, a sliding window of the last `context`
    tokens is used, so for N = 128 the very token to be copied has just left
    the window: the 128-token context itself limits this bound."""
    B, N = passages.shape
    ctx = model.cfg.model.context
    seq = torch.cat([passages, torch.full((B, 1), recite_id, dtype=passages.dtype, device=passages.device), passages], 1)
    inp = seq[:, :-1]                                   # predicts seq[:, 1:]
    if inp.shape[1] <= ctx:
        return model(inp)["logits"][:, N:]
    rows = []
    for i in range(N):
        end = N + 1 + i                                 # position whose next token is passages[:, i]
        win = inp[:, max(0, end - ctx):end]
        rows.append(model(win)["logits"][:, -1])
    return torch.stack(rows, 1)


@torch.no_grad()
def greedy_recite(model, passages: torch.Tensor, recite_id: int, use_memory: bool) -> tuple[torch.Tensor, float]:
    """Read (if memory), clear the context, then generate the passage freely."""
    B, N = passages.shape
    state = None
    if isinstance(model, MemoryLM):
        state = model.new_state()
        if use_memory:
            model.process(passages, state, write=True)
    prompt = torch.full((B, 1), recite_id, dtype=passages.dtype, device=passages.device)
    gen = generate(model, prompt, N, state=state, use_memory=use_memory)
    return gen, (state.size_kb() if state is not None and use_memory else 0.0)


def _acc_ci(correct: np.ndarray) -> tuple[float, float]:
    """Mean and a 95 % half-width over PASSAGES (passages are the independent
    units; tokens within one passage are not independent)."""
    per_passage = correct.mean(1)
    return float(per_passage.mean()), float(1.96 * per_passage.std(ddof=1) / math.sqrt(len(per_passage)))


@torch.no_grad()
def evaluate_recall(cfg, models: dict, pool: D.Pool, recite_id: int, device, batch: int = 64) -> dict:
    res = {}
    dt = amp_dtype(cfg, device)
    for L in cfg.eval.lengths:
        P = D.eval_passages(pool, L, int(cfg.eval.passages_per_length), seed=cfg.seed)
        conds = {"baseline_full": [], "baseline_cleared": [], "memorylm_cleared": [], "memorylm_disabled": []}
        greedy = {"baseline_cleared": [], "memorylm_cleared": [], "memorylm_disabled": []}
        kb, wr = [], []
        for s in range(0, len(P), batch):
            p = P[s:s + batch].to(device)
            with torch.autocast("cuda", dtype=dt, enabled=dt is not None and device.type == "cuda"):
                lg = {"baseline_full": full_context_logits(models["baseline"], p, recite_id),
                      "baseline_cleared": recitation_forward(models["baseline"], p, recite_id)[0],
                      "memorylm_disabled": recitation_forward(models["memorylm"], p, recite_id, use_memory=False)[0]}
                lm, st = recitation_forward(models["memorylm"], p, recite_id, use_memory=True)
                lg["memorylm_cleared"] = lm
                kb.append(st.size_kb())
                wr.append((st.hard_writes / max(1, st.decisions)).tolist())
                for k, v in lg.items():
                    conds[k].append((v.argmax(-1) == p).cpu().numpy())
                for k in greedy:
                    m = models["baseline" if k.startswith("baseline") else "memorylm"]
                    g, _ = greedy_recite(m, p, recite_id, use_memory=(k == "memorylm_cleared"))
                    greedy[k].append((g == p).cpu().numpy())
        r = {"n_passages": len(P)}
        for k, v in conds.items():
            r[f"{k}_acc"], r[f"{k}_ci"] = _acc_ci(np.concatenate(v))
        for k, v in greedy.items():
            r[f"{k}_greedy_acc"], r[f"{k}_greedy_ci"] = _acc_ci(np.concatenate(v))
        r["memory_kb"] = float(np.mean(kb))
        r["write_rate"] = np.mean(wr, 0).tolist()
        # Compression: correctly recalled tokens per KB of memory written.
        # "excess" counts only what the memory adds beyond the lower bound.
        correct = r["memorylm_cleared_acc"] * L
        excess = (r["memorylm_cleared_acc"] - r["baseline_cleared_acc"]) * L
        r["tokens_per_kb"] = correct / r["memory_kb"] if r["memory_kb"] > 0 else 0.0
        r["excess_tokens_per_kb"] = excess / r["memory_kb"] if r["memory_kb"] > 0 else 0.0
        res[str(L)] = r
        print(f"  L={L}: full {r['baseline_full_acc']:.3f} | cleared {r['baseline_cleared_acc']:.3f} | "
              f"memory {r['memorylm_cleared_acc']:.3f} | memory off {r['memorylm_disabled_acc']:.3f} | "
              f"greedy memory {r['memorylm_cleared_greedy_acc']:.3f} | {r['memory_kb']:.1f} KB")
    return res


@torch.no_grad()
def validation_loss(cfg, model, pool: D.Pool, device, windows: int = 2048, batch: int = 64) -> float:
    dt = amp_dtype(cfg, device)
    ws = D.WindowStream(pool, cfg.model.context, cfg.seed + 7)
    windows = min(windows, ws.n)
    tot, n = 0.0, 0
    for s in range(0, windows, batch):
        x = ws.get(s, min(batch, windows - s)).to(device)
        with torch.autocast("cuda", dtype=dt, enabled=dt is not None and device.type == "cuda"):
            tot += lm_loss(model(x[:, :-1])["logits"], x[:, 1:]).item() * x.shape[0]
        n += x.shape[0]
    return tot / n


# ======================================================================
# plots and report
# ======================================================================
COLORS = {"baseline_full": "#7a7a7a", "baseline_cleared": "#c44e52", "memorylm_cleared": "#2a6fdb",
          "memorylm_disabled": "#e3a33b", "baseline": "#c44e52", "memorylm": "#2a6fdb"}
LABELS = {"baseline_full": "Baseline, full context (upper bound)", "baseline_cleared": "Baseline, cleared (lower bound)",
          "memorylm_cleared": "MemoryLM, cleared, memory on", "memorylm_disabled": "MemoryLM, cleared, memory off"}


def make_plots(results: dict, metrics: dict, out: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    files = []
    # 1. loss curves
    fig, ax = plt.subplots(figsize=(7, 4))
    for kind, rows in metrics.items():
        rows = [r for r in rows if "val_loss" in r]
        if rows:
            ax.plot([r["tokens"] / 1e6 for r in rows], [r["val_loss"] for r in rows], label=f"{kind} (validation)",
                    color=COLORS[kind], lw=2)
            ax.plot([r["tokens"] / 1e6 for r in rows], [r["train_loss"] for r in rows], color=COLORS[kind], lw=1, alpha=0.4)
    ax.set(xlabel="training tokens (millions)", ylabel="loss (lower is better)", title="Learning curves")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "loss_curves.png", dpi=130)
    files.append("loss_curves.png")
    plt.close(fig)
    # 2. recall by length
    Ls = [int(k) for k in results["recall"]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, suffix, title in ((axes[0], "_acc", "Teacher-forced recall"), (axes[1], "_greedy_acc", "Free (greedy) recall")):
        for c in LABELS:
            key = c + suffix
            if key not in results["recall"][str(Ls[0])]:
                continue
            y = [results["recall"][str(L)][key] for L in Ls]
            e = [results["recall"][str(L)][key.replace("_acc", "_ci")] for L in Ls]
            ax.errorbar(Ls, y, yerr=e, label=LABELS[c], color=COLORS[c], marker="o", lw=2, capsize=3)
        ax.set(xscale="log", xticks=Ls, xticklabels=[str(L) for L in Ls], xlabel="passage length (tokens)",
               title=title, ylim=(0, 1))
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("token accuracy")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "recall_by_length.png", dpi=130)
    files.append("recall_by_length.png")
    plt.close(fig)
    # 3. memory size and compression
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar([str(L) for L in Ls], [results["recall"][str(L)]["memory_kb"] for L in Ls], color=COLORS["memorylm"])
    axes[0].set(xlabel="passage length (tokens)", ylabel="KB written per passage", title="Memory written")
    axes[1].plot(Ls, [results["recall"][str(L)]["tokens_per_kb"] for L in Ls], marker="o", label="correct tokens / KB",
                 color=COLORS["memorylm"], lw=2)
    axes[1].plot(Ls, [results["recall"][str(L)]["excess_tokens_per_kb"] for L in Ls], marker="o", ls="--",
                 label="above lower bound / KB", color=COLORS["memorylm"], lw=1.5)
    axes[1].set(xscale="log", xticks=Ls, xticklabels=[str(L) for L in Ls], xlabel="passage length (tokens)",
                ylabel="tokens per KB", title="Compression")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "memory_and_compression.png", dpi=130)
    files.append("memory_and_compression.png")
    plt.close(fig)
    # 4. write rates during training
    rows = [r for r in metrics.get("memorylm", []) if "write_rate" in r]
    if rows:
        fig, ax = plt.subplots(figsize=(7, 4))
        x = [r["tokens"] / 1e6 for r in rows]
        ax.plot(x, [r["write_rate"][0] for r in rows], label="knowledge gate", lw=2, color="#2a6fdb")
        ax.plot(x, [r["write_rate"][1] for r in rows], label="rule gate", lw=2, color="#55a868")
        ax.set(xlabel="training tokens (millions)", ylabel="share of tokens written", ylim=(0, 1.05),
               title="Write rate during training")
        ax.legend(frameon=False)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(out / "write_rates.png", dpi=130)
        files.append("write_rates.png")
        plt.close(fig)
    return files


def write_report(results: dict, plan: dict, out: Path) -> str:
    v = results["val_loss"]
    rec = results["recall"]
    Ls = list(rec)
    lines = ["# MemoryLM – results", "",
             f"Run: preset `{results['preset']}`, {plan.get('phase1_tokens', 0):,} phase-1 training tokens per model, "
             f"device {plan.get('device', '?')}, evaluated {results['evaluated_at']}.", "",
             "> **One seed only.** Every number comes from a single training run. Differences of a few "
             "hundredths in loss or a few percentage points in accuracy may be noise. The ± values show "
             "uncertainty from the choice of test passages only, not from training randomness.", "",
             "## 1. Language modelling (validation loss, lower is better)", "",
             "| Model | after phase 1 (same tokens) | after phase 2 |", "|---|---|---|",
             f"| Baseline | {v['baseline_phase1']:.3f} | – |",
             f"| MemoryLM | {v['memorylm_phase1']:.3f} | {v.get('memorylm_phase2', float('nan')):.3f} |", "",
             "## 2. Recall after the context was cleared (token accuracy)", "",
             "| Length | Baseline full context (upper) | Baseline cleared (lower) | MemoryLM memory on | MemoryLM memory off | MemoryLM greedy |",
             "|---|---|---|---|---|---|"]
    for L in Ls:
        r = rec[L]
        f = lambda k: f"{r[k + '_acc']:.2f} ± {r[k + '_ci']:.2f}"
        lines.append(f"| {L} | {f('baseline_full')} | {f('baseline_cleared')} | {f('memorylm_cleared')} | "
                     f"{f('memorylm_disabled')} | {r['memorylm_cleared_greedy_acc']:.2f} |")
    lines += ["", "## 3. Memory cost and compression", "",
              "| Length | KB written | write rate (knowledge, rule) | correct tokens / KB | above lower bound / KB |",
              "|---|---|---|---|---|"]
    for L in Ls:
        r = rec[L]
        lines.append(f"| {L} | {r['memory_kb']:.1f} | {r['write_rate'][0]:.2f}, {r['write_rate'][1]:.2f} | "
                     f"{r['tokens_per_kb']:.2f} | {r['excess_tokens_per_kb']:.2f} |")
    # plain-language summary
    mem_gain = {L: rec[L]["memorylm_cleared_acc"] - rec[L]["memorylm_disabled_acc"] for L in Ls}
    off_vs_lower = {L: rec[L]["memorylm_disabled_acc"] - rec[L]["baseline_cleared_acc"] for L in Ls}
    dv = v["memorylm_phase1"] - v["baseline_phase1"]
    tokens = plan.get("phase1_tokens", 0)
    lines += ["", "## In plain words", "",
              f"- **Language ability:** after the same {tokens:,} training tokens, MemoryLM's validation loss is "
              f"{abs(dv):.3f} {'higher' if dv > 0 else 'lower'} than the baseline's "
              f"({'probably noise' if abs(dv) < 0.03 else 'a noticeable difference, but still one seed'}).",
              "- **What the memory adds** (MemoryLM with memory on minus the same model with memory off): "
              + ", ".join(f"{mem_gain[L] * 100:+.1f} points at {L} tokens" for L in Ls) + ".",
              "- **Memory off vs. lower bound:** "
              + ", ".join(f"{off_vs_lower[L] * 100:+.1f}" for L in Ls)
              + " points. Any difference here is NOT memory: in phase 2 MemoryLM was trained on recitation, so it "
              "learned what typical story beginnings after `<recite>` look like; the baseline never saw that "
              "format. The leak tests (run before training) prove that nothing about the passage itself can reach "
              "the cleared context except through the memory.",
              "- **Upper bound:** the baseline that still sees the passage reaches "
              + ", ".join(f"{rec[L]['baseline_full_acc']:.2f} at {L}" for L in Ls)
              + ". It can copy only while the passage fits into its 128-token window together with the copy.",
              "", "Plots: " + ", ".join(f"![{p}]({p})" for p in results.get("plots", []))]
    text = "\n".join(lines) + "\n"
    (out / "REPORT.md").write_text(text)
    return text


def run_evaluation(cfg, device=None, force: bool = False) -> dict:
    device = device or get_device()
    rd = run_dir(cfg)
    out = rd / "eval"
    out.mkdir(exist_ok=True)
    if (out / "results.json").exists() and not force:
        print("  evaluation already done (eval/results.json)")
        return json.loads((out / "results.json").read_text())
    ddir = D.localize(cfg)
    tok = D.load_tokenizer(cfg)
    rid = D.special_id(tok, "<recite>")
    plan = json.loads((rd / "plan.json").read_text())
    t0 = time.time()
    base = load_model(cfg, "baseline", rd / "baseline_phase1_final.pt", device)
    mem1 = load_model(cfg, "memorylm", rd / "memorylm_phase1_final.pt", device)
    p2 = rd / "memorylm_phase2_final.pt"
    mem2 = load_model(cfg, "memorylm", p2, device) if p2.exists() else mem1
    valid = D.Pool(ddir, "valid")
    vw = int(cfg.eval.get("val_windows", 2048))
    val = {"baseline_phase1": validation_loss(cfg, base, valid, device, vw),
           "memorylm_phase1": validation_loss(cfg, mem1, valid, device, vw)}
    if p2.exists():
        val["memorylm_phase2"] = validation_loss(cfg, mem2, valid, device, vw)
    print("  validation loss:", {k: round(x, 4) for k, x in val.items()})
    recall = evaluate_recall(cfg, {"baseline": base, "memorylm": mem2}, D.Pool(ddir, "recite_eval"), rid, device)
    log = RunLog(rd)
    results = {"preset": cfg.preset, "evaluated_at": time.strftime("%Y-%m-%d %H:%M"), "val_loss": val,
               "recall": recall, "seconds": time.time() - t0}
    results["plots"] = make_plots(results, {k: log.read_metrics(k) for k in ("baseline", "memorylm")}, out)
    atomic_write_json(out / "results.json", results)
    write_report(results, plan, out)
    log.update_status(evaluation={"done": True, "seconds": results["seconds"]})
    print(f"  report written to {out / 'REPORT.md'}")
    return results


if __name__ == "__main__":
    import argparse
    from .config import load_config
    ap = argparse.ArgumentParser(description="Phase 3: evaluate both trained models.")
    ap.add_argument("--preset", default="quick")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    run_evaluation(load_config(a.preset, a.set), force=a.force)
