"""Training: phase 1 (next-token prediction, both models) and phase 2
(recitation episodes, MemoryLM only), plus the phase-0 speed test that sets
the token budget.

    python -m memorylm.train --preset quick --model memorylm --phase auto

`--phase auto` continues from the newest checkpoint, wherever the run
stopped (start, middle of phase 1, between phases, middle of phase 2).
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from . import data as D
from .checkpoint import CheckpointManager, RunLog, atomic_write_json, rng_state, set_rng_state
from .config import load_config, run_root
from .memory import MemoryLM, build_model
from .model_core import count_params


# ======================================================================
# small helpers
# ======================================================================
def run_dir(cfg) -> Path:
    d = run_root(cfg) / f"run_{cfg.run_name}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def amp_dtype(cfg, device) -> Optional[torch.dtype]:
    """Mixed precision: most maths in 16-bit floats (2x faster on a T4, half
    the memory), the sensitive parts (norms, softmax, loss) in 32-bit.
    The T4 has no fast bfloat16, so it uses float16 with loss scaling."""
    mode = cfg.train.mixed_precision
    if device.type != "cuda" or mode == "none":
        return None
    return torch.bfloat16 if mode == "bf16" else torch.float16


def make_scaler(enabled: bool):
    # Loss scaling: fp16 cannot represent very small gradients (they become 0).
    # The loss is multiplied by a large factor before backprop and the
    # gradients divided by it afterwards; the factor shrinks automatically
    # whenever an overflow (inf) appears, and that step is skipped.
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)      # torch >= 2.3
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)         # torch 2.2 (Intel Mac)


def lr_factor(step: int, total: int, warmup: int, min_ratio: float) -> float:
    """Linear warm-up, then cosine decay to min_ratio.
    Warm-up: Adam's moment estimates are unreliable in the first steps; small
    steps avoid an early blow-up. Cosine decay: large steps early for fast
    progress, small steps late to settle into a good minimum."""
    if step < warmup:
        return (step + 1) / warmup
    p = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p))


def make_optimizer(model, cfg) -> torch.optim.Optimizer:
    """AdamW. Weight decay (pulling weights slowly towards 0) only on
    matrices; not on norm gains, biases or the memory scalars, where pulling
    towards 0 would just fight their purpose."""
    scalars = [p for n, p in model.named_parameters() if n in ("memory.beta", "memory.alpha")]
    ids = {id(p) for p in scalars}
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2 and id(p) not in ids]
    groups = [{"params": decay, "weight_decay": cfg.train.weight_decay, "lr_mult": 1.0},
              {"params": no_decay, "weight_decay": 0.0, "lr_mult": 1.0}]
    if scalars:
        # The memory scalars start at exactly 0. With a normal learning rate
        # they stay near 0 for a long time: while they are ~0 the memory's
        # projections get almost no gradient, and while the projections are
        # untrained the scalars only get noise (chicken and egg). A larger
        # step size for these 8 numbers breaks the deadlock without giving up
        # the exact zero start.
        groups.append({"params": scalars, "weight_decay": 0.0, "lr_mult": float(cfg.memory.scalar_lr_mult)})
    kw = {"fused": True} if torch.cuda.is_available() else {}
    return torch.optim.AdamW(groups, lr=cfg.train.phase1.lr, betas=tuple(cfg.train.betas), **kw)


def lm_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # Cross-entropy in float32 for numerical safety under fp16.
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1))


def recitation_forward(model, passages: torch.Tensor, recite_id: int, use_memory: bool = True):
    """One recitation episode per row of `passages` (B, N):
       1. READ the passage (MemoryLM may write into its memory),
       2. CLEAR THE CONTEXT completely (new attention cache, positions restart at 0),
       3. feed <recite> and predict the passage token by token.
    Returns logits (B, N, vocab) for the recited tokens and the memory state.
    For the baseline, step 1 has no lasting effect: without memory, nothing
    survives the cleared context (this is the lower bound)."""
    B = passages.shape[0]
    rin = torch.cat([torch.full((B, 1), recite_id, dtype=passages.dtype, device=passages.device),
                     passages[:, :-1]], 1)
    if isinstance(model, MemoryLM):
        state = model.new_state()
        if use_memory:
            model.process(passages, state, write=True, use_memory=True)
        out = model.process(rin, state, write=False, use_memory=use_memory)
        return out["logits"], state
    return model(rin)["logits"], None


def penalty_lambda(cfg, phase: str, global_step: int, phase1_steps: int) -> float:
    m = cfg.memory
    start = 0 if int(m.write_penalty_from_phase) <= 1 else phase1_steps
    if phase == "phase1" and start > 0:
        return 0.0
    ramp = max(1, int(m.write_penalty_ramp_steps))
    return float(m.write_penalty_lambda) * min(1.0, max(0, global_step - start) / ramp)


# ======================================================================
# the trainer
# ======================================================================
class Divergence(RuntimeError):
    pass


class Trainer:
    def __init__(self, cfg, kind: str, plan: dict, device: Optional[torch.device] = None, verbose: bool = True):
        self.cfg, self.kind, self.plan = cfg, kind, plan
        self.device = device or get_device()
        self.verbose = verbose
        self.dir = run_dir(cfg)
        torch.manual_seed(cfg.seed)
        self.model = build_model(cfg, kind).to(self.device)
        self.amp = amp_dtype(cfg, self.device)
        self.scaler = make_scaler(self.amp == torch.float16)
        self.opt = make_optimizer(self.model, cfg)
        self.ckpt = CheckpointManager(self.dir, kind, cfg.checkpoint.keep_last)
        self.log = RunLog(self.dir)

        ddir = D.localize(cfg)
        meta = D.load_meta(cfg)
        self.tok = D.load_tokenizer(cfg)
        self.recite_id = D.special_id(self.tok, "<recite>")
        T = cfg.model.context
        self.B = int(plan["batch_size"])
        self.stream = D.WindowStream(D.Pool(ddir, "train", meta["train_file"]["pools"]["train"]["tokens"]), T, cfg.seed)
        self.recite_pool = D.Pool(ddir, "recite_train", meta["train_file"]["pools"]["recite_train"]["tokens"])
        vb = min(self.B, 64)
        self.val = D.WindowStream(D.Pool(ddir, "valid"), T, cfg.seed).get(0, vb * cfg.train.val_batches).split(vb)

        self.phase1_steps = int(plan["phase1_steps"])
        self.phase2_steps = int(plan["phase2_steps"]) if kind == "memorylm" else 0
        self.state = {"phase": "phase1", "step": 0, "tokens": 0, "lr_scale": 1.0, "rollbacks": 0,
                      "best_val": float("inf"), "train_seconds": 0.0, "serial": 0,
                      "phase2_n": float(cfg.train.phase2.n_start), "done": False, "last_val": None}
        self.np_rng = np.random.default_rng(cfg.seed + 2)       # phase-2 passage sampling
        self._last_ckpt_time = time.time()
        self._last_eval_time = time.time()
        self._acc = {"loss": 0.0, "n": 0}
        if not self.resume():
            self.save("start")      # step 0 is the first "last good" checkpoint for NaN rollback
        if verbose:
            print(f"[{kind}] {count_params(self.model):,} parameters, batch {self.B}, device {self.device}, "
                  f"phase {self.state['phase']} step {self.state['step']}")

    # ------------------------------------------------------------ checkpointing
    def _snapshot(self) -> dict:
        return {"model": self.model.state_dict(), "opt": self.opt.state_dict(), "scaler": self.scaler.state_dict(),
                "state": dict(self.state), "rng": rng_state(), "np_rng": self.np_rng.bit_generator.state,
                "plan": self.plan, "kind": self.kind, "preset": self.cfg.preset}

    def save(self, reason: str = "") -> None:
        self.state["serial"] += 1
        self.ckpt.save(self._snapshot(), self.state["serial"])
        self._last_ckpt_time = time.time()
        if self.verbose and reason:
            print(f"  [{self.kind}] checkpoint ({reason}) phase {self.state['phase']} step {self.state['step']}")

    def _load(self, ck: dict) -> None:
        self.model.load_state_dict(ck["model"])
        self.opt.load_state_dict(ck["opt"])
        self.scaler.load_state_dict(ck["scaler"])
        # Update IN PLACE: the training loop holds a reference to this dict;
        # replacing it would leave the loop counting steps in a stale copy.
        self.state.clear()
        self.state.update(ck["state"])
        set_rng_state(ck["rng"])
        self.np_rng.bit_generator.state = ck["np_rng"]

    def resume(self) -> bool:
        ck = self.ckpt.load_latest(map_location=self.device)
        if ck is None:
            return False
        if ck.get("plan", {}).get("phase1_steps") != self.plan["phase1_steps"]:
            raise RuntimeError("checkpoint belongs to a different plan; delete the run folder to start over")
        self._load(ck)
        if self.verbose:
            print(f"  [{self.kind}] resumed from checkpoint: phase {self.state['phase']} step {self.state['step']}")
        return True

    def rollback(self, why: str) -> None:
        """NaN/inf: return to the last good checkpoint and continue with half
        the learning rate (and the same data, so nothing is skipped)."""
        lr_scale, rollbacks = self.state["lr_scale"] * 0.5, self.state["rollbacks"] + 1
        if rollbacks > self.cfg.train.max_rollbacks:
            raise Divergence(f"training diverged {rollbacks} times ({why}); giving up")
        self._load(self.ckpt.load_latest(map_location=self.device))
        self.state["lr_scale"], self.state["rollbacks"] = lr_scale, rollbacks
        self.log.update_status(**{self.kind: {"last_rollback": why, "rollbacks": rollbacks}})
        print(f"  [{self.kind}] {why}: rolled back to step {self.state['step']}, lr scale now {lr_scale}")

    # ------------------------------------------------------------ one step
    def _autocast(self):
        if self.amp is None:
            return torch.autocast("cpu", enabled=False)
        return torch.autocast("cuda", dtype=self.amp)

    def _lr(self) -> float:
        p = self.state["phase"]
        c = self.cfg.train.phase1 if p == "phase1" else self.cfg.train.phase2
        total = self.phase1_steps if p == "phase1" else self.phase2_steps
        min_ratio = getattr(c, "min_lr_ratio", 0.1)
        return c.lr * self.state["lr_scale"] * lr_factor(self.state["step"], total, c.warmup_steps, min_ratio)

    def _global_step(self) -> int:
        return self.state["step"] + (self.phase1_steps if self.state["phase"] == "phase2" else 0)

    def compute_loss(self) -> tuple[torch.Tensor, dict]:
        st, B, dev = self.state, self.B, self.device
        lam = penalty_lambda(self.cfg, st["phase"], self._global_step(), self.phase1_steps)
        info = {}
        if st["phase"] == "phase1":
            x = self.stream.get(st["step"] * B, B).to(dev, non_blocking=True)
            out = self.model(x[:, :-1])
            ce = lm_loss(out["logits"], x[:, 1:])
            loss = ce
            if out["expected_writes"] is not None:
                loss = loss + lam * out["expected_writes"]
                info["write_rate"], info["state"] = out["write_rate"], out["state"]
            info["ce"] = ce
            st["tokens"] += B * (x.shape[1] - 1)
            return loss, info
        # ---- phase 2: half recitation episodes, half normal text
        p2 = self.cfg.train.phase2
        n = min(int(p2.n_max), math.ceil(st["phase2_n"] - 1e-9))
        br = max(1, int(round(B * (1 - p2.text_fraction))))
        bt = B - br
        passages = D.sample_passages(self.recite_pool, self.np_rng, n, br).to(dev)
        logits, mstate = recitation_forward(self.model, passages, self.recite_id)
        ce_r = lm_loss(logits, passages)
        ew_r = mstate.expected_writes / max(1, mstate.decisions)
        text_pos = self.phase1_steps * B + st["step"] * bt        # continues after the phase-1 data
        x = self.stream.get(text_pos, bt).to(dev)
        out = self.model(x[:, :-1])
        ce_t = lm_loss(out["logits"], x[:, 1:])
        loss = 0.5 * (ce_r + ce_t) + lam * 0.5 * (ew_r + out["expected_writes"])
        st["tokens"] += br * 2 * n + bt * (x.shape[1] - 1)
        info.update(ce=ce_t, ce_recite=ce_r, recite_n=n, write_rate=out["write_rate"], state=mstate,
                    recite_acc=(logits.argmax(-1) == passages).float().mean())
        return loss, info

    def train_step(self) -> dict:
        self.model.train()
        lr = self._lr()
        for g in self.opt.param_groups:
            g["lr"] = lr * g.get("lr_mult", 1.0)
        with self._autocast():
            loss, info = self.compute_loss()
        self.opt.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.opt)
        # Gradient clipping: if one unlucky batch produces a huge gradient,
        # the step is shortened to length 1.0 instead of throwing the
        # weights far away.
        gnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)
        if self.scaler.is_enabled() or torch.isfinite(gnorm):
            self.scaler.step(self.opt)      # fp16: skipped automatically if gradients contain inf/NaN
        self.scaler.update()
        info.update(loss=loss.detach(), lr=lr, gnorm=gnorm.detach())
        return info

    # ------------------------------------------------------------ evaluation & logging
    @torch.no_grad()
    def validate(self) -> dict:
        self.model.eval()
        tot, rates, kb = 0.0, [], []
        for x in self.val:
            x = x.to(self.device)
            with self._autocast():
                out = self.model(x[:, :-1])
                tot += lm_loss(out["logits"], x[:, 1:]).item()
            if out.get("state") is not None:
                rates.append(out["write_rate"])
                kb.append(out["state"].size_kb())
        r = {"val_loss": tot / len(self.val)}
        if rates:
            r["val_write_rate"] = np.mean(rates, 0).tolist()
            r["val_mem_kb"] = float(np.mean(kb))
        return r

    def _report(self, info: dict, steps_done: int, seconds: float) -> None:
        st = self.state
        total = self.phase1_steps if st["phase"] == "phase1" else self.phase2_steps
        val = self.validate()
        st["last_val"] = val["val_loss"]
        row = {"phase": st["phase"], "step": st["step"], "tokens": st["tokens"], "lr": info["lr"],
               "train_loss": self._acc["loss"] / max(1, self._acc["n"]), **val,
               "steps_per_s": steps_done / max(seconds, 1e-9), "rollbacks": st["rollbacks"]}
        if "write_rate" in info:
            row["write_rate"] = info["write_rate"]
            row["mem_kb"] = info["state"].size_kb()
            # The memory scalars start at 0; their size shows how much the model
            # has started to rely on its memory (knowledge read beta, rule alpha).
            row["mem_beta"] = self.model.memory.beta.detach().cpu().tolist()
            row["mem_alpha"] = self.model.memory.alpha.detach().cpu().tolist()
        if "ce_recite" in info:
            row.update(recite_loss=info["ce_recite"].item(), recite_n=info["recite_n"],
                       recite_acc=info["recite_acc"].item())
        remaining = (total - st["step"]) / max(row["steps_per_s"], 1e-9)
        if self.kind == "memorylm" and st["phase"] == "phase1":
            remaining += self.plan.get("phase2_minutes", 0) * 60
        row["eta_min"] = remaining / 60
        self.log.metric(self.kind, row)
        self.log.update_status(current=self.kind, **{self.kind: {**row, "total_steps": total}})
        if val["val_loss"] < st["best_val"]:
            st["best_val"] = val["val_loss"]
            self.ckpt.save_best(self._snapshot())
        self._acc = {"loss": 0.0, "n": 0}
        if self.verbose:
            extra = f" write {np.round(row['write_rate'], 2).tolist()} mem {row['mem_kb']:.0f}KB" if "write_rate" in row else ""
            extra += f" recite n={row['recite_n']} acc {row['recite_acc']:.2f}" if "recite_acc" in row else ""
            print(f"  [{self.kind}] {st['phase']} {st['step']}/{total} loss {row['train_loss']:.3f} "
                  f"val {val['val_loss']:.3f} lr {info['lr']:.2e} ETA {row['eta_min']:.0f}min{extra}")

    def _finish_phase(self) -> None:
        st = self.state
        torch.save({"model": self.model.state_dict(), "kind": self.kind, "preset": self.cfg.preset,
                    "phase": st["phase"], "val": self.validate()}, self.dir / f"{self.kind}_{st['phase']}_final.pt.tmp")
        Path(self.dir / f"{self.kind}_{st['phase']}_final.pt.tmp").replace(self.dir / f"{self.kind}_{st['phase']}_final.pt")
        finished = st["phase"]
        if st["phase"] == "phase1" and self.phase2_steps > 0:
            st.update(phase="phase2", step=0, best_val=float("inf"))
        else:
            st["done"] = True
        self.save(f"end of {finished}")
        self.log.update_status(**{self.kind: {"phase": st["phase"], "done": st["done"]}})

    # ------------------------------------------------------------ main loop
    def train(self, max_steps: Optional[int] = None, stop_after_phase1: bool = False) -> dict:
        """Run until done. max_steps simulates a session that dies (tests)."""
        cfg, st = self.cfg, self.state
        steps_this_call, t_window, steps_window, bad = 0, time.time(), 0, 0
        while not st["done"]:
            total = self.phase1_steps if st["phase"] == "phase1" else self.phase2_steps
            if st["step"] >= total:
                if st["phase"] == "phase1" and stop_after_phase1 and self.phase2_steps:
                    self._finish_phase()
                    break
                self._finish_phase()
                continue
            if max_steps is not None and steps_this_call >= max_steps:
                break
            t0 = time.time()
            info = self.train_step()
            loss = info["loss"]
            st["step"] += 1
            if st["phase"] == "phase2":
                # Curriculum: passages grow by 0.1 % per step (16 -> 128 in
                # ~2100 steps); start easy, so the model first learns THAT the
                # memory is useful, then how to store longer texts.
                st["phase2_n"] = min(float(cfg.train.phase2.n_max), st["phase2_n"] * float(cfg.train.phase2.n_growth))
            steps_this_call += 1
            steps_window += 1
            st["train_seconds"] += time.time() - t0
            # NaN watch. Checked only every few steps: reading a number back
            # from the GPU forces it to wait, which costs speed. A single
            # non-finite loss is normal with fp16 (the scaler skips that step);
            # three checks in a row, or broken weights, mean real divergence.
            if st["step"] % cfg.train.log_every_steps == 0 or st["step"] == total:
                lf = loss.item()
                if math.isfinite(lf):
                    bad = 0
                    self._acc["loss"] += lf
                    self._acc["n"] += 1
                else:
                    bad += 1
                if bad >= 3 or any(not torch.isfinite(p).all() for p in self.model.parameters()):
                    self.rollback("loss or weights not finite")
                    bad = 0
                    continue
            now = time.time()
            if now - self._last_eval_time >= cfg.train.eval_interval_minutes * 60 or st["step"] == total:
                if self._acc["n"] == 0:
                    self._acc = {"loss": loss.item(), "n": 1}
                self._report(info, steps_window, now - t_window)
                self._last_eval_time, t_window, steps_window = time.time(), time.time(), 0
            every_steps = int(cfg.checkpoint.every_steps or 0)
            if (every_steps and st["step"] % every_steps == 0) or \
                    time.time() - self._last_ckpt_time >= cfg.checkpoint.every_minutes * 60:
                self.save("timer")
        return st


# ======================================================================
# phase 0: batch-size search, speed test, time plan
# ======================================================================
def _oom(e: Exception) -> bool:
    return "out of memory" in str(e).lower()


def find_batch_size(cfg, device) -> int:
    """Largest power-of-two batch whose worst case (a phase-2 step with
    128-token recitations) fits in GPU memory with 10 % to spare. The same
    batch size is then used for BOTH models (identical conditions)."""
    if cfg.train.batch_size != "auto":
        return int(cfg.train.batch_size)
    if device.type != "cuda":
        return 8
    torch.manual_seed(0)
    model = build_model(cfg, "memorylm").to(device)
    opt = make_optimizer(model, cfg)
    dt = amp_dtype(cfg, device)
    total_mem = torch.cuda.get_device_properties(device).total_memory
    V, T, N = cfg.tokenizer.vocab_size, cfg.model.context, int(cfg.train.phase2.n_max)
    b = int(cfg.train.max_batch_size)
    while b >= 4:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            for _ in range(2):
                with torch.autocast("cuda", dtype=dt, enabled=dt is not None):
                    x = torch.randint(0, V, (b - b // 2, T + 1), device=device)
                    p = torch.randint(0, V, (b // 2, N), device=device)
                    logits, st = recitation_forward(model, p, 1)
                    out = model(x[:, :-1])
                    loss = lm_loss(logits, p) + lm_loss(out["logits"], x[:, 1:]) + out["expected_writes"]
                loss.backward()
                opt.zero_grad(set_to_none=True)
            if torch.cuda.max_memory_allocated(device) < 0.9 * total_mem:
                break
        except RuntimeError as e:
            if not _oom(e):
                raise
        b //= 2
    del model, opt
    torch.cuda.empty_cache()
    return b


def speed_test(cfg, plan_seed: dict, device, minutes: float) -> dict:
    """Measure real training speed on the GPU: MemoryLM phase-1 steps for most
    of the time, a few baseline steps, and phase-2 steps at N = 128."""
    import shutil
    from .config import Config
    cfg_tmp = Config(cfg)                       # same settings, separate run folder
    cfg_tmp["run_name"] = f"{cfg.run_name}_speedtest"
    shutil.rmtree(run_dir(cfg_tmp), ignore_errors=True)
    plan = {**plan_seed, "phase1_steps": 10 ** 9, "phase2_steps": 10 ** 9}
    res = {}
    budget = {"memorylm": 0.7, "baseline": 0.2}
    for kind in ("memorylm", "baseline"):
        tr = Trainer(cfg_tmp, kind, plan, device, verbose=False)
        tr.train(max_steps=3)                                   # warm-up (kernel selection, allocator)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0, n = time.time(), 0
        while time.time() - t0 < budget[kind] * minutes * 60 or n < 3:
            tr.train(max_steps=1)
            n += 1
        if device.type == "cuda":
            torch.cuda.synchronize()
        res[f"{kind}_tokens_per_s"] = n * tr.B * cfg.model.context / (time.time() - t0)
        if kind == "memorylm":
            tr.state.update(phase="phase2", step=10 ** 6, phase2_n=float(cfg.train.phase2.n_max))
            tr.train(max_steps=2)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0, n = time.time(), 0
            while time.time() - t0 < 0.1 * minutes * 60 or n < 3:
                tr.train(max_steps=1)
                n += 1
            if device.type == "cuda":
                torch.cuda.synchronize()
            res["phase2_step_s"] = (time.time() - t0) / n
        del tr
    shutil.rmtree(run_dir(cfg_tmp), ignore_errors=True)
    return res


def make_plan(cfg, device=None, phase0_seconds_so_far: float = 0.0) -> dict:
    """Decide batch size, phase-1 token budget and phase-2 steps.
    The budget is set so MemoryLM's phase 1 takes ~60 minutes; if the whole
    plan (phase 0 + baseline + MemoryLM + phase 2 + evaluation) would exceed
    the GPU time limit, the phase-1 budget is shrunk automatically."""
    device = device or get_device()
    t, T = cfg.train, cfg.model.context
    B = find_batch_size(cfg, device)
    plan = {"batch_size": B, "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    fixed_tokens = t.phase1.tokens != "auto"
    fixed_p2 = t.phase2.steps != "auto"
    if fixed_tokens and fixed_p2:
        plan.update(phase1_tokens=int(t.phase1.tokens), phase2_steps=int(t.phase2.steps), phase2_minutes=0)
    else:
        sp = speed_test(cfg, {"batch_size": B}, device, t.speed_test_minutes)
        plan.update(sp)
        tok = int(t.phase1.tokens) if fixed_tokens else int(sp["memorylm_tokens_per_s"] * t.phase1.target_minutes_memorylm * 60 * 0.95)
        p2 = int(t.phase2.steps) if fixed_p2 else int(t.phase2.minutes * 60 * 0.95 / sp["phase2_step_s"])
        eval_min = float(t.eval_minutes)
        def total_min(tokens):
            return (phase0_seconds_so_far / 60 + t.speed_test_minutes
                    + tokens / sp["baseline_tokens_per_s"] / 60 + tokens / sp["memorylm_tokens_per_s"] / 60
                    + p2 * sp["phase2_step_s"] / 60 + eval_min)
        limit = t.total_gpu_hours * 60
        plan["shrunk"] = False
        if total_min(tok) > limit and not fixed_tokens:
            per_token = (1 / sp["baseline_tokens_per_s"] + 1 / sp["memorylm_tokens_per_s"]) / 60
            tok = int(max(0.1 * tok, tok - (total_min(tok) - limit) / per_token))
            plan["shrunk"] = True
        plan.update(phase1_tokens=tok, phase2_steps=p2, phase2_minutes=p2 * sp["phase2_step_s"] / 60,
                    estimate_minutes={"phase0": phase0_seconds_so_far / 60 + t.speed_test_minutes,
                                      "phase1_baseline": tok / sp["baseline_tokens_per_s"] / 60,
                                      "phase1_memorylm": tok / sp["memorylm_tokens_per_s"] / 60,
                                      "phase2_memorylm": p2 * sp["phase2_step_s"] / 60,
                                      "phase3_eval": eval_min, "total": total_min(tok)})
    plan["phase1_steps"] = max(1, plan["phase1_tokens"] // (B * T))
    return plan


def load_or_make_plan(cfg, device=None, phase0_seconds: float = 0.0) -> dict:
    p = run_dir(cfg) / "plan.json"
    if p.exists():
        import json
        return json.loads(p.read_text())
    plan = make_plan(cfg, device, phase0_seconds)
    atomic_write_json(p, plan)
    return plan


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train one model (resumes automatically).")
    ap.add_argument("--preset", default="quick")
    ap.add_argument("--model", required=True, choices=["baseline", "memorylm"])
    ap.add_argument("--phase", default="auto", choices=["auto", "1", "2"])
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. train.phase1.lr=5e-4")
    a = ap.parse_args(argv)
    cfg = load_config(a.preset, a.set)
    D.prepare_data(cfg, min_train_tokens=None if cfg.train.phase1.tokens == "auto" else int(cfg.train.phase1.tokens) + 10 ** 6)
    plan = load_or_make_plan(cfg)
    tr = Trainer(cfg, a.model, plan)
    if a.phase == "2" and tr.state["phase"] == "phase1":
        raise SystemExit("phase 1 is not finished yet")
    tr.train(stop_after_phase1=(a.phase == "1"))


if __name__ == "__main__":
    main()
