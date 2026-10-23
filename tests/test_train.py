"""Required pre-training tests for the training loop (milestone M4).
All run on CPU with the tiny preset in well under a minute."""
import numpy as np
import torch

from memorylm import data as D
from memorylm.checkpoint import RunLog
from memorylm.memory import MemoryLM
from memorylm.train import Trainer, make_plan, penalty_lambda, recitation_forward, run_dir


def trainer(cfg, kind, plan=None, **kw):
    plan = plan or make_plan(cfg, torch.device("cpu"))
    return Trainer(cfg, kind, plan, torch.device("cpu"), verbose=False, **kw)


def params(model):
    return torch.cat([p.detach().flatten() for p in model.parameters()])


# ---------------------------------------------------------------- loss falls
def test_tiny_training_reduces_loss(tiny_cfg):
    for kind in ("baseline", "memorylm"):
        cfg = tiny_cfg(f"run_name=loss_{kind}", "train.eval_interval_minutes=0")
        tr = trainer(cfg, kind)
        tr.train(stop_after_phase1=True)
        rows = [r for r in RunLog(run_dir(cfg)).read_metrics(kind) if r["phase"] == "phase1"]
        assert len(rows) >= 3
        assert rows[-1]["val_loss"] < rows[0]["val_loss"] - 0.5, (kind, rows[0]["val_loss"], rows[-1]["val_loss"])
        assert (run_dir(cfg) / f"{kind}_phase1_final.pt").exists()


# ---------------------------------------------------------------- exact resume
def test_checkpoint_round_trip_gives_identical_steps(tiny_cfg):
    """Run A trains without interruption. Run B is 'killed' in the middle of
    phase 2 and a NEW trainer resumes from the checkpoint. Both must end
    with bit-identical weights (same data, same random passages, same LR)."""
    plan_overrides = ["train.phase1.tokens=4096"]                  # 8 phase-1 steps of 8 x 64 tokens
    cfg_a = tiny_cfg("run_name=resume_a", *plan_overrides)
    cfg_b = tiny_cfg("run_name=resume_b", *plan_overrides)
    plan = make_plan(cfg_a, torch.device("cpu"))
    assert plan["phase1_steps"] == 8 and plan["phase2_steps"] == 10

    a = trainer(cfg_a, "memorylm", plan)
    a.train()
    assert a.state["done"]

    b = trainer(cfg_b, "memorylm", plan)
    b.train(max_steps=8 + 7)                    # dies at phase-2 step 7; last checkpoint: phase-2 step 5
    assert b.state["phase"] == "phase2" and b.state["step"] == 7
    del b                                       # the session ends; everything in RAM is lost
    torch.manual_seed(999)                      # a new session has different random state
    np.random.seed(999)
    b2 = trainer(cfg_b, "memorylm", plan)
    assert b2.state["phase"] == "phase2" and b2.state["step"] == 5
    b2.train()
    assert b2.state["done"]
    assert torch.equal(params(a.model), params(b2.model))
    assert a.state["tokens"] == b2.state["tokens"]


def test_resume_between_phases(tiny_cfg):
    cfg = tiny_cfg("train.phase1.tokens=2048")
    plan = make_plan(cfg, torch.device("cpu"))
    t = trainer(cfg, "memorylm", plan)
    t.train(stop_after_phase1=True)
    t2 = trainer(cfg, "memorylm", plan)
    assert t2.state["phase"] == "phase2" and t2.state["step"] == 0
    t2.train()
    assert t2.state["done"]


# ---------------------------------------------------------------- NaN rollback
def test_nan_rolls_back_and_lowers_lr(tiny_cfg):
    cfg = tiny_cfg("train.phase1.tokens=8192")
    plan = make_plan(cfg, torch.device("cpu"))
    t = trainer(cfg, "baseline", plan)
    t.train(max_steps=6)                                    # checkpoint at step 5
    with torch.no_grad():
        t.model.core.blocks[0].ffn.w_up.weight[0, 0] = float("nan")   # simulate a blow-up
    t.train()
    assert t.state["done"] and t.state["rollbacks"] == 1 and t.state["lr_scale"] == 0.5
    assert all(torch.isfinite(p).all() for p in t.model.parameters())


# ---------------------------------------------------------------- leak test
def _random_passages(cfg, n=16, b=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(10, cfg.tokenizer.vocab_size, (b, n), generator=g)


def test_leak_cleared_context_without_memory_is_chance(tiny_cfg):
    """With the memory disabled and the context cleared, nothing about the
    passage may reach the recitation: (1) the first recited token is predicted
    identically for every passage, (2) accuracy on random passages is at
    chance level. Run on a model that was trained on recitation."""
    cfg = tiny_cfg("train.phase1.tokens=2048", "train.phase2.steps=15")
    t = trainer(cfg, "memorylm")
    t.train()
    model = t.model.eval()
    recite = D.special_id(t.tok, "<recite>")
    p = _random_passages(cfg)
    with torch.no_grad():
        logits, _ = recitation_forward(model, p, recite, use_memory=False)
        first = logits[:, 0]
        assert torch.allclose(first, first[:1].expand_as(first), atol=1e-5)
        acc = (logits.argmax(-1) == p).float().mean().item()
        assert acc < 5 / cfg.tokenizer.vocab_size + 0.01, acc
        # Sanity check of the test itself: WITH memory, the first prediction
        # does depend on the passage (information flows only through memory).
        with_mem, _ = recitation_forward(model, p, recite, use_memory=True)
        assert not torch.allclose(with_mem[:, 0], with_mem[:1, 0].expand_as(with_mem[:, 0]), atol=1e-5)


def test_leak_baseline_cleared_context(tiny_cfg):
    cfg = tiny_cfg()
    t = trainer(cfg, "baseline")
    p = _random_passages(cfg)
    with torch.no_grad():
        logits, state = recitation_forward(t.model.eval(), p, D.special_id(t.tok, "<recite>"))
    assert state is None
    assert torch.allclose(logits[:, 0], logits[:1, 0].expand_as(logits[:, 0]), atol=1e-5)


def test_recitation_starts_with_empty_memory(tiny_cfg):
    cfg = tiny_cfg()
    t = trainer(cfg, "memorylm")
    p = _random_passages(cfg)
    rid = D.special_id(t.tok, "<recite>")
    with torch.no_grad():
        l1, s1 = recitation_forward(t.model, p, rid)
        l2, s2 = recitation_forward(t.model, p, rid)        # same passage again: no carry-over
    assert s1 is not s2 and torch.equal(l1, l2)
    assert len(s1.knowledge) == p.shape[1]                  # exactly one slot per read token, nothing older


# ---------------------------------------------------------------- schedules
def test_write_penalty_schedule(tiny_cfg):
    cfg = tiny_cfg()
    assert penalty_lambda(cfg, "phase1", 50, 40) == 0.0                     # off during phase 1
    assert 0 < penalty_lambda(cfg, "phase2", 45, 40) < cfg.memory.write_penalty_lambda
    assert penalty_lambda(cfg, "phase2", 40 + 1000, 40) == cfg.memory.write_penalty_lambda
    cfg1 = tiny_cfg("memory.write_penalty_from_phase=1")
    assert penalty_lambda(cfg1, "phase1", 10, 40) > 0


def test_recitation_length_grows_and_caps(tiny_cfg):
    cfg = tiny_cfg("train.phase1.tokens=1024", "train.phase2.steps=6", "train.phase2.n_growth=1.2",
                   "train.phase2.n_start=8", "train.phase2.n_max=12", "train.log_every_steps=1",
                   "train.eval_interval_minutes=0")
    t = trainer(cfg, "memorylm")
    t.train()
    rows = [r for r in RunLog(run_dir(cfg)).read_metrics("memorylm") if r["phase"] == "phase2"]
    ns = [r["recite_n"] for r in rows]
    assert ns == sorted(ns) and ns[0] == 8 and ns[-1] == 12
    assert isinstance(t.model, MemoryLM)


def test_speed_test_and_plan_shrinking(tiny_cfg):
    """Phase 0 on CPU: the speed test runs, the budget follows from the
    measured speed, and an impossible time limit shrinks the budget."""
    cfg = tiny_cfg("train.phase1.tokens=auto", "train.phase2.steps=auto", "train.speed_test_minutes=0.03",
                   "train.phase1.target_minutes_memorylm=1", "train.phase2.minutes=1", "train.eval_minutes=1")
    plan = make_plan(cfg, torch.device("cpu"))
    assert plan["memorylm_tokens_per_s"] > 0 and plan["baseline_tokens_per_s"] > plan["memorylm_tokens_per_s"] * 0.8
    assert abs(plan["estimate_minutes"]["phase1_memorylm"] - 0.95) < 0.05 and not plan["shrunk"]
    assert plan["phase2_steps"] > 0 and plan["phase1_steps"] == plan["phase1_tokens"] // (8 * 64)
    tight = make_plan(tiny_cfg("run_name=tight", "train.phase1.tokens=auto", "train.phase2.steps=auto",
                               "train.speed_test_minutes=0.03", "train.phase1.target_minutes_memorylm=1",
                               "train.phase2.minutes=1", "train.eval_minutes=1", "train.total_gpu_hours=0.05"),
                      torch.device("cpu"))
    assert tight["shrunk"] and tight["phase1_tokens"] < plan["phase1_tokens"]
    assert tight["estimate_minutes"]["total"] <= 0.05 * 60 + 0.01
