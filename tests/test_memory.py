"""Tests of the memory module (milestone M2)."""
import pytest
import torch
import torch.nn.functional as F

from memorylm.config import load_config
from memorylm.memory import BaselineLM, MemoryLM, param_report


def make(overrides=(), seed=0, live=False):
    """Tiny MemoryLM. live=True sets the memory scalars to non-zero values,
    so that the memory actually changes the output (needed for leak tests)."""
    torch.manual_seed(seed)
    cfg = load_config("tiny", list(overrides))
    model = MemoryLM(cfg)
    if live:
        with torch.no_grad():
            model.memory.beta.fill_(1.0)
            model.memory.alpha.fill_(1.0)
            for p in model.memory.parameters():
                if p.dim() == 2:
                    p.mul_(20)          # make memory effects large enough to see
    model.eval()
    return model, cfg


def tokens(cfg, B=2, T=None, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, cfg.tokenizer.vocab_size, (B, T or cfg.model.context), generator=g)


# ---------------------------------------------------------------- zero init
def test_zero_init_equals_core():
    """All memory scalars start at 0, so MemoryLM must behave exactly like its
    own core without memory, even though entries ARE being written."""
    model, cfg = make()
    idx = tokens(cfg)
    with torch.no_grad():
        out = model(idx)
        no_mem = model(idx, use_memory=False)["logits"]
        core = model.core(idx)["logits"]
    assert out["state"].knowledge.n_written().sum() > 0      # memory really was filled
    assert torch.equal(out["logits"], no_mem)                # bit-for-bit identical
    assert torch.allclose(out["logits"], core, atol=1e-5)    # same as one full pass of the core


# ---------------------------------------------------------------- causality
def test_no_future_leak_through_memory():
    """Change one token in the middle of a chunk; no earlier position may change,
    including earlier positions of the SAME chunk (a write-before-read bug
    would show up exactly there)."""
    model, cfg = make(live=True)
    C = cfg.memory.chunk_size
    a = tokens(cfg)
    t = C + C // 2                                          # middle of chunk 1
    b = a.clone()
    b[:, t:] = (b[:, t:] + 7) % cfg.tokenizer.vocab_size
    with torch.no_grad():
        la, lb = model(a)["logits"], model(b)["logits"]
    assert torch.allclose(la[:, :t], lb[:, :t], atol=1e-5)
    assert not torch.allclose(la[:, t:], lb[:, t:], atol=1e-3)


def test_memory_affects_only_later_chunks():
    """With the memory switched on, a token's own chunk is unaffected by the
    memory; later chunks are. Proves writes become readable only from chunk j+1."""
    model, cfg = make(live=True)
    C = cfg.memory.chunk_size
    idx = tokens(cfg)
    with torch.no_grad():
        on, off = model(idx)["logits"], model(idx, use_memory=False)["logits"]
    assert torch.equal(on[:, :C], off[:, :C])               # chunk 0: memory still empty
    assert not torch.allclose(on[:, C:], off[:, C:], atol=1e-3)


def test_gradient_never_flows_from_future():
    model, cfg = make(live=True)
    model.train()
    embs = []
    def keep(module, inputs, output):       # remember each chunk's embeddings (returns None: output unchanged)
        output.retain_grad()
        embs.append(output)
    h = model.core.tok_emb.register_forward_hook(keep)
    idx = tokens(cfg, B=1)
    t = 20
    out = model(idx)["logits"]
    h.remove()
    out[0, t].sum().backward()
    grads = torch.cat([e.grad if e.grad is not None else torch.zeros_like(e) for e in embs], 1)
    assert grads[0, t + 1:].abs().max() == 0
    assert grads[0, :t + 1].abs().max() > 0


# ---------------------------------------------------------------- episodes
def test_memory_empty_at_start_of_every_episode():
    model, cfg = make(live=True)
    assert model.new_state().is_empty()
    a, b = tokens(cfg, seed=1), tokens(cfg, seed=2)
    with torch.no_grad():
        model(a)                                            # episode 1 fills a memory ...
        second = model(b)["logits"]                         # ... episode 2 must not see it
        fresh, _ = make(live=True)
        alone = fresh(b)["logits"]
    assert torch.equal(second, alone)


def test_closed_gates_mean_no_memory():
    """If every gate is closed, nothing is written and the output equals the
    memory-free output exactly, even with large memory scalars."""
    model, cfg = make(live=True)
    with torch.no_grad():
        model.core.extra_head.bias.fill_(-100.0)
    idx = tokens(cfg)
    with torch.no_grad():
        out = model(idx)
        off = model(idx, use_memory=False)["logits"]
    assert out["write_rate"] == [0.0, 0.0]
    assert out["state"].size_kb() == 0
    assert torch.equal(out["logits"], off)


# ---------------------------------------------------------------- gradients
def test_backprop_through_writes():
    """The loss on a later chunk must send gradient back into the write
    projections and the write gates (backprop through time within an episode)."""
    model, cfg = make(live=True)
    model.train()
    idx = tokens(cfg)
    out = model(idx)
    C = cfg.memory.chunk_size
    logits = out["logits"][:, C:-1]
    F.cross_entropy(logits.reshape(-1, logits.shape[-1]), idx[:, C + 1:].reshape(-1)).backward()
    mem = model.memory
    for p in (mem.w_kval.weight, mem.w_kkey.weight, mem.w_rb.weight, mem.w_ra.weight, mem.w_rkey.weight,
              mem.beta, mem.alpha, model.core.extra_head.weight):
        assert p.grad is not None and p.grad.abs().sum() > 0


def test_write_penalty_value():
    model, cfg = make()
    out = model(tokens(cfg))
    # Both gates start at sigmoid(0.5) ~ 0.62 per token and store.
    expected = 2 * torch.sigmoid(torch.tensor(cfg.memory.gate_bias_init)).item()
    assert abs(out["expected_writes"].item() - expected) < 0.1
    out["expected_writes"].backward()
    assert model.core.extra_head.bias.grad.abs().sum() > 0



def test_closed_gates_get_no_task_gradient_but_the_floor_reopens_them():
    """Why the quick run's gates died: a closed gate's slot is masked out of
    every read, so the task loss gives it exactly zero gradient and nothing
    can reopen it. The write floor (min_write_rate) does push it back up."""
    model, cfg = make(["memory.min_write_rate=0.1"], live=True)
    model.train()
    with torch.no_grad():
        model.core.extra_head.bias.fill_(-5.0)
    idx = tokens(cfg)
    out = model(idx)
    assert out["write_rate"] == [0.0, 0.0]
    logits = out["logits"][:, :-1]
    F.cross_entropy(logits.reshape(-1, logits.shape[-1]), idx[:, 1:].reshape(-1)).backward()
    g = model.core.extra_head
    assert g.bias.grad is None or g.bias.grad.abs().sum() == 0
    model.zero_grad()
    out = model(idx)
    assert out["write_floor"].item() > 0
    out["write_floor"].backward()
    assert (g.bias.grad < 0).all()          # gradient descent raises the bias = opens the gates
    model.zero_grad()
    assert make(["memory.min_write_rate=0"])[0](idx)["write_floor"] is None   # 0 = off

# ---------------------------------------------------------------- rule store maths
def test_rule_correction_equals_full_low_rank_matrix():
    """sum_i g_i * b_i * (a_i . x) must equal (sum_i g_i * b_i a_i^T) x,
    i.e. a genuine low-rank correction of W_up, built without the matrix."""
    model, cfg = make(live=True)
    idx = tokens(cfg, B=1)
    with torch.no_grad():
        state = model.new_state()
        model.process(idx[:, :cfg.memory.chunk_size * 2], state)
        h2 = torch.randn(1, 1, cfg.model.d_model)
        corr = model.memory.rule_correction(0, h2, state)
        q = model.memory.q_rule[0](h2)
        vals, sel, ok = model.memory._select(q, state.rule)
        a, b = state.rule.vals
        W = torch.zeros(b.shape[-1], a.shape[-1])
        for j in range(sel.shape[-1]):
            if ok[0, 0, j]:
                i = sel[0, 0, j]
                g = torch.tanh(cfg.memory.rule_score_scale * vals[0, 0, j])
                W += g * torch.outer(b[0, i], a[0, i])
        explicit = model.memory.alpha[0] * (W @ h2[0, 0])
    assert torch.allclose(corr[0, 0], explicit, atol=1e-4)


# ---------------------------------------------------------------- capacity, options
def test_capacity_and_compaction():
    model, cfg = make(["memory.capacity=40"])
    with torch.no_grad():
        out = model(tokens(cfg))
    st = out["state"]
    assert len(st.knowledge) <= 40 and len(st.rule) <= 40
    assert st.knowledge.valid.all()          # all gates open at init: only written entries kept


@pytest.mark.parametrize("opts", [["memory.gate_mode=soft"], ["memory.sparse_values=true", "memory.sparse_value_k=4"],
                                  ["memory.product_keys=true", "memory.product_key_candidates=4"]])
def test_switchable_options_run(opts):
    model, cfg = make(opts, live=True)
    out = model(tokens(cfg))
    assert torch.isfinite(out["logits"]).all()
    if "memory.sparse_values=true" in opts:
        dense, _ = make(live=True)
        assert out["state"].size_kb() < dense(tokens(cfg))["state"].size_kb()


def test_param_counts_match_within_2_percent():
    for preset in ("quick", "full", "tiny"):
        r = param_report(load_config(preset))
        assert r["relative_difference"] < 0.02, (preset, r)
    r = param_report(load_config("quick"))
    assert 4e6 < r["memorylm_params"] < 5e6 and 4e6 < r["baseline_params"] < 5e6


def test_baseline_interface():
    cfg = load_config("tiny")
    torch.manual_seed(0)
    out = BaselineLM(cfg)(tokens(cfg))
    assert out["logits"].shape == (2, cfg.model.context, cfg.tokenizer.vocab_size)
