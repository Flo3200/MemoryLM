"""Tests of the shared transformer core (milestone M1)."""
import torch
import torch.nn.functional as F

from memorylm.config import load_config
from memorylm.model_core import CoreTransformer, KVCache, core_config_from, count_params


def tiny_core(ffn=None, seed=0):
    torch.manual_seed(seed)
    cfg = load_config("tiny")
    return CoreTransformer(core_config_from(cfg, ffn or cfg.model.ffn_hidden_memorylm)), cfg


def test_causal_mask_no_future_leak():
    # Changing tokens AFTER position t must not change the prediction AT t.
    model, cfg = tiny_core()
    model.eval()
    T = cfg.model.context
    a = torch.randint(0, cfg.tokenizer.vocab_size, (2, T))
    b = a.clone()
    b[:, 40:] = torch.randint(0, cfg.tokenizer.vocab_size, (2, T - 40))
    la, lb = model(a)["logits"], model(b)["logits"]
    assert torch.allclose(la[:, :40], lb[:, :40], atol=1e-5)
    assert not torch.allclose(la[:, 40:], lb[:, 40:], atol=1e-3)


def test_causal_mask_gradient():
    # Stronger form: the gradient of the output at position t with respect
    # to every LATER input embedding is exactly zero.
    model, cfg = tiny_core()
    T = 32
    idx = torch.randint(0, cfg.tokenizer.vocab_size, (1, T))
    emb = model.tok_emb(idx).detach().requires_grad_(True)
    cos, sin = model.rope_cos[:T], model.rope_sin[:T]
    x = emb
    for blk in model.blocks:
        x = blk(x, cos, sin)
    x[0, 10].sum().backward()
    assert emb.grad[0, 11:].abs().max() == 0
    assert emb.grad[0, :11].abs().max() > 0


def test_chunked_with_cache_equals_full_sequence():
    # MemoryLM processes text in chunks with a key/value cache. Without memory
    # this must give exactly the same logits as one pass over the whole text.
    model, cfg = tiny_core()
    model.eval()
    T, C = cfg.model.context, cfg.memory.chunk_size
    idx = torch.randint(0, cfg.tokenizer.vocab_size, (3, T))
    full = model(idx)["logits"]
    cache = KVCache(cfg.model.n_layers)
    parts = [model(idx[:, s:s + C], pos0=s, cache=cache)["logits"] for s in range(0, T, C)]
    assert torch.allclose(full, torch.cat(parts, dim=1), atol=1e-5)


def test_weight_tying_and_param_count():
    model, cfg = tiny_core()
    assert model.output_weight() is model.tok_emb.weight
    m, V = cfg.model, cfg.tokenizer.vocab_size
    d, H = m.d_model, m.ffn_hidden_memorylm
    per_layer = 4 * d * d + 3 * d * H + 2 * d          # attention + SwiGLU + two norm gains
    expected = V * d + m.n_layers * per_layer + d       # + tied embedding + final norm
    assert count_params(model) == expected


def test_overfits_one_batch():
    # Sanity check of the whole forward/backward path: a model must be able
    # to memorise a single small batch.
    model, cfg = tiny_core()
    idx = torch.randint(0, cfg.tokenizer.vocab_size, (4, 33))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    losses = []
    for _ in range(40):
        logits = model(idx[:, :-1])["logits"]
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), idx[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.5 * losses[0]
