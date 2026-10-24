"""The external vector memory of MemoryLM, and the two model wrappers.

Two stores, shared by all layers, emptied after every episode:

* KNOWLEDGE STORE: entries (key, value). Each layer can read from it with a
  softmax over the 8 best-matching keys, like a second attention that looks
  at stored vectors instead of at the current context. The result is added to
  the attention output.

* RULE STORE: entries (key, a, b). A retrieved entry changes HOW the
  feed-forward layer processes the current token: it adds the rank-1 matrix
  b a^T to the expansion matrix W_up, weighted by a bounded score g. Summed
  over 8 entries this is a low-rank correction of W_up, computed as
  sum_i g_i * b_i * (a_i . x) without ever building the 704 x 256 matrix.

Writing: the model's output layer has 2 extra rows, one write gate per store.
A positive gate means "store this token". Stored vectors are computed from
the hidden state after layer 2 by small projections.

Causality: text is processed in chunks of 32 tokens. Writes happen after a
chunk has been processed, so entries from chunk j are first visible in chunk
j+1. A token can therefore never read anything derived from itself or from
later tokens.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_core import CoreTransformer, KVCache, core_config_from, count_params

NEG = -1e4      # "minus infinity" for masked scores that is still safe in float32 maths


class Store:
    """One memory store for a batch of independent episodes.

    Every processed token gets a slot (written or not); `valid` marks the
    slots whose gate said "write". Unwritten slots are masked, so they behave
    exactly like empty memory. If the slots exceed the capacity, the store is
    compacted: written entries are kept, newest first.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.key: Optional[torch.Tensor] = None     # (B, S, key_dim)
        self.vals: list[torch.Tensor] = []          # each (B, S, dim)
        self.valid: Optional[torch.Tensor] = None   # (B, S) bool

    def __len__(self) -> int:
        return 0 if self.key is None else self.key.shape[1]

    def append(self, key: torch.Tensor, vals: list[torch.Tensor], valid: torch.Tensor) -> None:
        if self.key is None:
            self.key, self.vals, self.valid = key, list(vals), valid
        else:
            # torch.cat keeps the autograd graph: gradients flow back through
            # the write into earlier chunks (backprop through time).
            self.key = torch.cat([self.key, key], 1)
            self.vals = [torch.cat([o, n], 1) for o, n in zip(self.vals, vals)]
            self.valid = torch.cat([self.valid, valid], 1)
        if len(self) > self.capacity:
            self._compact()

    def _compact(self) -> None:
        S = len(self)
        recency = torch.arange(S, device=self.valid.device).float() / S        # newer = larger
        rank = self.valid.float() * 2.0 + recency                              # written first, then newest
        keep = rank.topk(self.capacity, dim=1).indices.sort(dim=1).values      # (B, capacity), keep order
        take = lambda t: t.gather(1, keep.unsqueeze(-1).expand(-1, -1, t.shape[-1]))
        self.key, self.vals, self.valid = take(self.key), [take(v) for v in self.vals], self.valid.gather(1, keep)

    def n_written(self) -> torch.Tensor:
        return torch.zeros(()) if self.valid is None else self.valid.sum(1).float()

    def nonzeros(self) -> int:
        """Stored non-zero numbers in written entries (for the memory size in KB)."""
        if self.key is None:
            return 0
        m = self.valid.unsqueeze(-1)
        return int(sum(((t != 0) & m).sum().item() for t in [self.key, *self.vals]))


class MemoryState:
    """The per-episode contents: one knowledge store and one rule store."""

    def __init__(self, capacity: int, trace: bool = False):
        self.knowledge = Store(capacity)
        self.rule = Store(capacity)
        self.trace = [] if trace else None     # per-token gate values, for the memory inspector
        self.expected_writes = 0.0      # sum of sigmoid(gate) over all write decisions (for the penalty)
        self.decisions = 0              # number of tokens that had a write decision
        self.hard_writes = torch.zeros(2)
        self.ste_writes = torch.zeros(2)  # hard writes per store, with the sigmoid gradient (write floor)

    def write_floor(self, min_rate: float):
        """Hinge loss that keeps each store's REAL write rate >= min_rate.
        A closed gate gets no gradient from the task loss (its slot is masked
        out of every read), so without this floor gates that close once stay
        closed for good. The rate counts hard writes (an expected sigmoid
        rate can be 10 % while every logit is below 0 and nothing is
        written); its gradient is the sigmoid's, as in the gate itself.
        None if off or nothing was decided."""
        if min_rate <= 0 or not self.decisions:
            return None
        return torch.relu(min_rate - self.ste_writes / self.decisions).sum()

    def is_empty(self) -> bool:
        return len(self.knowledge) == 0 and len(self.rule) == 0

    def size_kb(self, bytes_per_number: int = 2) -> float:
        """Average memory size of ONE episode in kilobytes. Stored as fp16
        (2 bytes per number); only non-zeros are counted, so sparse values
        (if switched on) really shrink the reported size."""
        if self.knowledge.key is None:
            return 0.0
        batch = self.knowledge.key.shape[0]
        return (self.knowledge.nonzeros() + self.rule.nonzeros()) * bytes_per_number / 1024 / batch


def sparsify(x: torch.Tensor, k: int) -> torch.Tensor:
    """Keep the k largest-magnitude numbers of each vector, zero the rest."""
    if k >= x.shape[-1]:
        return x
    idx = x.abs().topk(k, dim=-1).indices
    return x * torch.zeros_like(x).scatter(-1, idx, 1.0)


class MemoryModule(nn.Module):
    """All learned parameters of the memory (the stores themselves are not
    parameters: they are filled anew from the text in every episode)."""

    def __init__(self, mcfg, d_model: int, ffn_hidden: int, n_layers: int):
        super().__init__()
        self.cfg = mcfg
        dk = mcfg.key_dim
        self.key_dim = dk
        # Per-layer query projections: each layer asks its own question.
        self.q_knowledge = nn.ModuleList(nn.Linear(d_model, dk, bias=False) for _ in range(n_layers))
        self.q_rule = nn.ModuleList(nn.Linear(d_model, dk, bias=False) for _ in range(n_layers))
        # Per-layer mixing scalars, initialised to 0: at the start the memory
        # contributes exactly nothing, so MemoryLM begins as the plain core and
        # the optimiser "turns the memory on" only as far as it helps.
        self.beta = nn.Parameter(torch.zeros(n_layers))    # knowledge read strength
        self.alpha = nn.Parameter(torch.zeros(n_layers))   # rule correction strength
        # Write projections (from the layer-2 hidden state, shared by both stores' gates).
        self.w_kkey = nn.Linear(d_model, dk, bias=False)
        self.w_kval = nn.Linear(d_model, d_model, bias=False)
        self.w_rkey = nn.Linear(d_model, dk, bias=False)
        self.w_ra = nn.Linear(d_model, d_model, bias=False)
        self.w_rb = nn.Linear(d_model, ffn_hidden, bias=False)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

    # ---- reading -------------------------------------------------------
    def _select(self, q: torch.Tensor, store: Store):
        """Brute-force dot product of every query with every stored key, then
        the top-k. Returns raw scores (B,C,k), slot indices (B,C,k) and which
        of the selected slots are real written entries."""
        scores = q.float() @ store.key.float().transpose(1, 2)            # (B, C, S)
        if self.cfg.product_keys:
            scores = self._product_key_filter(q.float(), store.key.float(), scores)
        scores = scores.masked_fill(~store.valid.unsqueeze(1), NEG)       # mask empty slots
        k = min(self.cfg.top_k, scores.shape[-1])
        vals, idx = scores.topk(k, dim=-1)
        ok = store.valid.unsqueeze(1).expand(-1, q.shape[1], -1).gather(-1, idx)
        return vals, idx, ok

    def _product_key_filter(self, q, keys, scores):
        """Product-key style search (switchable, off in the quick run).
        Each key is split in two halves; only entries that rank among the best
        candidates for at least one half are scored exactly. With millions of
        entries this avoids the full brute-force search; here it only
        demonstrates the mechanism."""
        h = q.shape[-1] // 2
        c = min(self.cfg.product_key_candidates, keys.shape[1])
        s1 = q[..., :h] @ keys[..., :h].transpose(1, 2)
        s2 = q[..., h:] @ keys[..., h:].transpose(1, 2)
        cand = torch.zeros_like(scores, dtype=torch.bool)
        cand.scatter_(-1, s1.topk(c, -1).indices, True)
        cand.scatter_(-1, s2.topk(c, -1).indices, True)
        return scores.masked_fill(~cand, NEG)

    def knowledge_read(self, layer: int, h: torch.Tensor, state: MemoryState) -> Optional[torch.Tensor]:
        store = state.knowledge
        if len(store) == 0:
            return None
        q = self.q_knowledge[layer](h)
        vals, idx, ok = self._select(q, store)
        # Softmax over the 8 selected entries only, scaled by 1/sqrt(key_dim)
        # as in attention. Masked (empty) selections get weight 0.
        p = torch.softmax((vals / math.sqrt(self.key_dim)).masked_fill(~ok, NEG), -1) * ok
        weights = torch.zeros(vals.shape[:2] + (len(store),), device=h.device).scatter(-1, idx, p)
        read = weights.to(store.vals[0].dtype) @ store.vals[0]            # (B, C, d_model)
        return self.beta[layer].to(read.dtype) * read

    def rule_correction(self, layer: int, h2: torch.Tensor, state: MemoryState) -> Optional[torch.Tensor]:
        store = state.rule
        if len(store) == 0:
            return None
        q = self.q_rule[layer](h2)
        vals, idx, ok = self._select(q, store)
        # No softmax: several rules may apply at full strength at once, and a
        # non-matching rule should contribute ~0 instead of taking a share of
        # a fixed total. tanh keeps every score g in (-1, 1), so the correction
        # cannot explode.
        g = torch.tanh(self.cfg.rule_score_scale * vals) * ok              # (B, C, k)
        a, b = store.vals
        ax = (h2.to(a.dtype) @ a.transpose(1, 2)).float().gather(-1, idx)   # a_i . x for the selected i
        coef = torch.zeros(vals.shape[:2] + (len(store),), device=h2.device).scatter(-1, idx, g * ax)
        corr = coef.to(b.dtype) @ b                                        # sum_i g_i (a_i.x) b_i
        return self.alpha[layer].to(corr.dtype) * corr

    # ---- writing -------------------------------------------------------
    def write(self, state: MemoryState, x2: torch.Tensor, gate_logits: torch.Tensor) -> None:
        """Store the tokens of the chunk just processed whose gate is open.
        x2: hidden after the write layer (B, C, d); gate_logits: (B, C, 2)."""
        g = gate_logits.float()
        sig = torch.sigmoid(g)
        if state.trace is not None:
            state.trace.append(g.detach().cpu())
        if self.cfg.gate_mode == "hard_ste":
            # Forward: a hard yes/no. Backward: pretend the gate was the smooth
            # sigmoid (straight-through estimator), so the gate still learns.
            hard = (g > 0).float()
            w = hard + sig - sig.detach()
            valid = hard > 0
        else:
            # Soft fallback: every token is stored, scaled by its gate in (0,1).
            w, valid = sig, torch.ones_like(sig, dtype=torch.bool)
        w = w.to(x2.dtype)
        kval, ra, rb = self.w_kval(x2), self.w_ra(x2), self.w_rb(x2)
        if self.cfg.sparse_values:
            kval, ra, rb = (sparsify(t, self.cfg.sparse_value_k) for t in (kval, ra, rb))
        # The gate multiplies the value (and b): this is the path along which
        # the gate receives its gradient ("was this entry useful?").
        state.knowledge.append(self.w_kkey(x2), [kval * w[..., :1]], valid[..., 0])
        state.rule.append(self.w_rkey(x2), [ra, rb * w[..., 1:]], valid[..., 1])
        state.expected_writes = state.expected_writes + sig.sum()
        hard_ste = (g > 0).float() + sig - sig.detach()
        state.ste_writes = state.ste_writes.to(g.device) + hard_ste.sum((0, 1))
        state.decisions += g.shape[0] * g.shape[1]
        state.hard_writes = state.hard_writes.to(g.device) + valid.float().sum((0, 1)).detach()


class _BoundMemory:
    """Adapter that the core calls inside each layer (memory + current state)."""

    def __init__(self, module: MemoryModule, state: MemoryState):
        self.m, self.s = module, state

    def knowledge_read(self, layer, h):
        return self.m.knowledge_read(layer, h, self.s)

    def rule_correction(self, layer, h2):
        return self.m.rule_correction(layer, h2, self.s)


class BaselineLM(nn.Module):
    """The standard transformer: the core alone, with a wider SwiGLU so that
    its parameter count matches MemoryLM."""

    kind = "baseline"

    def __init__(self, cfg, ffn_hidden: Optional[int] = None):
        super().__init__()
        self.cfg = cfg
        self.core = CoreTransformer(core_config_from(cfg, ffn_hidden or baseline_ffn_width(cfg)))

    def forward(self, idx: torch.Tensor, **_) -> dict:
        return {"logits": self.core(idx)["logits"], "expected_writes": None}


class MemoryLM(nn.Module):
    kind = "memorylm"

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        m = cfg.model
        self.core = CoreTransformer(core_config_from(cfg, m.ffn_hidden_memorylm, extra_out_rows=2))
        # Gate bias slightly positive: at the start every token is written, so
        # the reads have something to learn from; the write penalty then
        # teaches the model to be selective.
        nn.init.constant_(self.core.extra_head.bias, cfg.memory.gate_bias_init)
        self.memory = MemoryModule(cfg.memory, m.d_model, m.ffn_hidden_memorylm, m.n_layers)
        self.chunk = cfg.memory.chunk_size
        self.write_layer = cfg.memory.write_layer
        self.memory_enabled = bool(cfg.memory.enabled)

    def new_state(self, trace: bool = False) -> MemoryState:
        # A fresh, EMPTY memory. Called at the start of every episode.
        return MemoryState(self.cfg.memory.capacity, trace)

    def process(self, idx: torch.Tensor, state: MemoryState, cache: Optional[KVCache] = None,
                pos0: int = 0, write: bool = True, use_memory: Optional[bool] = None) -> dict:
        """Run tokens through the model chunk by chunk, reading from and (if
        `write`) writing to `state`. Returns logits for all tokens."""
        use_memory = self.memory_enabled if use_memory is None else use_memory
        if cache is None:
            cache = KVCache(self.cfg.model.n_layers)
        bound = _BoundMemory(self.memory, state) if use_memory else None
        logits = []
        for s in range(0, idx.shape[1], self.chunk):
            out = self.core(idx[:, s:s + self.chunk], pos0=pos0 + s, cache=cache, mem=bound,
                            capture_layer=self.write_layer)
            logits.append(out["logits"])
            if use_memory and write:
                # The write happens AFTER the whole chunk was processed, so
                # nothing in this chunk can read what it wrote itself.
                self.memory.write(state, out["captured"], out["extra"])
        return {"logits": torch.cat(logits, 1)}

    def forward(self, idx: torch.Tensor, use_memory: Optional[bool] = None, **_) -> dict:
        """Next-token prediction on one batch of episodes. The memory starts
        empty here and is thrown away afterwards (one episode = one sequence)."""
        state = self.new_state()
        out = self.process(idx, state, use_memory=use_memory)
        n = max(state.decisions, 1)
        out["expected_writes"] = state.expected_writes / n if state.decisions else None
        out["write_floor"] = state.write_floor(float(self.cfg.memory.min_write_rate))
        out["state"] = state
        out["write_rate"] = (state.hard_writes / n).tolist() if state.decisions else [0.0, 0.0]
        return out


def baseline_ffn_width(cfg, multiple: int = 32) -> int:
    """SwiGLU width of the baseline such that both models have (almost) the
    same number of parameters. Rounded to a multiple of 32, which GPUs process
    efficiently. Explicit numbers in the config override the automatic choice."""
    fixed = cfg.model.ffn_hidden_baseline
    if fixed != "auto":
        return int(fixed)
    target = memorylm_param_count(cfg)
    m = cfg.model
    core_704 = count_params(CoreTransformer(core_config_from(cfg, m.ffn_hidden_memorylm)))
    per_unit = 3 * m.d_model * m.n_layers           # one extra SwiGLU unit adds 3 weights per layer
    exact = m.ffn_hidden_memorylm + (target - core_704) / per_unit
    width = int(round(exact / multiple) * multiple)
    if abs(width - exact) * per_unit > 0.02 * target:   # tiny models: 32 is too coarse a step
        width = int(round(exact / 8) * 8)
    return width


def memorylm_param_count(cfg) -> int:
    with torch.device("meta"):                      # count without allocating memory
        return count_params(MemoryLM(cfg))


def build_model(cfg, kind: str) -> nn.Module:
    if kind == "baseline":
        return BaselineLM(cfg)
    if kind == "memorylm":
        return MemoryLM(cfg)
    raise ValueError(f"unknown model kind {kind!r}")


def param_report(cfg) -> dict:
    with torch.device("meta"):
        b, m = BaselineLM(cfg), MemoryLM(cfg)
    nb, nm = count_params(b), count_params(m)
    return {"baseline_params": nb, "memorylm_params": nm,
            "memory_params": count_params(m.memory) + count_params(m.core.extra_head),
            "baseline_ffn": b.core.cfg.ffn_hidden, "memorylm_ffn": m.core.cfg.ffn_hidden,
            "relative_difference": abs(nb - nm) / nm}


if __name__ == "__main__":
    import argparse
    from .config import load_config
    ap = argparse.ArgumentParser(description="Print the parameter counts of both models.")
    ap.add_argument("--preset", default="quick")
    r = param_report(load_config(ap.parse_args().preset))
    for k, v in r.items():
        print(f"{k:22s} {v:,.4f}" if isinstance(v, float) else f"{k:22s} {v:,}")
