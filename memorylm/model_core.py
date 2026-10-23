"""The shared transformer core used by BOTH models.

A decoder-only transformer turns a sequence of token ids into, for every
position, a probability distribution over the next token. It is a stack of
identical layers; each layer lets every token (1) look at earlier tokens
(attention) and (2) process what it found on its own (feed-forward).

The baseline is exactly this core. MemoryLM is this core plus two "hooks"
per layer (see memorylm/memory.py): a knowledge read that is added to the
attention result, and a rule correction that is added to the feed-forward
expansion. With no memory object the hooks do nothing, so the code path of
the baseline and of MemoryLM-without-memory is literally the same.

Note: only features available in torch 2.2 are used (the last version for
Intel Macs), so the same file runs on the laptop and on Colab.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CoreConfig:
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    ffn_hidden: int
    context: int
    rope_base: float = 10000.0
    norm_eps: float = 1e-5
    tie_weights: bool = True
    extra_out_rows: int = 0      # MemoryLM: 2 extra output rows (write gates)

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


class RMSNorm(nn.Module):
    """Rescales each token vector to a fixed average size (root mean square = 1),
    then multiplies by a learned per-dimension gain.

    Why: without normalisation, vector sizes drift as they pass through many
    layers, and training becomes unstable. RMSNorm is LayerNorm without the
    mean subtraction and bias: cheaper and works just as well in practice.
    """

    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Computed in float32 even under fp16 mixed precision: squaring
        # fp16 values can overflow (max ~65 000).
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * self.weight.float()).type_as(x)


def rope_tables(head_dim: int, max_pos: int, base: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute the rotation angles of RoPE for every position.

    Each pair of dimensions in a query/key vector is rotated by an angle that
    grows with the position; different pairs rotate at different speeds
    (like the hands of a clock). Low pairs rotate fast (fine position
    detail), high pairs slowly (coarse position).
    """
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    angles = torch.outer(torch.arange(max_pos).float(), inv_freq)   # (max_pos, head_dim/2)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate query or key vectors x of shape (B, H, T, head_dim).

    Why RoPE: after rotating both q and k, their dot product depends only on
    the DISTANCE between two tokens, not on their absolute positions. That is
    exactly what attention needs ("the word 3 tokens back"), and it costs no
    parameters. Alternatives: learned absolute position embeddings (GPT-2),
    sinusoidal embeddings (original transformer), ALiBi (a distance penalty).
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class KVCache:
    """Keys and values of earlier chunks, so a new chunk can attend to them.

    MemoryLM must process text chunk by chunk (memory written in chunk j may
    only be read from chunk j+1 on). The cache makes chunked processing give
    exactly the same attention result as processing the whole sequence at once.
    """

    def __init__(self, n_layers: int):
        self.k: list[Optional[torch.Tensor]] = [None] * n_layers
        self.v: list[Optional[torch.Tensor]] = [None] * n_layers

    def length(self) -> int:
        return 0 if self.k[0] is None else self.k[0].shape[2]


class Attention(nn.Module):
    """Causal multi-head self-attention.

    Every token builds a query ("what am I looking for?"), a key ("what do I
    contain?") and a value ("what do I pass on?"). The query is compared with
    the keys of all EARLIER tokens; softmax turns the scores into weights;
    the output is the weighted sum of their values. Several heads do this in
    parallel on 64-dim slices, so different heads can track different
    relations (e.g. one head follows the subject, another the last comma).
    """

    def __init__(self, cfg: CoreConfig):
        super().__init__()
        self.n_heads, self.head_dim = cfg.n_heads, cfg.head_dim
        # One matrix for Q, K and V together: one big matrix multiply is
        # faster on a GPU than three small ones; mathematically identical.
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, h: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                cache: Optional[KVCache], layer: int) -> torch.Tensor:
        """Returns the concatenated head outputs BEFORE the output projection,
        so MemoryLM can add its knowledge read at exactly that point."""
        B, T, D = h.shape
        q, k, v = self.qkv(h).split(D, dim=-1)
        q, k, v = (t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2) for t in (q, k, v))
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        past = 0
        if cache is not None:
            if cache.k[layer] is not None:
                past = cache.k[layer].shape[2]
                k = torch.cat([cache.k[layer], k], dim=2)
                v = torch.cat([cache.v[layer], v], dim=2)
            cache.k[layer], cache.v[layer] = k, v
        if past == 0:
            # The causal mask: position i may only look at positions <= i.
            # Without it the model could "cheat" by reading the very token it
            # has to predict.
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            # New chunk: query i sits at absolute position past+i and may see
            # every cached key plus new keys up to itself.
            qi = torch.arange(T, device=h.device).unsqueeze(1) + past
            kj = torch.arange(past + T, device=h.device).unsqueeze(0)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=kj <= qi)
        return y.transpose(1, 2).reshape(B, T, D)


class SwiGLU(nn.Module):
    """The feed-forward block: expand, gate, compress.

    content = W_up x, gate = Swish(W_gate x), out = W_down (gate * content).
    The gate lets each hidden unit decide, per token, how much of its content
    to let through. Compared with the classic ReLU MLP (2 matrices) it uses 3
    matrices, so for equal parameters the hidden width is ~2/3 as large;
    empirically it reaches lower loss (Shazeer 2020, used in LLaMA).
    """

    def __init__(self, d: int, hidden: int):
        super().__init__()
        self.w_gate = nn.Linear(d, hidden, bias=False)
        self.w_up = nn.Linear(d, hidden, bias=False)      # the "content" branch
        self.w_down = nn.Linear(hidden, d, bias=False)

    def forward(self, h: torch.Tensor, up_correction: Optional[torch.Tensor] = None) -> torch.Tensor:
        content = self.w_up(h)
        if up_correction is not None:
            # MemoryLM's rule store adds a low-rank correction to W_up x here.
            content = content + up_correction
        return self.w_down(F.silu(self.w_gate(h)) * content)


class Block(nn.Module):
    """One transformer layer in pre-norm order:
        x = x + Out(Attention(Norm(x)) [+ knowledge read])
        x = x + SwiGLU(Norm(x))        [+ rule correction inside]
    Pre-norm (normalise BEFORE each sub-block, add the raw result back) keeps
    an untouched "residual highway" from input to output, so gradients reach
    early layers easily and no learning-rate warm-up tricks are needed for
    stability. The original transformer used post-norm, which is harder to
    train when deep.
    """

    def __init__(self, cfg: CoreConfig, layer: int):
        super().__init__()
        self.layer = layer
        self.norm1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = SwiGLU(cfg.d_model, cfg.ffn_hidden)

    def forward(self, x, cos, sin, cache=None, mem=None):
        h = self.norm1(x)
        a = self.attn(h, cos, sin, cache, self.layer)
        if mem is not None:
            read = mem.knowledge_read(self.layer, h)          # step 4 of a MemoryLM layer
            if read is not None:                              # None = store still empty
                a = a + read
        x = x + self.attn.out(a)
        h2 = self.norm2(x)
        corr = mem.rule_correction(self.layer, h2) if mem is not None else None
        return x + self.ffn(h2, corr)


class CoreTransformer(nn.Module):
    def __init__(self, cfg: CoreConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg, i) for i in range(cfg.n_layers))
        self.norm_f = RMSNorm(cfg.d_model, cfg.norm_eps)
        if not cfg.tie_weights:
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.extra_out_rows:
            # MemoryLM's 2 extra output rows (write gates for knowledge and rule
            # store). They are NOT tied to the embedding: no token corresponds to them.
            self.extra_head = nn.Linear(cfg.d_model, cfg.extra_out_rows, bias=True)
        cos, sin = rope_tables(cfg.head_dim, cfg.context, cfg.rope_base)
        # Buffers move with .to(device) but are not trained and not saved.
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init)
        # Residual-branch outputs start smaller (GPT-2 trick): each layer adds to
        # the residual stream, so with 2 additions per layer the total variance
        # would grow with depth unless each addition is scaled by 1/sqrt(2L).
        for b in self.blocks:
            for w in (b.attn.out.weight, b.ffn.w_down.weight):
                nn.init.normal_(w, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def output_weight(self) -> torch.Tensor:
        # Weight tying: the matrix that turns a token id into a vector (input)
        # is reused, transposed, to score every token as the next one (output).
        # Saves vocab x d_model parameters (1 M of our ~4.5 M) and ties the
        # meaning of "token as input" and "token as prediction" together.
        return self.tok_emb.weight if self.cfg.tie_weights else self.lm_head.weight

    def forward(self, idx: torch.Tensor, pos0: int = 0, cache: Optional[KVCache] = None,
                mem=None, capture_layer: Optional[int] = None) -> dict:
        """idx: (B, T) token ids starting at absolute position pos0.
        Returns logits (B, T, vocab), optional extra (gate) logits, optional
        hidden state after layer `capture_layer` (1-based), and the final hidden."""
        B, T = idx.shape
        assert pos0 + T <= self.cfg.context, "sequence longer than the context window"
        cos, sin = self.rope_cos[pos0:pos0 + T], self.rope_sin[pos0:pos0 + T]
        x = self.tok_emb(idx)
        captured = None
        for i, block in enumerate(self.blocks):
            x = block(x, cos, sin, cache, mem)
            if capture_layer is not None and i + 1 == capture_layer:
                captured = x
        h = self.norm_f(x)
        out = {"logits": F.linear(h, self.output_weight()), "hidden": h, "captured": captured}
        if self.cfg.extra_out_rows:
            out["extra"] = self.extra_head(h)
        return out


def count_params(model: nn.Module) -> int:
    """Trainable parameters; a tied matrix is counted once (parameters()
    already de-duplicates shared tensors)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def core_config_from(cfg, ffn_hidden: int, extra_out_rows: int = 0) -> CoreConfig:
    m = cfg.model
    return CoreConfig(vocab_size=cfg.tokenizer.vocab_size, d_model=m.d_model, n_layers=m.n_layers,
                      n_heads=m.n_heads, ffn_hidden=ffn_hidden, context=m.context,
                      rope_base=float(m.rope_base), norm_eps=float(m.norm_eps),
                      tie_weights=bool(m.tie_weights), extra_out_rows=extra_out_rows)
