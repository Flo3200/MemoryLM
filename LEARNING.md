# LEARNING.md – why the models are built this way

One section per component: **what it does**, **why it exists**, **what
alternatives exist**. Sections are added milestone by milestone.

---

## 0. Project setup

### Config presets (`configs/*.yaml`)
- **What:** every number (model size, learning rate, memory capacity, time
  budget) lives in one YAML file. `full.yaml` inherits from `quick.yaml` and
  only lists what changes; `tiny.yaml` is a toy size for CPU tests.
- **Why:** a fair comparison requires that both models see *exactly* the
  same settings. One file makes that checkable, and `--preset full` switches
  the whole experiment with one flag.
- **Alternatives:** command-line flags only (hard to audit), Python
  dataclasses (type-safe but less readable), Hydra (powerful, but heavy for a
  project this size).

### Three sizes: tiny → quick → full
- **What:** `tiny` (~0.1 M parameters) runs on a laptop CPU in seconds;
  `quick` (~4–5 M) is the 4-hour GPU run; `full` (~15 M) is for later.
- **Why:** GPU time is the scarcest resource. Almost all bugs (wrong masks,
  shape errors, broken resume) show up at any size, so they are found on the
  tiny model for free.
- **Alternatives:** debugging directly on the GPU – faster to start, but every
  crash costs session time and quota.

### Secret scanner (`scripts/check_secrets.py`)
- **What:** searches every commit for patterns of access tokens
  (Hugging Face `hf_…`, GitHub `ghp_…`, private keys, hard-coded passwords)
  and blocks the commit. It runs locally as a git hook and again on GitHub.
- **Why:** the repository is public; a leaked token can be found by bots
  within minutes. The Hugging Face token therefore lives only in Colab's
  secrets manager.
- **Alternatives:** dedicated tools such as `gitleaks` or `detect-secrets`
  (more patterns, but extra installation); GitHub's own push protection
  (works only for some token types).

### Remote Colab control (colab-mcp)
- **What:** Google's official MCP server lets the coding agent run cells in
  an open Colab browser tab.
- **Why:** the free Colab tier disconnects without warning; the agent can
  reconnect and run the resume cell itself instead of asking a human.
- **Alternatives:** clicking manually; paid Colab Pro background execution;
  Kaggle notebooks (also free, used as fallback).

---

## 1. The transformer core (`memorylm/model_core.py`)

Both models share this code. A layer is: `x = x + Attention(Norm(x))`,
then `x = x + SwiGLU(Norm(x))`. The quick preset stacks 4 such layers of
width 256.

### Decoder-only language model
- **What:** reads tokens left to right and predicts, at every position, the
  next token. Training signal: cross-entropy between prediction and the
  actual next token, at all positions at once.
- **Why:** the simplest set-up that learns language from raw text without
  labels; the same objective as GPT models.
- **Alternatives:** encoder-decoder (T5; needs input/output pairs),
  masked language models (BERT; cannot generate text left to right).

### Token embedding
- **What:** a table with one learned 256-dim vector per token id.
- **Why:** the network computes with vectors, not ids; similar tokens end up
  with similar vectors.
- **Alternatives:** one-hot input times a matrix (identical maths, slower);
  character-level input (no tokenizer, but 4–5× longer sequences).

### Causal self-attention
- **What:** every token forms a query, key and value; it reads a weighted
  average of the values of all *earlier* tokens, weighted by
  softmax(query·key / √64).
- **Why:** it lets any token fetch information from any earlier token in one
  step, regardless of distance, and all positions are computed in parallel.
  The **causal mask** forbids looking at later tokens – otherwise the model
  would simply copy the answer during training and fail at generation time.
  The division by √64 keeps scores in a range where softmax still has useful
  gradients.
- **Alternatives:** recurrent networks (LSTM; sequential, forget over long
  distances), convolutions (fixed reach), linear attention / state-space
  models (Mamba; cheaper for long texts, weaker at exact copying).
- **Test:** changing future tokens leaves earlier predictions unchanged, and
  the gradient from position t to any later input is exactly zero.

### Multiple heads (4 × 64)
- **What:** attention is run 4 times in parallel on 64-dim slices.
- **Why:** one softmax can focus on only one "kind" of relation at a time;
  several heads can follow several relations at once for the same cost.
- **Alternatives:** one large head; multi-query / grouped-query attention
  (share keys and values between heads to save memory at inference).

### RoPE (rotary position embedding)
- **What:** rotates query and key vectors by angles proportional to the
  position.
- **Why:** attention alone is blind to word order. With RoPE the score
  between two tokens depends on their *distance*, has no parameters, and
  generalises well.
- **Alternatives:** learned absolute positions (GPT-2), sinusoidal additions
  (original 2017 transformer), ALiBi (linear distance penalty), none (NoPE).

### RMSNorm, pre-norm placement
- **What:** rescales each vector to unit root-mean-square, times a learned
  gain; applied *before* each sub-block.
- **Why:** keeps activations at a stable scale. Pre-norm leaves a clean
  residual path, which makes deep stacks train reliably without delicate
  warm-up. RMSNorm drops LayerNorm's mean-centring – cheaper, equally good.
  It is computed in float32 because squaring fp16 numbers can overflow.
- **Alternatives:** LayerNorm, post-norm (original transformer), no norm with
  careful initialisation.

### Residual connections
- **What:** each sub-block *adds* its result to its input (`x = x + f(x)`).
- **Why:** information and gradients can flow straight through the stack;
  each layer only has to learn a correction. Output projections start with a
  smaller scale (0.02/√(2·layers)) so the sum of all corrections does not
  grow with depth at initialisation.
- **Alternatives:** none in practice; plain stacking of deep layers trains
  badly.

### SwiGLU feed-forward
- **What:** `W_down( Swish(W_gate x) ⊙ W_up x )`, widening 256 → 704 and back.
- **Why:** attention moves information *between* tokens; the feed-forward
  block processes each token on its own and stores much of the learned
  "knowledge". The gate lets every hidden unit switch itself on or off per
  token; this reaches lower loss than the classic ReLU MLP at equal size.
  The width 704 ≈ 8/3 × 256 keeps the parameter count of a 4× ReLU MLP.
- **Alternatives:** ReLU or GELU MLP with width 4×d (GPT-2), mixture of experts.
- **MemoryLM hook:** its rule store adds a correction to the `W_up x` branch.

### Weight tying
- **What:** the output layer reuses the embedding table (transposed).
- **Why:** saves 1.05 M of ~4.3 M parameters and makes "token as input" and
  "token as prediction" share one meaning; for small models it usually helps.
- **Alternatives:** separate output matrix (common in larger models).

### No dropout
- **What:** no random switching-off of units during training.
- **Why:** dropout fights overfitting, but here each token is seen at most
  about once (the data set is far larger than the training budget), so there
  is nothing to overfit; dropout would only slow learning.
- **Alternatives:** dropout 0.1 (GPT-2), weight decay (used instead, 0.1).

### Key/value cache for chunked processing
- **What:** keys and values of earlier chunks are kept, so a new 32-token
  chunk attends to them as if the whole text were processed at once.
- **Why:** MemoryLM must work chunk by chunk (memory written in one chunk
  becomes readable only in the next). A test proves that chunked processing
  gives the same output as one full pass, so the baseline and MemoryLM see
  exactly the same attention context.
- **Alternatives:** process chunks without the cache (would shrink the
  attention context and make the comparison unfair).

### Parameter count (quick preset)
Core with SwiGLU width 704: **4,262,144** parameters. With width 1024:
**5,245,184**. The baseline width is chosen automatically in M2 so that
baseline and MemoryLM (core + memory projections) differ by less than 2 %.

---

## 2. The external memory (`memorylm/memory.py`)

### Overview
MemoryLM = the core above + two stores that are filled **while reading**
and emptied after every episode. Storing is not learning in the usual sense:
no weights change at test time; the model writes vectors and later reads them.

### Knowledge store (key → value), read with top-8 softmax
- **What:** each layer projects its normalised input to a 64-dim query,
  compares it with every stored key (brute-force dot product), keeps the
  8 best, softmax-weights their 256-dim values and adds the result to the
  attention output, scaled by a learned per-layer number β (start: 0).
- **Why:** attention can only see the 128 tokens in the context window. The
  store keeps compressed traces of earlier text after the context is cleared.
  Top-k keeps the read sharp and cheap; softmax lets the model blend a few
  similar entries.
- **Alternatives:** attend over *all* entries (Memorizing Transformers use
  kNN + attention like here), recurrent memory tokens (RMT), compressive
  memories (Compressive Transformer, Infini-attention), retrieval from an
  external text database (RETRO).

### Rule store (key → a, b), a low-rank correction of the feed-forward layer
- **What:** an entry adds the rank-1 matrix `b aᵀ` to the expansion matrix
  `W_up`, weighted by `g = tanh(score/8)`. Eight entries give a rank-≤8
  correction: `content = W_up x + α Σ g_i b_i (a_i·x)`. The 704×256 matrix is
  never built; `a_i·x` is one number per entry.
- **Why:** the knowledge store adds *facts* (vectors) to the stream; the rule
  store changes *how* tokens are processed – closer to "learning a rule from
  the text" (fast weights). No softmax: rules should not compete for a fixed
  budget, several may apply fully or none. tanh bounds each score so the
  correction cannot explode. A test checks that the cheap formula equals the
  explicit matrix product.
- **Alternatives:** fast-weight programmers / linear transformers (write
  outer products into one matrix), LoRA (a *trained*, fixed low-rank update),
  hypernetworks (generate full weights).

### One shared store for all layers; scalars initialised to 0
- **What:** all 4 layers read the same two stores; β and α start at 0.
- **Why:** sharing keeps memory size independent of depth. Starting at 0
  means MemoryLM begins as an exact copy of its core (proven by a test,
  bit-for-bit), so any later difference to the baseline is learned, not an
  accident of initialisation.
- **Alternatives:** one store per layer (more memory, more flexible); small
  random initial scalars (faster start, but no clean starting point).

### Write path: gates on the output layer, straight-through estimator
- **What:** the output matrix has 2 extra rows → two gate numbers per token.
  Gate > 0 → the token is stored. The stored vectors are projections of the
  hidden state after layer 2. During backpropagation the hard 0/1 step is
  treated as if it were a smooth sigmoid (straight-through estimator, STE).
  The gate bias starts at +0.5, so at first nearly every token is written.
- **Why:** a hard decision gives a truly sparse memory (unwritten = empty),
  but a step function has zero gradient everywhere; the STE gives the gate a
  useful learning signal anyway. Layer 2 of 4 is a compromise: deep enough to
  encode context, early enough to be cheap. The stored value is multiplied by
  the gate, which is the path through which the gate learns whether an entry
  was useful. **Known limit:** a closed gate's entry is invisible, so it gets no
  signal that writing *would have* helped; the positive start bias and the
  soft-gate fallback (`gate_mode: soft`) address this.
- **Alternatives:** soft gates only (no sparsity), REINFORCE / Gumbel-softmax
  (unbiased or annealed estimators, noisier), writing everything with a
  learned eviction policy.

### Write penalty
- **What:** loss += λ · (expected writes per token), with expected writes =
  sum of the gates' sigmoids; λ grows linearly from 0 during the first steps.
- **Why:** without a cost the cheapest strategy is to store everything. The
  penalty makes memory a budget; ramping it in lets the model first discover
  that memory is useful before it learns to economise.
- **Alternatives:** a hard write budget per chunk (top-n tokens), L0
  regularisation, a fixed small capacity only.

### Chunks of 32 tokens and causality
- **What:** text is processed chunk by chunk; writes happen after a chunk is
  complete, so entries from chunk j are readable only from chunk j+1.
- **Why:** within one chunk all tokens are computed in parallel; if a token
  could read an entry written in the same chunk, it could read information
  from tokens after it (a leak). Tests change one token in the middle of a
  chunk and check that no earlier position – including earlier ones in the
  same chunk – changes, and that gradients never flow from the future.
- **Alternatives:** token-by-token writing (exact but 32× more sequential
  steps), longer chunks (faster, but fresh text becomes readable later).

### Episodes, capacity and backprop through time
- **What:** one episode = one training sequence. The memory starts empty
  (tested), holds at most 512 entries per store (oldest unwritten slots are
  dropped first), and is discarded at the end. Gradients flow back through
  every write within the episode.
- **Why:** an empty start makes the memory a property of the *current* text,
  not a hidden second training set; backprop through writes is what teaches
  the write projections what is worth storing.
- **Alternatives:** persistent memory across episodes (lifelong memory; much
  harder to evaluate and to keep causal), truncated backprop (cheaper, but
  the model cannot learn long-range write → read dependencies).

### Memory size
Counted as stored non-zero numbers × 2 bytes (fp16), per episode. One
entry = 64 (key) + 256 (value) in the knowledge store and
64 + 256 + 704 in the rule store; writing all 128 tokens of a sequence to
both stores costs about 340 KB.

### Parameter balance
Memory parameters (queries, write projections, gates, scalars): 475,658.
MemoryLM: **4,737,802**. Baseline with automatically chosen SwiGLU width 864:
**4,753,664** (0.33 % apart). The specification expected about 1024; that
width would give the baseline 0.5 M (10 %) more parameters than MemoryLM,
because the memory projections are smaller than originally estimated.

### Switchable, off in the quick run
- **Sparse values:** keep only the 32 largest numbers per stored vector
  (smaller memory, tested to shrink the reported size).
- **Product-key search:** split each key in two halves, keep only entries
  that rank well on either half, score those exactly (sub-linear search for
  very large memories, after Lample et al. 2019).
