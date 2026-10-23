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

---

## 3. Data and tokenizer (`memorylm/data.py`)

### TinyStories
- **What:** ~2.7 M short stories (2.2 GB) written by GPT-4 in the vocabulary
  of a 3–4-year-old child; median 179 tokens per story.
- **Why:** with a tiny vocabulary and simple grammar, a model of a few
  million parameters can already produce fluent, coherent text, so effects of
  the architecture are visible after a few GPU hours.
- **Alternatives:** Wikipedia/OpenWebText (far harder for small models),
  synthetic tasks only (clean, but say little about language).

### Byte-level BPE tokenizer with 4096 tokens
- **What:** starts from the 256 bytes and repeatedly merges the most frequent
  adjacent pair into a new token until 4096 tokens exist. Trained on 100 MB of
  training text. ~4 characters per token; frequent words (" little",
  " named") become one token.
- **Why:** byte-level means any text can be encoded (no "unknown" token).
  A small vocabulary suits a small model: the embedding table (vocab × 256)
  is already a quarter of all parameters; GPT-2's 50 257 tokens would need
  12.9 M parameters for the embedding alone. Too small a vocabulary
  (characters) makes sequences long and attention expensive.
- **Alternatives:** character level, WordPiece (BERT), Unigram/SentencePiece
  (T5, LLaMA 1), the GPT-2 tokenizer as is (not "from scratch").

### Fast exact encoding
- **What:** BPE never merges across the "words" found by the byte-level
  splitting rule, so each distinct word is encoded once and then looked up.
  Tests require *identical* ids to the reference tokenizer (also on tricky
  text: double spaces, apostrophes, umlauts, emoji).
- **Why:** Colab's free tier has 2 CPU cores; the reference library would need
  ~40 minutes for the 2.2 GB file, the lookup ~10.

### Tokenize once, cache as binary files
- **What:** token ids as `uint16` (2 bytes per token, ~1.1 GB in total) plus
  story start offsets, written to Google Drive; progress is saved after each
  block of 20 000 stories, so an interrupted session continues exactly.
  On Colab the files are copied to the local disk before training (random
  reads from Drive are slow).
- **Why:** tokenization is pure overhead; doing it once, resumably, saves GPU
  session time.
- **Alternatives:** tokenizing on the fly (costs CPU every step), the
  Hugging Face `datasets` library (convenient, heavier dependency).

### Four disjoint pools of stories
- **What:** `train` (phase 1), `recite_train` (every 50th train story, used
  only for phase-2 recitation), `recite_eval` (first 2000 validation stories,
  used only for recall evaluation), `valid` (remaining validation stories,
  for validation loss). A test decodes every pool and checks that no story
  appears in two of them.
- **Why:** if an evaluation passage had been seen in training, the model could
  "recite" it from its weights instead of its memory – the measurement of
  memory would be contaminated.
- **Alternatives:** random splits by hash (also fine), deduplication by n-gram
  overlap (stricter; TinyStories contains many near-duplicate stories, which
  is a known limit of this and any split).

### Fixed, resumable data order
- **What:** the training pool is cut into non-overlapping windows of 129
  tokens (128 inputs + 128 shifted targets) and visited in one random
  permutation fixed by the seed. The position in the permutation is one
  integer stored in every checkpoint.
- **Why:** both models must see the identical data in the identical order;
  resuming after a crash must not repeat or skip data.
- **Alternatives:** sampling random offsets each step (simpler, but windows
  repeat and resume is not exact), story-aligned packing.

### Recitation passages
- **What:** the first N tokens of a story with at least N tokens (95 % of
  stories have ≥ 128). Evaluation uses one fixed set per length.
- **Why:** a story beginning is a natural, self-contained passage; a fixed
  evaluation set makes all models and conditions directly comparable.

---

## 4. Training loop and robustness (`train.py`, `checkpoint.py`, `run_all.py`)

### AdamW, warm-up, cosine decay
- **What:** Adam keeps a running average of each weight's gradient and of its
  square and takes steps of roughly equal size for every weight; "W" means
  weight decay is applied directly (0.1, on matrices only). The learning rate
  rises linearly for 300 steps, then falls along a cosine curve to 10 %.
- **Why:** Adam copes with the very different gradient scales of embeddings,
  attention and memory scalars. Warm-up avoids huge steps while Adam's
  averages are still unreliable; the slow decay lets the model settle.
- **Alternatives:** SGD with momentum (needs much more tuning for
  transformers), Lion, Muon; constant or step-wise learning rate.

### Gradient clipping at 1.0
- **What:** if the gradient vector is longer than 1.0, it is shortened to 1.0.
- **Why:** a single unusual batch can produce a huge gradient that destroys
  what was learned; clipping caps the damage without changing normal steps.

### Mixed precision with loss scaling (fp16 on the T4)
- **What:** matrix products in 16-bit floats, sensitive operations (norms,
  softmax, loss) in 32-bit. The loss is multiplied by a large factor before
  backpropagation so that tiny gradients do not round to zero; steps whose
  gradients overflow are skipped and the factor is reduced.
- **Why:** roughly 2× faster and half the memory on a T4 GPU.
- **Alternatives:** bfloat16 (no loss scaling needed, but not fast on a T4),
  full 32-bit (slow).

### Same batch size, same data, same seed for both models
- **What:** the batch size is searched once for MemoryLM (the larger memory
  user, worst case = phase-2 step with 128-token recitations) and reused for
  the baseline; both read the same windows in the same order.
- **Why:** any difference in the results should come from the architecture,
  not from different data or optimisation settings.

### Token budget from a speed test
- **What:** phase 0 measures real tokens per second on the GPU and sets the
  phase-1 budget so MemoryLM needs ~60 min; the baseline gets the same number
  of tokens (it finishes sooner). If the whole plan would exceed 4 GPU hours,
  the budget is shrunk automatically.
- **Why:** a fair comparison fixes the *data* (tokens), not the time.
  Equal time would give the faster baseline more data.

### Phase 2: recitation curriculum
- **What:** read N tokens (writes allowed) → clear the context completely →
  `<recite>` → reproduce the passage; loss only on the recited tokens, plus
  the write penalty. N starts at 16 and grows by 0.1 % per step (interpreted
  as N_t = ⌈16 · 1.001^t⌉, reaching 128 after ~2100 steps; rounding up
  *every* step would add +1 per step and reach 128 after only 112 steps).
  Half of every batch is normal text, so language ability is not forgotten.
- **Why:** start easy so the model first discovers that the memory pays off.

### Write penalty starts in phase 2 (design decision)
- **What:** in phase 1 the penalty is 0; it ramps up over 2000 steps from
  the start of phase 2 (`memory.write_penalty_from_phase`, switchable).
- **Why:** in phase 1 attention already sees all 128 tokens, so the memory
  helps little there; a penalty would close the gates first. A closed gate
  receives no learning signal through the straight-through estimator, so the
  gates might never reopen when recitation starts.

### Checkpoints, resume, NaN rollback
- **What:** every 15 minutes the full training state (weights, optimizer,
  loss scaler, step, data position, all random-generator states, phase-2
  length and sampler) is written atomically (temporary file → rename); the
  last two plus the best (by validation loss) are kept. On restart the newest
  readable checkpoint is loaded. If the loss stays non-finite or a weight
  becomes NaN, training returns to the last good checkpoint with half the
  learning rate.
- **Why:** free Colab sessions end without warning. Tests prove that a run
  "killed" in the middle of phase 2 and resumed in a fresh process ends with
  bit-identical weights to an uninterrupted run, and that an injected NaN is
  rolled back.
- **Bugs found on CPU by these tests (no GPU time lost):** after a rollback
  the training loop kept counting in a stale copy of its state (it would have
  looped forever); the recitation length never grew.

### Status and metrics on Drive
- **What:** `logs/status.json` (current phase, step, losses, ETA, write
  rates, memory size) is overwritten after every evaluation interval;
  `logs/metrics_<model>.jsonl` keeps one line per evaluation.
- **Why:** progress can be checked from anywhere (Drive, the local app)
  without streaming the training log.

### Learning-rate multiplier for the memory scalars (added after a CPU probe)
- **What:** the 8 memory scalars (β and α per layer) use a 10× larger
  learning rate than all other weights (`memory.scalar_lr_mult`).
- **Why:** a CPU experiment showed a chicken-and-egg problem. While β and α
  are ~0, the memory's projections get almost no gradient; while the
  projections are untrained, the memory adds only noise, so β and α receive
  no consistent signal and stay near 0 (they reached ±0.08 after 600 steps).
  A larger step for just these numbers breaks the deadlock while keeping the
  exact zero start (MemoryLM still begins as a copy of its core).
- **Alternatives:** small non-zero initial scalars (gives up the clean start),
  a separate warm-up phase in which only the memory is trained.
- **Observed:** on random-word stories the memory barely helps even with all
  gates forced open, because recalling the next token requires knowing which
  token came before each stored entry; attention learns such
  "previous-token" patterns only on real, structured text in phase 1.

---

## 5. Colab notebook and Hugging Face Hub (`notebooks/`, `memorylm/hub.py`)
- **What:** 4 cells: mount Drive, clone + install, run/resume everything,
  optional upload. The same notebook runs on Kaggle (no Drive; outputs go to
  `/kaggle/working`). The Hugging Face token is read only from Colab's or
  Kaggle's secrets manager, never written into a cell.
- **Why:** after a disconnect, rerunning cells 1–3 is the entire recovery
  procedure – everything else is on Drive.
- **Alternatives:** paid always-on GPUs, local GPU, Weights & Biases for
  logging (needs an account; here plain JSON files suffice).

## 6. Evaluation (`memorylm/evaluate.py`)

### Four conditions instead of one number
- **What:** baseline with the passage still in context (upper bound),
  baseline with cleared context (lower bound), MemoryLM with cleared context
  and memory on, the same with memory off.
- **Why:** a raw accuracy is meaningless – a language model guesses many
  tokens correctly without any memory ("Once upon a" → "time"). The memory's
  contribution is **memory on − memory off** (same weights, same training).
  "Memory off" may still beat the baseline's lower bound: MemoryLM was trained
  on recitation in phase 2 and learned what story beginnings after `<recite>`
  look like. That is not memory and not a leak (the leak tests prove the
  passage cannot reach the cleared context any other way).

### Teacher-forced vs. greedy recall
- **Teacher-forced:** at each position the model sees the true previous
  tokens and predicts the next one; measures knowledge per position.
- **Greedy:** the model recites alone, its own mistakes carry forward; the
  strict, human-like notion of reciting.

### Compression
- **What:** correctly recalled tokens per KB written; also only the tokens
  above the lower bound per KB.
- **Why:** a memory that stores every token verbatim would recall perfectly
  but cost a lot; the interesting quantity is recall per unit of storage.

### Uncertainty
- **What:** 95 % intervals over passages (passages are the independent
  units). One training seed only: differences between models of a few
  points or a few hundredths of loss may be noise from training randomness,
  which these intervals do not capture.

### Fair language-modelling comparison
Validation loss is compared after phase 1, where both models have seen
exactly the same tokens; MemoryLM's phase-2 value is reported separately.

## 7. Local app (`app/app.py`)
A Gradio web app on the laptop CPU (the models are small enough):
chat comparison, memory test (read → clear → recite, with and without
memory), memory inspector (which tokens each gate writes, learned β and α),
training dashboard (from Drive or Hugging Face logs), model download.
Generation inside the app uses temperature and top-k sampling: temperature
0 always picks the most likely token (repetitive), higher values add
variety; top-k forbids all but the k most likely tokens (avoids nonsense).

---

## Appendix: CPU probe on real text (before any GPU time)
Tiny model (64-dim), real TinyStories text (80/20 split of the validation
file, 3 M phase-1 tokens), then 1500 recitation steps with 24-token passages.
Teacher-forced recall on 128 held-out passages:

| Variant | memory on | memory off | memory gain | tokens written |
|---|---|---|---|---|
| default gates + write penalty | 0.681 | 0.609 | **+0.072** | ~10 % |
| gates forced open, no penalty | 0.705 | 0.559 | **+0.147** | 100 % |

Lessons: (1) the memory only becomes useful after the model has learned
real language (on random-word stories the gain stayed ≈ 0); (2) the write
penalty buys a 10× smaller memory at the cost of about half the recall gain –
exactly the compression trade-off the main experiment measures; (3) the
learned read strengths end far from 0 (β up to 1.5, α down to −1.2), so the
layers actively use the memory. One seed, tiny model: indicative only.
