"""Data: download TinyStories, train the BPE tokenizer, tokenize once, cache
as binary files, and serve batches in a fixed, resumable order.

Four strictly separate pools of stories:
  train         phase-1 next-token training (and the text half of phase 2)
  recite_train  phase-2 recitation training: every 50th TRAIN story, never
                used in phase 1
  recite_eval   recitation evaluation: the first 2000 VALID stories
  valid         validation loss: the remaining VALID stories
So no story used for evaluation was ever seen in training, and the stories
the model learns to recite are not the ones it learned language from.

Token ids are stored as uint16 (vocab < 65 536): half the size of int32.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch

from .config import run_root

SEP = "<|endoftext|>"          # story separator in the TinyStories text files
RECITE_EVERY = 50              # every 50th train story goes to the recitation-training pool
BLOCK = 20_000                 # stories per encoding block (also the resume granularity)


# ------------------------------------------------------------------ paths
def data_dir(cfg) -> Path:
    # One folder per vocabulary size: quick (4096) and full (8192) need
    # different tokenizers and therefore different token files.
    d = run_root(cfg) / f"data_v{cfg.tokenizer.vocab_size}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def download(cfg, which: str) -> Path:
    """Download a TinyStories text file to a LOCAL cache (not Drive: the raw
    text is 2.2 GB and can be re-downloaded in ~1 minute; only the much
    smaller token files are worth keeping on Drive)."""
    from huggingface_hub import hf_hub_download
    fname = cfg.data.train_file if which == "train" else cfg.data.valid_file
    cache = os.environ.get("MEMORYLM_HF_CACHE", str(Path(__file__).resolve().parent.parent / "runs" / "cache"))
    if Path("/content").exists():
        cache = "/content/hf_cache"
    return Path(hf_hub_download(cfg.data.hf_repo, fname, repo_type="dataset", cache_dir=cache))


def _pieces(path: Path) -> Iterator[str]:
    # Streaming read in 8 MB blocks: the train file (2.2 GB) is too large to
    # load at once on a small machine.
    buf = ""
    with open(path, encoding="utf-8", errors="replace") as f:
        while block := f.read(1 << 23):
            buf += block
            parts = buf.split(SEP)
            buf = parts.pop()
            yield from parts
    yield buf


def iter_stories(path: Path, skip: int = 0) -> Iterator[str]:
    """Yield cleaned, non-empty stories, skipping the first `skip` of them
    (resume). The first piece of a file is dropped: the TinyStories files
    start in the middle of a story."""
    n = 0
    for k, p in enumerate(_pieces(path)):
        p = p.strip()
        if k == 0 or not p:
            continue
        if n >= skip:
            yield p
        n += 1


# ------------------------------------------------------------------ tokenizer
def train_tokenizer(cfg, train_txt: Path, out: Path):
    """Byte-level BPE: start from the 256 possible bytes, repeatedly merge the
    most frequent adjacent pair into a new token until the vocabulary has
    4096 entries. Byte-level means ANY text can be encoded (no "unknown"
    token), while frequent words become single tokens."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=cfg.tokenizer.vocab_size,
                                  special_tokens=list(cfg.tokenizer.special_tokens),
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                  show_progress=False)
    limit = cfg.data.tokenizer_train_mb * 1_000_000

    def sample():
        used = 0
        for i, s in enumerate(iter_stories(train_txt)):
            if i % RECITE_EVERY == 0:
                continue                 # recitation stories stay unseen, even by the tokenizer
            used += len(s)
            yield s
            if used >= limit:
                break

    tok.train_from_iterator(sample(), trainer)
    tmp = out.with_suffix(".tmp")
    tok.save(str(tmp))
    os.replace(tmp, out)


def load_tokenizer(cfg):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(data_dir(cfg) / "tokenizer.json"))


def special_id(tok, name: str) -> int:
    i = tok.token_to_id(name)
    assert i is not None, f"special token {name} missing"
    return i


# ------------------------------------------------------------------ encoding
class FastEncoder:
    """Gives exactly the same token ids as the HF tokenizer, ~7x faster.

    BPE never merges across "words" (the pieces produced by the byte-level
    pre-tokenizer's regular expression below). TinyStories uses only a few
    tens of thousands of distinct words, so each distinct word is encoded
    once by the real tokenizer and then looked up. The HF library is slower
    because it builds an object with offsets for every token. On Colab's 2
    CPU cores this turns ~40 minutes of tokenization into ~10.
    """

    # The GPT-2 / byte-level pre-tokenization pattern used by `tokenizers`.
    PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def __init__(self, tok, eos: int):
        import regex
        self.rx = regex.compile(self.PATTERN)
        self.tok = tok
        self.word_id: dict[str, int] = {"\x00EOS": 0}      # word 0 = the <eos> token
        self.lens = np.array([1], dtype=np.int64)
        self.offs = np.array([0], dtype=np.int64)
        self.flat = np.array([eos], dtype=np.uint16)

    def encode(self, stories: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Returns all token ids of the block (each story followed by <eos>)
        and the token count of every story."""
        known = len(self.word_id)
        wid = self.word_id
        seqs = [[wid.setdefault(w, len(wid)) for w in self.rx.findall(s)] + [0] for s in stories]
        if len(wid) > known:
            new = list(wid)[known:]
            ids = [np.asarray(e.ids, dtype=np.uint16) for e in self.tok.encode_batch(new)]
            new_lens = np.array([len(i) for i in ids], dtype=np.int64)
            self.offs = np.concatenate([self.offs, len(self.flat) + np.cumsum(new_lens) - new_lens])
            self.lens = np.concatenate([self.lens, new_lens])
            self.flat = np.concatenate([self.flat, *ids])
        w = np.fromiter((x for s in seqs for x in s), dtype=np.int64)
        l = self.lens[w]
        # Vectorised gather: token positions of every word, laid end to end.
        idx = np.repeat(self.offs[w] - (np.cumsum(l) - l), l) + np.arange(l.sum())
        story_words = np.array([len(s) for s in seqs])
        story_lens = np.add.reduceat(l, np.cumsum(story_words) - story_words)
        return self.flat[idx], story_lens

    def matches_reference(self, stories: list[str], eos: int) -> bool:
        flat, lens = self.encode(stories)
        ref = [e.ids + [eos] for e in self.tok.encode_batch(stories)]
        return flat.tolist() == [t for r in ref for t in r]


class _Writer:
    """Appends token ids (uint16) and story start offsets (int64) to a pool."""

    def __init__(self, d: Path, name: str, n_tokens: int, n_stories: int):
        self.bin, self.idx = d / f"{name}.bin", d / f"{name}.idx"
        for path, count, size in ((self.bin, n_tokens, 2), (self.idx, n_stories, 8)):
            # Resume: cut the file back to the last recorded consistent state.
            with open(path, "ab") as f:
                f.truncate(count * size)
        self.n_tokens, self.n_stories = n_tokens, n_stories

    def add(self, flat: np.ndarray, lens: np.ndarray) -> None:
        if len(lens) == 0:
            return
        starts = np.cumsum(lens) - lens + self.n_tokens
        with open(self.bin, "ab") as f:
            f.write(flat.astype(np.uint16).tobytes())
        with open(self.idx, "ab") as f:
            f.write(starts.astype(np.int64).tobytes())
        self.n_tokens += int(lens.sum())
        self.n_stories += len(lens)


def _encode_pools(tok, txt: Path, d: Path, assign, state_key: str, meta: dict,
                  stop_tokens: Optional[int] = None) -> None:
    """Encode a text file story by story into several pools, resumably.
    `assign(i)` names the pool for story i. Progress is written to meta.json
    after every block, so an interrupted session continues where it stopped."""
    eos = special_id(tok, "<eos>")
    st = meta.setdefault(state_key, {"stories_done": 0, "pools": {}, "complete": False})
    if st["complete"]:
        return
    pools = {}
    i = st["stories_done"]
    batch: list[str] = []
    t0 = time.time()
    enc = FastEncoder(tok, eos)
    checked = False

    def flush():
        nonlocal batch, checked
        if not checked:
            # Safety net: the fast path must reproduce the real tokenizer exactly.
            if not enc.matches_reference(batch[:500], eos):
                raise RuntimeError("fast encoder disagrees with the HF tokenizer")
            checked = True
        names = [assign(i - len(batch) + k) for k in range(len(batch))]
        for name in dict.fromkeys(names):
            sel = [s for s, n in zip(batch, names) if n == name]
            flat, lens = enc.encode(sel)
            if name not in pools:
                p = st["pools"].get(name, {"tokens": 0, "stories": 0})
                pools[name] = _Writer(d, name, p["tokens"], p["stories"])
            pools[name].add(flat, lens)
        st["stories_done"] = i
        for name, w in pools.items():
            st["pools"][name] = {"tokens": w.n_tokens, "stories": w.n_stories}
        _save_meta(d, meta)
        batch = []

    # Re-open existing pools so that their files are truncated to the recorded state.
    for name, p in st["pools"].items():
        pools[name] = _Writer(d, name, p["tokens"], p["stories"])
    for story in iter_stories(txt, skip=st["stories_done"]):
        batch.append(story)
        i += 1
        if len(batch) == BLOCK:
            flush()
            if stop_tokens and st["pools"].get("train", {}).get("tokens", 0) >= stop_tokens:
                print(f"  {state_key}: reached {stop_tokens:,} train tokens ({time.time()-t0:.0f}s)")
                return
    if batch:
        flush()
    st["complete"] = True
    _save_meta(d, meta)
    print(f"  {state_key}: complete, {i:,} stories ({time.time()-t0:.0f}s)")


def _save_meta(d: Path, meta: dict) -> None:
    tmp = d / "meta.json.tmp"
    tmp.write_text(json.dumps(meta, indent=1))
    os.replace(tmp, d / "meta.json")            # atomic: never a half-written file


def load_meta(cfg) -> dict:
    p = data_dir(cfg) / "meta.json"
    return json.loads(p.read_text()) if p.exists() else {}


def prepare_data(cfg, min_train_tokens: Optional[int] = None, txt_override: Optional[dict] = None) -> dict:
    """Make sure tokenizer and token files exist (idempotent and resumable).
    min_train_tokens: tokenize the train file only until the phase-1 pool has
    at least this many tokens (None = the whole file)."""
    d = data_dir(cfg)
    meta = load_meta(cfg)
    txt = txt_override or {}
    get = lambda which: Path(txt[which]) if which in txt else download(cfg, which)

    if not (d / "tokenizer.json").exists():
        t0 = time.time()
        train_tokenizer(cfg, get("train"), d / "tokenizer.json")
        print(f"  tokenizer trained ({time.time()-t0:.0f}s)")
    tok = load_tokenizer(cfg)

    n_eval = cfg.data.recite_eval_stories
    if not meta.get("valid_file", {}).get("complete"):      # no download at all once everything is cached
        _encode_pools(tok, get("valid"), d, lambda i: "recite_eval" if i < n_eval else "valid", "valid_file", meta)
    have = meta.get("train_file", {}).get("pools", {}).get("train", {}).get("tokens", 0)
    if not meta.get("train_file", {}).get("complete") and (min_train_tokens is None or have < min_train_tokens):
        _encode_pools(tok, get("train"), d, lambda i: "recite_train" if i % RECITE_EVERY == 0 else "train",
                      "train_file", meta, stop_tokens=min_train_tokens)
    return load_meta(cfg)


def localize(cfg) -> Path:
    """On Colab, copy the token files from Drive to the local disk once:
    random access into a memory-mapped file on Drive is slow."""
    src = data_dir(cfg)
    if not str(src).startswith("/content/drive"):
        return src
    dst = Path("/content/local_data") / src.name
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        target = dst / f.name
        if not target.exists() or target.stat().st_size != f.stat().st_size or f.name == "meta.json":
            shutil.copy2(f, target)
    return dst


# ------------------------------------------------------------------ batches
class Pool:
    """A memory-mapped pool of tokens plus story start offsets."""

    def __init__(self, d: Path, name: str, n_tokens: Optional[int] = None):
        self.tokens = np.memmap(d / f"{name}.bin", dtype=np.uint16, mode="r")
        if n_tokens is not None:
            self.tokens = self.tokens[:n_tokens]
        self.starts = np.fromfile(d / f"{name}.idx", dtype=np.int64)
        self.starts = self.starts[self.starts < len(self.tokens)]
        ends = np.append(self.starts[1:], len(self.tokens))
        self.lengths = ends - self.starts - 1          # without the closing <eos>


class WindowStream:
    """Phase-1 batches: the pool is cut into non-overlapping windows of T+1
    tokens (T inputs, T shifted targets), visited in one fixed random order
    given by the seed. The position in that order is a single integer, so
    resuming after a crash is exact, and both models get the identical
    sequence of windows."""

    def __init__(self, pool: Pool, T: int, seed: int, n_windows: Optional[int] = None):
        self.pool, self.T = pool, T
        total = (len(pool.tokens) - 1) // T
        self.n = min(total, n_windows) if n_windows else total
        self.order = np.random.default_rng(seed).permutation(self.n)

    def get(self, position: int, batch_size: int) -> torch.Tensor:
        rows = []
        for j in range(position, position + batch_size):
            w = self.order[j % self.n] * self.T
            rows.append(np.asarray(self.pool.tokens[w:w + self.T + 1], dtype=np.int64))
        return torch.from_numpy(np.stack(rows))


def sample_passages(pool: Pool, rng: np.random.Generator, n: int, count: int) -> torch.Tensor:
    """`count` passages of exactly n tokens, each the beginning of a story
    that is at least n tokens long (a natural, self-contained text to recite)."""
    long_enough = np.nonzero(pool.lengths >= n)[0]
    if len(long_enough) == 0:
        raise ValueError(f"no story with at least {n} tokens in the pool")
    pick = rng.choice(long_enough, size=count, replace=len(long_enough) < count)
    return torch.from_numpy(np.stack([np.asarray(pool.tokens[pool.starts[s]:pool.starts[s] + n], dtype=np.int64)
                                      for s in pick]))


def eval_passages(pool: Pool, n: int, count: int, seed: int = 0) -> torch.Tensor:
    """A fixed evaluation set (same passages for every model and condition)."""
    return sample_passages(pool, np.random.default_rng(seed + n), n, count)


if __name__ == "__main__":
    import argparse
    from .config import load_config
    ap = argparse.ArgumentParser(description="Download, train tokenizer, tokenize and cache TinyStories.")
    ap.add_argument("--preset", default="quick")
    ap.add_argument("--min-train-tokens", type=int, default=None)
    a = ap.parse_args()
    print(json.dumps(prepare_data(load_config(a.preset), a.min_train_tokens), indent=1))
