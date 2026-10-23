"""Tests of tokenizer, caching and batching (milestone M3). Offline: they use a
small synthetic corpus in the TinyStories file format."""
import numpy as np
import pytest

from memorylm import data
from memorylm.config import load_config

from conftest import write_corpus

@pytest.fixture()
def corpus(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORYLM_ROOT", str(tmp_path / "root"))
    train = write_corpus(tmp_path / "train.txt", 600, 1)
    valid = write_corpus(tmp_path / "valid.txt", 120, 2)
    cfg = load_config("tiny", ["data.recite_eval_stories=20"])
    txt = {"train": tmp_path / "train.txt", "valid": tmp_path / "valid.txt"}
    return cfg, txt, train, valid


def decode_pool(tok, pool):
    out = []
    for s, n in zip(pool.starts, pool.lengths):
        out.append(tok.decode(pool.tokens[s:s + n].tolist()))
    return out


def test_story_reader_drops_fragment(corpus, tmp_path):
    _, txt, train, _ = corpus
    assert list(data.iter_stories(txt["train"])) == train
    assert list(data.iter_stories(txt["train"], skip=5)) == train[5:]


def test_tokenizer_round_trip(corpus):
    cfg, txt, train, _ = corpus
    data.prepare_data(cfg, txt_override=txt)
    tok = data.load_tokenizer(cfg)
    assert tok.get_vocab_size() <= cfg.tokenizer.vocab_size
    for s in train[:20]:
        assert tok.decode(tok.encode(s).ids) == s
    for name in cfg.tokenizer.special_tokens:
        assert tok.token_to_id(name) is not None


def test_pools_are_disjoint_and_complete(corpus):
    cfg, txt, train, valid = corpus
    data.prepare_data(cfg, txt_override=txt)
    tok, d = data.load_tokenizer(cfg), data.data_dir(cfg)
    got = {name: decode_pool(tok, data.Pool(d, name)) for name in ("train", "recite_train", "valid", "recite_eval")}
    assert got["recite_train"] == train[::data.RECITE_EVERY]
    assert got["train"] == [s for i, s in enumerate(train) if i % data.RECITE_EVERY]
    assert got["recite_eval"] == valid[:20] and got["valid"] == valid[20:]
    sets = [set(v) for v in got.values()]
    for i in range(4):
        for j in range(i + 1, 4):
            assert not sets[i] & sets[j]


def test_interrupted_tokenization_resumes_identically(corpus, tmp_path, monkeypatch):
    cfg, txt, _, _ = corpus
    monkeypatch.setattr(data, "BLOCK", 50)
    data.prepare_data(cfg, min_train_tokens=2000, txt_override=txt)        # stops early ("session ended")
    partial = data.load_meta(cfg)["train_file"]
    assert not partial["complete"]
    data.prepare_data(cfg, txt_override=txt)                               # resumes
    resumed = (data.data_dir(cfg) / "train.bin").read_bytes()

    monkeypatch.setenv("MEMORYLM_ROOT", str(tmp_path / "fresh"))
    fresh_cfg = load_config("tiny", ["data.recite_eval_stories=20"])
    data.prepare_data(fresh_cfg, txt_override=txt)                         # one uninterrupted run
    assert resumed == (data.data_dir(fresh_cfg) / "train.bin").read_bytes()


def test_window_stream_is_deterministic_and_resumable(corpus):
    cfg, txt, _, _ = corpus
    data.prepare_data(cfg, txt_override=txt)
    pool = data.Pool(data.data_dir(cfg), "train")
    T = cfg.model.context
    a, b = data.WindowStream(pool, T, seed=7), data.WindowStream(pool, T, seed=7)
    assert (a.get(0, 4) == b.get(0, 4)).all()
    full = a.get(0, 8)
    assert (a.get(4, 4) == full[4:]).all()                 # position 4 = exactly where we stopped
    assert full.shape == (8, T + 1)
    assert sorted(a.order.tolist()) == list(range(a.n))    # every window exactly once per pass


def test_passages_start_at_story_beginnings(corpus):
    cfg, txt, _, valid = corpus
    data.prepare_data(cfg, txt_override=txt)
    tok = data.load_tokenizer(cfg)
    pool = data.Pool(data.data_dir(cfg), "recite_eval")
    p = data.eval_passages(pool, 16, 5)
    assert p.shape == (5, 16)
    starts = {tuple(tok.encode(s).ids[:16]) for s in valid[:20]}
    assert all(tuple(row.tolist()) in starts for row in p)
    assert (data.eval_passages(pool, 16, 5) == p).all()    # fixed evaluation set
    np.testing.assert_array_less(-1, pool.lengths)


def test_fast_encoder_matches_tokenizer_on_edge_cases(corpus):
    cfg, txt, train, _ = corpus
    data.prepare_data(cfg, txt_override=txt)
    tok = data.load_tokenizer(cfg)
    eos = data.special_id(tok, "<eos>")
    tricky = ["Hello  world!\n\nNew   line.", "It's Tom's ball, isn't it? They'll go.", "Numbers 123 and 4,5.6",
              "Ümlaut café naïve — “quotes” ’ and emoji 🙂", "  leading spaces", "trailing spaces   ", "\ttab\tsep",
              "ALL CAPS!!! ...", "a" * 300] + train[:30]
    assert data.FastEncoder(tok, eos).matches_reference(tricky, eos)
