"""Presets load, inherit correctly, and switching is one flag."""
from memorylm.config import load_config


def test_quick_preset_values():
    c = load_config("quick")
    assert c.model.d_model == 256 and c.model.n_layers == 4 and c.model.n_heads == 4
    assert c.model.d_model // c.model.n_heads == 64      # head dim 64
    assert c.model.context == 128 and c.tokenizer.vocab_size == 4096
    assert c.memory.chunk_size == 32 and c.memory.capacity == 512 and c.memory.top_k == 8


def test_full_preset_inherits_and_overrides():
    c = load_config("full")
    assert c.model.d_model == 384 and c.model.n_layers == 6 and c.model.n_heads == 6
    assert c.model.context == 256 and c.tokenizer.vocab_size == 8192
    assert c.memory.capacity == 512                         # inherited from quick


def test_command_line_override():
    c = load_config("quick", ["model.d_model=128", "memory.gate_mode=soft"])
    assert c.model.d_model == 128 and c.memory.gate_mode == "soft"
