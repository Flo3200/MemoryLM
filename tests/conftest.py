"""Shared test fixtures: a small synthetic corpus in the TinyStories format."""
import os
import random

import pytest

WORDS = "the a cat dog girl boy ran saw big small happy sad tree ball sun park played went home friend".split()


def write_corpus(path, n, seed):
    rnd = random.Random(seed)
    stories = []
    for i in range(n):
        body = " ".join(rnd.choice(WORDS) for _ in range(rnd.randint(30, 90)))
        stories.append(f"Story number {i}. Once upon a time {body}.")
    # Like the real files: starts with a fragment, stories separated by the marker.
    path.write_text("ment of an older story.\n" + "".join(f"<|endoftext|>\n{s}\n" for s in stories))
    return stories


@pytest.fixture(scope="session")
def tiny_data(tmp_path_factory):
    """Tokenizer and token files for the tiny preset, prepared once per test session."""
    from memorylm import data
    from memorylm.config import load_config
    base = tmp_path_factory.mktemp("tiny")
    os.environ["MEMORYLM_ROOT"] = str(base / "root")
    write_corpus(base / "train.txt", 1500, 1)
    write_corpus(base / "valid.txt", 200, 2)
    txt = {"train": base / "train.txt", "valid": base / "valid.txt"}
    cfg = load_config("tiny", ["data.recite_eval_stories=40"])
    data.prepare_data(cfg, txt_override=txt)
    return base / "root"


@pytest.fixture()
def tiny_cfg(tiny_data, monkeypatch, request):
    """A tiny config with its own run folder (one per test)."""
    from memorylm.config import load_config
    monkeypatch.setenv("MEMORYLM_ROOT", str(tiny_data))

    def make(*overrides):
        return load_config("tiny", ["data.recite_eval_stories=40", f"run_name={request.node.name}", *overrides])
    return make
