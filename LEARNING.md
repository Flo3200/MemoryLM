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
