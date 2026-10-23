# MemoryLM

A small research project: two tiny language models are trained **from scratch**
on children's stories under identical conditions and then compared.

1. **Baseline** – a standard decoder-only transformer (~4–5 M parameters).
2. **MemoryLM** – the same transformer plus an external vector memory:
   a *knowledge store* (key → value, read by attention-like lookup) and a
   *rule store* (key → low-rank correction of the feed-forward layer).
   The model itself decides, token by token, what to write into memory.

The question: can a model that writes into an explicit memory reproduce a
passage after its context window has been completely cleared – and how much
memory does it need per correctly recalled token?

Everything runs on free resources: Google Colab (free T4 GPU, Kaggle as a
fallback), Google Drive, this public GitHub repository and the free
Hugging Face Hub. The GPU part fits into about 4 hours.

> **Status:** under construction (milestone M0: setup). See
> [LEARNING.md](LEARNING.md) for why every component is built the way it is.

## Schnellstart (Deutsch)

**Training auf Colab**
1. `notebooks/train_colab.ipynb` in Google Colab öffnen
   (Laufzeit → Laufzeittyp ändern → **T4 GPU**).
2. Zelle 1 ausführen: Google Drive verbinden (einmal „Zulassen“ klicken).
3. Zelle 2 ausführen: Code klonen und installieren.
4. Zelle 3 ausführen: startet alle Phasen. Bricht die Sitzung ab, einfach
   Zelle 1–3 erneut ausführen – das Training setzt am letzten Checkpoint fort.
5. Optional Zelle 4: fertige Modelle zu Hugging Face hochladen
   (Token vorher unter 🔑 *Secrets* als `HF_TOKEN` speichern, nie in den Code).

**Lokale App (Laptop, ohne GPU)** – folgt in Meilenstein M7
(Doppelklick auf `start_app.command` bzw. `start_app.bat`, dann
http://localhost:7860 öffnen).

## Repository layout

| Path | Content |
|---|---|
| `configs/` | `quick.yaml` (this run), `full.yaml` (larger later run), `tiny.yaml` (CPU tests) |
| `memorylm/` | model, memory, data, training, checkpoints, evaluation |
| `notebooks/` | the 4-cell Colab/Kaggle notebook |
| `app/` | local Gradio app |
| `tests/` | the test suite that must pass before every training start |
| `scripts/` | setup and the secret scanner that blocks commits containing tokens |

## Data and licence

Training data: **TinyStories** by Ronen Eldan and Yuanzhi Li
([roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories),
files `TinyStoriesV2-GPT4-train.txt` / `-valid.txt`), released under the
**Community Data License Agreement – Sharing 1.0 (CDLA-Sharing-1.0)**.
The data is downloaded at run time and is not stored in this repository.
Paper: *TinyStories: How Small Can Language Models Be and Still Speak
Coherent English?* (arXiv:2305.07759).

Code in this repository: MIT licence (see `LICENSE`).

## Safety

No tokens or passwords are ever committed. A git hook
(`.githooks/pre-commit`, activated by `scripts/setup_dev.sh`) and a GitHub
Action scan every commit for access tokens and block it if one is found.
