"""MemoryLM local app (CPU), http://localhost:7860

Start: double-click start_app.command (Mac) / start_app.bat (Windows), or
    python app/app.py

Tabs: 1 chat & compare, 2 memory test, 3 memory inspector,
      4 training dashboard, 5 model download.
A "run folder" holds the trained models: either downloaded from the
Hugging Face Hub (tab 5) or the synced Google Drive folder
(…/MemoryLM/run_quick).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import gradio as gr                     # noqa: E402
import torch                            # noqa: E402

from memorylm.config import load_config        # noqa: E402
from memorylm.evaluate import generate, load_model, read_text   # noqa: E402

MODELS_DIR = ROOT / "app" / "models"
SETTINGS = ROOT / "app" / "settings.json"
torch.set_grad_enabled(False)


# ======================================================================
# loading a run folder
# ======================================================================
class Run:
    """The active run: config, tokenizer and the two (or three) models."""

    def __init__(self):
        self.folder = None
        self.cfg = self.tok = None
        self.models = {}

    def find_tokenizer(self, folder: Path) -> Path | None:
        for cand in [folder / "tokenizer.json", *folder.parent.glob("data_v*/tokenizer.json")]:
            if cand.exists():
                return cand
        return None

    def load(self, folder: str) -> str:
        from tokenizers import Tokenizer
        folder = Path(folder).expanduser()
        if not folder.exists():
            return f"Folder not found: {folder}"
        files = {"baseline": folder / "baseline_phase1_final.pt",
                 "memorylm": folder / "memorylm_phase2_final.pt"}
        if not files["memorylm"].exists():
            files["memorylm"] = folder / "memorylm_phase1_final.pt"
        missing = [k for k, f in files.items() if not f.exists()]
        if missing:
            return f"No finished {', '.join(missing)} model in {folder} (training not finished yet?)"
        tok_path = self.find_tokenizer(folder)
        if tok_path is None:
            return f"tokenizer.json not found in {folder} or ../data_v*/"
        preset = torch.load(files["memorylm"], map_location="cpu", weights_only=False).get("preset", "quick")
        cfg_dir = folder / "configs"
        self.cfg = load_config(str(cfg_dir / f"{preset}.yaml") if (cfg_dir / f"{preset}.yaml").exists() else preset)
        self.tok = Tokenizer.from_file(str(tok_path))
        cpu = torch.device("cpu")
        self.models = {k: load_model(self.cfg, k, f, cpu) for k, f in files.items()}
        self.folder = folder
        try:
            SETTINGS.write_text(json.dumps({"run_folder": str(folder)}))
        except OSError:
            pass
        phase = "after phase 2" if "phase2" in files["memorylm"].name else "after phase 1 only"
        return f"Loaded run '{folder.name}' (preset {preset}, MemoryLM {phase})."

    def ready(self) -> bool:
        return bool(self.models)

    def enc(self, text: str) -> list[int]:
        return self.tok.encode(text).ids

    def dec(self, ids) -> str:
        return self.tok.decode([int(i) for i in ids])

    def sid(self, name: str) -> int:
        return self.tok.token_to_id(name)


RUN = Run()
NOT_READY = "No model loaded yet. Go to tab 5 (Models) and load a run folder."


# ======================================================================
# tab 1: chat and compare
# ======================================================================
def chat(prompt, max_new, temperature, top_k, seed):
    if not RUN.ready():
        return NOT_READY, NOT_READY
    ids = RUN.enc(prompt)[-(RUN.cfg.model.context - 8):]       # keep room for an answer
    x = torch.tensor([ids])
    outs = []
    for kind in ("baseline", "memorylm"):
        g = torch.Generator().manual_seed(int(seed))
        y = generate(RUN.models[kind], x, int(max_new), temperature=float(temperature), top_k=int(top_k),
                     stop_id=RUN.sid("<eos>"), generator=g)[0].tolist()
        if RUN.sid("<eos>") in y:
            y = y[:y.index(RUN.sid("<eos>"))]
        outs.append(prompt + RUN.dec(y))
    return outs[0], outs[1]


# ======================================================================
# tab 2: memory test
# ======================================================================
def memory_test(text):
    if not RUN.ready():
        return NOT_READY, "", ""
    ids = RUN.enc(text.strip())
    n = min(len(ids), RUN.cfg.model.context - 1)
    if n < 2:
        return "Please paste a longer text.", "", ""
    x = torch.tensor([ids])
    m = RUN.models["memorylm"]
    state = read_text(m, x)                                     # read everything; memory persists
    prompt = torch.tensor([[RUN.sid("<recite>")]])               # context cleared: only <recite>
    rec = generate(m, prompt, n, state=state)[0].tolist()
    rec_off = generate(m, prompt, n, use_memory=False)[0].tolist()
    target = ids[:n]

    def score(y):
        return sum(int(a == b) for a, b in zip(y, target)) / n

    info = (f"Read {len(ids)} tokens, memory now holds {int(state.knowledge.n_written().sum())} knowledge and "
            f"{int(state.rule.n_written().sum())} rule entries = {state.size_kb():.1f} KB.\n"
            f"Token accuracy of the first {n} tokens: memory ON {score(rec):.0%}, memory OFF {score(rec_off):.0%} "
            f"(OFF = what the model guesses with no information about your text).")
    return info, RUN.dec(rec), RUN.dec(rec_off)


# ======================================================================
# tab 3: memory inspector
# ======================================================================
def inspect(text):
    if not RUN.ready():
        return [], [], NOT_READY
    ids = RUN.enc(text.strip())[:RUN.cfg.model.context]
    if not ids:
        return [], [], "Please paste a text."
    m = RUN.models["memorylm"]
    state = m.new_state(trace=True)
    m.process(torch.tensor([ids]), state, write=True)
    gates = torch.cat(state.trace, 1)[0]                        # (T, 2) gate logits
    pieces = [RUN.dec([i]) for i in ids]
    know = [(p, f"write {torch.sigmoid(g[0]).item():.2f}" if g[0] > 0 else None) for p, g in zip(pieces, gates)]
    rule = [(p, f"write {torch.sigmoid(g[1]).item():.2f}" if g[1] > 0 else None) for p, g in zip(pieces, gates)]
    rates = (gates > 0).float().mean(0).tolist()
    beta = m.memory.beta.tolist()
    alpha = m.memory.alpha.tolist()
    summary = (f"{len(ids)} tokens · knowledge gate wrote {rates[0]:.0%} · rule gate wrote {rates[1]:.0%} · "
               f"memory size {state.size_kb():.1f} KB (fp16, non-zeros only)\n"
               f"Learned read strengths per layer: knowledge β = {[round(b, 3) for b in beta]}, "
               f"rule α = {[round(a, 3) for a in alpha]} (0 = layer ignores the memory)\n"
               "Note: tokens of the last 32-token chunk are written after the chunk ends; "
               "within one text they are only readable from the next chunk on.")
    return know, rule, summary


# ======================================================================
# tab 4: training dashboard
# ======================================================================
def dashboard(folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = Path(folder or (RUN.folder or "")).expanduser()
    logs = folder / "logs"
    if not logs.exists():
        return None, None, f"No logs/ folder in {folder}", ""
    status = json.loads((logs / "status.json").read_text()) if (logs / "status.json").exists() else {}
    rows = {}
    for kind in ("baseline", "memorylm"):
        p = logs / f"metrics_{kind}.jsonl"
        if p.exists():
            seen = {}
            for line in p.read_text().splitlines():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                seen[(r["phase"], r["step"])] = r            # duplicates after a resume: keep the last
            rows[kind] = sorted(seen.values(), key=lambda r: r["tokens"])
    fig1, ax = plt.subplots(figsize=(7, 4))
    for kind, color in (("baseline", "#c44e52"), ("memorylm", "#2a6fdb")):
        rs = rows.get(kind, [])
        if rs:
            ax.plot([r["tokens"] / 1e6 for r in rs], [r["val_loss"] for r in rs], color=color, lw=2, label=f"{kind} validation")
            ax.plot([r["tokens"] / 1e6 for r in rs], [r["train_loss"] for r in rs], color=color, lw=1, alpha=0.4, label=f"{kind} training")
    ax.set(xlabel="training tokens (millions)", ylabel="loss", title="Loss curves")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig1.tight_layout()
    fig2, ax = plt.subplots(figsize=(7, 4))
    rs = [r for r in rows.get("memorylm", []) if "write_rate" in r]
    if rs:
        x = [r["tokens"] / 1e6 for r in rs]
        ax.plot(x, [r["write_rate"][0] for r in rs], lw=2, label="knowledge write rate", color="#2a6fdb")
        ax.plot(x, [r["write_rate"][1] for r in rs], lw=2, label="rule write rate", color="#55a868")
        rr = [r for r in rs if "recite_acc" in r]
        if rr:
            ax.plot([r["tokens"] / 1e6 for r in rr], [r["recite_acc"] for r in rr], lw=2, ls="--",
                    label="recitation accuracy (training)", color="#8172b3")
    ax.set(xlabel="training tokens (millions)", ylim=(0, 1.05), title="MemoryLM: writing and recitation")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig2.tight_layout()
    report = folder / "eval" / "REPORT.md"
    rep = report.read_text() if report.exists() else "_No evaluation report yet._"
    return fig1, fig2, json.dumps(status, indent=1)[:4000], rep


# ======================================================================
# tab 5: models
# ======================================================================
def download(repo_id):
    from memorylm.hub import download_run
    repo_id = repo_id.strip()
    if "/" not in repo_id:
        return "Enter a repository like  username/memorylm-quick", ""
    dest = MODELS_DIR / repo_id.replace("/", "__")
    try:
        download_run(repo_id, dest)
    except Exception as e:                     # noqa: BLE001 - show any download problem to the user
        return f"Download failed: {e}", ""
    return RUN.load(str(dest)), str(dest)


def default_folder() -> str:
    try:
        return json.loads(SETTINGS.read_text())["run_folder"]
    except Exception:                          # noqa: BLE001
        pass
    for cand in sorted(Path.home().glob("Library/CloudStorage/GoogleDrive-*/My Drive/MemoryLM/run_quick")) + \
            sorted(Path.home().glob("Google Drive/My Drive/MemoryLM/run_quick")) + sorted(MODELS_DIR.glob("*")):
        if cand.exists():
            return str(cand)
    return ""


def build_ui():
    with gr.Blocks(title="MemoryLM") as ui:
        gr.Markdown("# MemoryLM\nBaseline transformer vs. transformer with external memory – both trained from scratch on TinyStories.")
        status = gr.Markdown()
        with gr.Tab("1 · Chat & compare"):
            prompt = gr.Textbox(label="Start of a story", value="Once upon a time, there was a little dog named")
            with gr.Row():
                max_new = gr.Slider(8, 120, 60, step=1, label="new tokens")
                temp = gr.Slider(0, 1.5, 0.8, step=0.05, label="temperature (0 = always most likely)")
                topk = gr.Slider(0, 200, 40, step=1, label="top-k (0 = off)")
                seed = gr.Number(0, label="seed", precision=0)
            go = gr.Button("Generate", variant="primary")
            with gr.Row():
                out_b = gr.Textbox(label="Baseline", lines=8)
                out_m = gr.Textbox(label="MemoryLM", lines=8)
            go.click(chat, [prompt, max_new, temp, topk, seed], [out_b, out_m])
        with gr.Tab("2 · Memory test"):
            gr.Markdown("MemoryLM **reads** your text (writing into its memory), then its **context is cleared** "
                        "completely. Given only `<recite>`, it tries to reproduce the text from memory.")
            txt = gr.Textbox(label="Text to memorise", lines=5,
                             value="Lily had a red kite. She took it to the park with her brother Max.")
            run_btn = gr.Button("Read → clear context → recite", variant="primary")
            info = gr.Textbox(label="Result", lines=3)
            with gr.Row():
                rec_on = gr.Textbox(label="Recited WITH memory", lines=5)
                rec_off = gr.Textbox(label="Recited with memory OFF (lower bound)", lines=5)
            run_btn.click(memory_test, [txt], [info, rec_on, rec_off])
        with gr.Tab("3 · Memory inspector"):
            txt3 = gr.Textbox(label="Text", lines=4, value="Tom saw a big cat. The cat was orange and very happy.")
            b3 = gr.Button("Show what is written", variant="primary")
            s3 = gr.Textbox(label="Summary", lines=4)
            k3 = gr.HighlightedText(label="Knowledge store: highlighted tokens are written", combine_adjacent=False)
            r3 = gr.HighlightedText(label="Rule store: highlighted tokens are written", combine_adjacent=False)
            b3.click(inspect, [txt3], [k3, r3, s3])
        with gr.Tab("4 · Training dashboard"):
            f4 = gr.Textbox(label="Run folder (empty = the loaded run)", value="")
            b4 = gr.Button("Refresh", variant="primary")
            with gr.Row():
                p1 = gr.Plot(label="Loss")
                p2 = gr.Plot(label="Memory")
            st4 = gr.Code(label="status.json", language="json")
            rep4 = gr.Markdown()
            b4.click(dashboard, [f4], [p1, p2, st4, rep4])
        with gr.Tab("5 · Models"):
            gr.Markdown("Load a run either from the **Hugging Face Hub** or from the **synced Google Drive folder** "
                        "(e.g. `~/Library/CloudStorage/GoogleDrive-…/My Drive/MemoryLM/run_quick`).")
            with gr.Row():
                repo = gr.Textbox(label="Hugging Face repository", placeholder="username/memorylm-quick")
                dl = gr.Button("Download and load")
            with gr.Row():
                folder = gr.Textbox(label="Local run folder", value=default_folder())
                ld = gr.Button("Load folder", variant="primary")
            msg = gr.Textbox(label="Status")
            dl.click(download, [repo], [msg, folder])
            ld.click(RUN.load, [folder], [msg])
        ui.load(lambda: RUN.load(default_folder()) if default_folder() else NOT_READY, None, status)
    return ui


if __name__ == "__main__":
    build_ui().launch(server_name="127.0.0.1", server_port=7860, inbrowser="--no-browser" not in sys.argv)
