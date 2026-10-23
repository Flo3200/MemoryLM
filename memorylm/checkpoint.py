"""Crash-safe checkpoints, status file and metrics log.

Free Colab sessions end without warning. Everything needed to continue a
run EXACTLY is therefore saved regularly to Google Drive: model weights,
optimizer moments, gradient scaler, learning-rate position, data position,
all random-number-generator states, phase and step.

Writes are atomic: the file is first written under a temporary name and
then renamed. A rename is all-or-nothing, so a session that dies in the
middle of a save leaves the previous checkpoint intact instead of a
half-written, unreadable file.
"""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj) -> None:
    atomic_write_bytes(path, json.dumps(obj, indent=1, default=float).encode())


def rng_state() -> dict:
    s = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        s["cuda"] = torch.cuda.get_rng_state_all()
    return s


def set_rng_state(s: dict) -> None:
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    if "cuda" in s and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


class CheckpointManager:
    """Keeps the last `keep_last` checkpoints plus the best one (by
    validation loss) for one model in `<root>/checkpoints/<kind>/`."""

    def __init__(self, root: Path, kind: str, keep_last: int = 2):
        self.dir = root / "checkpoints" / kind
        self.dir.mkdir(parents=True, exist_ok=True)
        self.keep_last = keep_last

    def _numbered(self) -> list[Path]:
        files = [p for p in self.dir.glob("ckpt_*.pt")]
        return sorted(files, key=lambda p: int(p.stem.split("_")[1]))

    def save(self, state: dict, serial: int) -> Path:
        import io
        buf = io.BytesIO()
        torch.save(state, buf)
        path = self.dir / f"ckpt_{serial:08d}.pt"
        atomic_write_bytes(path, buf.getvalue())
        for old in self._numbered()[:-self.keep_last]:
            old.unlink(missing_ok=True)
        return path

    def save_best(self, state: dict) -> None:
        import io
        buf = io.BytesIO()
        torch.save(state, buf)
        atomic_write_bytes(self.dir / "best.pt", buf.getvalue())

    def load_latest(self, map_location="cpu") -> Optional[dict]:
        """Newest readable checkpoint; falls back to the older one if the
        newest is damaged (e.g. Drive sync interrupted)."""
        for path in reversed(self._numbered()):
            try:
                return torch.load(path, map_location=map_location, weights_only=False)
            except Exception as e:                       # noqa: BLE001 - any damage means "try the older one"
                print(f"  checkpoint {path.name} unreadable ({e}); trying an older one")
        return None

    def load_best(self, map_location="cpu") -> Optional[dict]:
        p = self.dir / "best.pt"
        return torch.load(p, map_location=map_location, weights_only=False) if p.exists() else None


class RunLog:
    """status.json (the current state, overwritten) and metrics_<kind>.jsonl
    (one line per evaluation, appended) in `<root>/logs/`."""

    def __init__(self, root: Path):
        self.dir = root / "logs"
        self.dir.mkdir(parents=True, exist_ok=True)

    def status(self) -> dict:
        p = self.dir / "status.json"
        try:
            return json.loads(p.read_text()) if p.exists() else {}
        except json.JSONDecodeError:
            return {}

    def update_status(self, **fields) -> None:
        s = self.status()
        for k, v in fields.items():
            if isinstance(v, dict) and isinstance(s.get(k), dict):
                s[k].update(v)
            else:
                s[k] = v
        s["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        atomic_write_json(self.dir / "status.json", s)

    def metric(self, kind: str, row: dict) -> None:
        row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), **row}
        with open(self.dir / f"metrics_{kind}.jsonl", "a") as f:
            f.write(json.dumps(row, default=float) + "\n")

    def read_metrics(self, kind: str) -> list[dict]:
        p = self.dir / f"metrics_{kind}.jsonl"
        if not p.exists():
            return []
        rows = []
        for line in p.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass                                    # a line cut off by a crash
        return rows
