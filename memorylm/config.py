"""Loading the YAML presets.

Why a config file at all? Both models and all phases must run under
identical conditions. Keeping every number in one file (instead of
scattered through the code) makes the comparison auditable and lets us
switch from the "quick" to the "full" run with one flag: --preset full.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


class Config(dict):
    """A dict that also allows attribute access: cfg.model.d_model."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def _wrap(obj: Any) -> Any:
    if isinstance(obj, dict):
        return Config({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    return obj


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_raw(path: Path) -> dict:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    parent = raw.pop("inherit", None)
    if parent:
        raw = _deep_merge(_load_raw(path.parent / parent), raw)
    return raw


def _parse_value(text: str) -> Any:
    # "--set model.d_model=128" arrives as a string; YAML turns it into 128.
    return yaml.safe_load(text)


def load_config(preset: str = "quick", overrides: list[str] | None = None) -> Config:
    """Load configs/<preset>.yaml (or a path) and apply "a.b=c" overrides."""
    path = Path(preset)
    if not path.suffix:
        path = CONFIG_DIR / f"{preset}.yaml"
    raw = _load_raw(path)
    for item in overrides or []:
        key, _, value = item.partition("=")
        node = raw
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _parse_value(value)
    return _wrap(raw)
