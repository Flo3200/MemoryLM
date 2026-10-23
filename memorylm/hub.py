"""Upload a finished run to the free Hugging Face Hub, and download it again
(used by the local app).

The token is NEVER written into code or files: on Colab it comes from the
secrets manager (key icon in the left sidebar, name HF_TOKEN), on Kaggle from
"Add-ons -> Secrets", locally from the environment variable HF_TOKEN.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

# Files of a run that are worth publishing (small: ~20 MB per model).
RUN_FILES = ["plan.json", "baseline_phase1_final.pt", "memorylm_phase1_final.pt", "memorylm_phase2_final.pt"]
RUN_DIRS = ["logs", "eval"]


def get_token() -> Optional[str]:
    try:                                               # Colab secrets manager
        from google.colab import userdata
        return userdata.get("HF_TOKEN")
    except Exception:                                  # noqa: BLE001 - not on Colab or secret missing
        pass
    try:                                               # Kaggle secrets
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:                                  # noqa: BLE001
        pass
    return os.environ.get("HF_TOKEN")


def upload_run(cfg, repo_name: Optional[str] = None, private: bool = False) -> str:
    """Upload models, tokenizer, logs and evaluation results of a run."""
    from huggingface_hub import HfApi
    from .data import data_dir
    from .train import run_dir
    token = get_token()
    if not token:
        raise SystemExit("No HF_TOKEN found. Colab: key icon on the left -> add secret HF_TOKEN -> allow notebook access.")
    api = HfApi(token=token)
    user = api.whoami()["name"]
    repo_id = f"{user}/{repo_name or 'memorylm-' + cfg.run_name}"
    api.create_repo(repo_id, private=private, exist_ok=True)
    rd = run_dir(cfg)
    for f in RUN_FILES:
        if (rd / f).exists():
            api.upload_file(path_or_fileobj=str(rd / f), path_in_repo=f, repo_id=repo_id)
    for d in RUN_DIRS:
        if (rd / d).exists():
            api.upload_folder(folder_path=str(rd / d), path_in_repo=d, repo_id=repo_id)
    api.upload_file(path_or_fileobj=str(data_dir(cfg) / "tokenizer.json"), path_in_repo="tokenizer.json", repo_id=repo_id)
    cfg_path = Path(__file__).resolve().parent.parent / "configs"
    api.upload_folder(folder_path=str(cfg_path), path_in_repo="configs", repo_id=repo_id)
    print(f"uploaded to https://huggingface.co/{repo_id}")
    return repo_id


def download_run(repo_id: str, dest: Path) -> Path:
    """Download a published run (no token needed for public repos)."""
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo_id, local_dir=str(dest)))
