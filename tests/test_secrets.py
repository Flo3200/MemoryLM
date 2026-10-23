"""The secret scanner must catch real-looking tokens and pass clean code."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("check_secrets", ROOT / "scripts" / "check_secrets.py")
cs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cs)


def test_detects_tokens():
    # Built by concatenation so this test file itself does not trip the scanner.
    fake_hf = "hf_" + "A1b2C3d4" * 5
    fake_gh = "ghp_" + "Z9y8X7w6" * 5
    assert cs.scan_text(f"token = '{fake_hf}'")
    assert cs.scan_text(fake_gh)
    assert cs.scan_text("password = " + "'hunter2hunter2'")


def test_clean_code_passes():
    assert not cs.scan_text("token = userdata.get('HF_TOKEN')")
    assert not cs.scan_text("d_model: 256")


def test_repository_is_clean():
    for path in ROOT.rglob("*"):
        if any(p in path.parts for p in (".venv", ".git", "runs", "__pycache__")):
            continue
        if path.is_file() and path.suffix in {".py", ".yaml", ".md", ".ipynb", ".txt", ".sh", ".yml", ".json"}:
            assert not cs.scan_text(path.read_text(errors="ignore")), path
