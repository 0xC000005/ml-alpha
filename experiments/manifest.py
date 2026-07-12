"""Run provenance manifest.

A result without a manifest is not reproducible, and this repo currently records a git SHA
for one experiment in five. This writes the four things needed to reconstruct any number:
which code, which config, which seeds, which data.

Use as a library from a driver:

    from experiments.manifest import write_manifest
    write_manifest(run_dir, cfg)

or standalone from sweep.sbatch, before the python run:

    python experiments/manifest.py output/exp/<screen>/<tag>_<year> "$CFG"

The data hash is the expensive part, so it is cached in .data_hash.json at the repo root
and only recomputed when a source file's size or mtime changes.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

# Paths mirror the Config defaults in train_nn.py:32-34 (data_dir / macro_file /
# sector_file). The 95 signal_*.parquet files are hashed as a set, not individually.
DATA_ROOT = Path("ml_alpha_data")
DATA_FILES = [
    DATA_ROOT / "gkx_full" / "returns.parquet",
    DATA_ROOT / "gkx_full" / "universe.parquet",
    DATA_ROOT / "gkx_full" / "sector_mapping.csv",
    DATA_ROOT / "welch_goyal_2024.xlsx",
    DATA_ROOT / "fama_french_factors.xlsx",
]
SIGNAL_GLOB = DATA_ROOT / "gkx_full"
HASH_CACHE = Path(".data_hash.json")


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def data_hashes(files: list[Path] | None = None) -> dict[str, str]:
    """SHA-256 of each data input, cached on (size, mtime)."""
    files = list(files or DATA_FILES)
    cache = json.loads(HASH_CACHE.read_text()) if HASH_CACHE.exists() else {}
    out, stale = {}, False

    def _hash(p: Path) -> str:
        nonlocal stale
        st = p.stat()
        key = f"{st.st_size}:{int(st.st_mtime)}"
        hit = cache.get(str(p))
        if hit and hit.get("key") == key:
            return hit["sha256"]
        digest = _sha256(p)
        cache[str(p)] = {"key": key, "sha256": digest}
        stale = True
        return digest

    for p in files:
        out[str(p)] = _hash(p) if p.exists() else "MISSING"

    # The 95 per-signal parquets are one logical input; collapse them into a single
    # order-independent digest so a swapped signal file still changes the manifest.
    signals = sorted(SIGNAL_GLOB.glob("signal_*.parquet"))
    if signals:
        roll = hashlib.sha256()
        for p in signals:
            roll.update(p.name.encode())
            roll.update(_hash(p).encode())
        out[f"{SIGNAL_GLOB}/signal_*.parquet"] = f"{roll.hexdigest()} ({len(signals)} files)"
    else:
        out[f"{SIGNAL_GLOB}/signal_*.parquet"] = "MISSING"

    if stale:
        HASH_CACHE.write_text(json.dumps(cache, indent=2))
    return out


def build_manifest(cfg: dict | str | None = None) -> dict:
    try:
        import torch

        torch_ver = torch.__version__
        cuda_ver = torch.version.cuda
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        torch_ver = cuda_ver = gpu = None

    if isinstance(cfg, str):
        try:
            cfg = json.loads(cfg)
        except json.JSONDecodeError:
            cfg = {"raw": cfg}

    status = _git("status", "--porcelain")
    return {
        "git_sha": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        # A dirty tree means the code that ran is not the code that is committed. Any
        # result produced under dirty=true is unreproducible; say so loudly.
        "dirty": bool(status),
        "dirty_files": status.splitlines() if status else [],
        "config": cfg,
        "data": data_hashes(),
        "python": sys.version.split()[0],
        "torch": torch_ver,
        "cuda": cuda_ver,
        "gpu": gpu,
        "hostname": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }


def write_manifest(run_dir: str | Path, cfg: dict | str | None = None) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "manifest.json"
    path.write_text(json.dumps(build_manifest(cfg), indent=2))
    return path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python experiments/manifest.py <run_dir> [config_json]")
    p = write_manifest(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    m = json.loads(p.read_text())
    warn = "  ** DIRTY TREE — result is not reproducible **" if m["dirty"] else ""
    print(f"wrote {p}  (git {m['git_sha'][:8]}){warn}")
