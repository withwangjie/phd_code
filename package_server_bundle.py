#!/usr/bin/env python3
"""package_server_bundle.py

Lightweight, self-contained, stdlib-only packaging script for the
quantum-classical rotamer-recovery benchmark's remote-server deployment
bundle.

Run from the project's repository root (the directory containing
``launch_server.sh``, ``run_server_pipeline.py``, etc.). It packages
exactly the fixed whitelist of 15 files this server-side pipeline needs to
run (see ``REQUIRED_FILES`` below) into ``phd_research_server_bundle.tar.gz``,
alongside a cryptographically verifiable ``BUNDLE_MANIFEST.json`` describing
every packaged file's SHA-256 digest, size, and modification time -- so the
receiving server can verify the bundle's integrity and provenance before
ever executing anything from it.

Nothing outside the whitelist is ever inspected or packaged: no ``.git``
history, no ``.venv``, no historical ``experiments_run_*`` output, no cache
directories. If any one of the 15 required files is missing from the
current directory, this script prints every missing filename and exits
with status 1 -- it never produces a partial or silently-incomplete bundle.

``model_egnn_pruning.py`` was added to the whitelist after a real
deployment surfaced ``ImportError: cannot import name 'select_ablation_active'
from 'model_egnn_pruning'``: ``batch_benchmark_hard_set.py`` has an
unconditional, module-level ``from model_egnn_pruning import
select_ablation_active, build_ablation_subgraph`` (it is reached the
instant ``batch_benchmark_hard_set`` is imported at all, regardless of
which code path is used afterward), so every deployment needs it even
though ``run_server_pipeline.py`` never calls the ablation-only functions
it defines. A companion transitive-import audit of every whitelisted file
(including imports inside function bodies, not just module-level ones)
found no other repository-root module dependency missing from this list --
the only other local import outside the whitelist,
``batch_benchmark_hard_set.py``'s ``from run_real_complex_pilot import
main`` (module ``main()``, gated behind a ``--real-complex-pilot`` CLI
flag), is lazy/conditional and is never reached by anything
``run_server_pipeline.py`` calls.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
import tarfile
import time
from pathlib import Path
from typing import Dict, List

# ---------------------------------------------------------------------------
# Whitelist: exactly these 15 repository-root files are eligible for
# packaging. This list is the single source of truth for what "the core
# files this server needs to run" means -- nothing is added implicitly by
# globbing a directory, so a stray file can never sneak into the bundle.
# ---------------------------------------------------------------------------
REQUIRED_FILES: List[str] = [
    # Scheduling & batch-execution control
    "launch_server.sh",
    "run_server_pipeline.py",
    "generate_final_report.py",
    # Core algorithm & evaluation business modules
    "qaoa_interface_sampler.py",
    "evaluate_complex_metrics.py",
    "evaluate_complex_dockq.py",
    "batch_benchmark_hard_set.py",
    "subgraph_to_qubo.py",
    "structural_quality.py",
    "prediction_contract.py",
    "validate_qaoa_repairs.py",
    # model_egnn_pruning.py: an unconditional, module-level import of
    # batch_benchmark_hard_set.py (`from model_egnn_pruning import
    # select_ablation_active, build_ablation_subgraph`) -- required to
    # import batch_benchmark_hard_set at all, not only for its ablation
    # code paths. See the module docstring for the deployment failure that
    # surfaced this.
    "model_egnn_pruning.py",
    # Automated test / preflight quality-gate suite
    "test_audit_remediation.py",
    "test_robust_qaoa.py",
    "test_evaluate_complex_metrics.py",
]

BUNDLE_NAME = "phd_research_server_bundle.tar.gz"
MANIFEST_NAME = "BUNDLE_MANIFEST.json"
LAUNCH_SCRIPT_NAME = "launch_server.sh"
LAUNCH_SCRIPT_MODE = 0o755   # explicit Unix executable permission
DEFAULT_FILE_MODE = 0o644


# ---------------------------------------------------------------------------
# Step 1: strong preflight existence validation
# ---------------------------------------------------------------------------

def verify_required_files(root: Path, required: List[str]) -> List[Path]:
    """Check every required file exists directly under ``root``.

    Returns the resolved path of each required file, in ``required``'s
    order, if -- and only if -- every single one is present. Otherwise
    prints every missing filename to stderr and exits with status 1,
    before any hashing or archive creation begins: this script never
    packages an incomplete file set.
    """
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        print(
            f"ERROR: {len(missing)} required file(s) are missing from {root}; "
            f"no bundle was created:",
            file=sys.stderr,
        )
        for name in missing:
            print(f"  - {name}", file=sys.stderr)
        sys.exit(1)
    return [root / name for name in required]


# ---------------------------------------------------------------------------
# Step 2: cryptographically verifiable manifest
# ---------------------------------------------------------------------------

def _sha256_of(path: Path) -> str:
    """Stream a file's SHA-256 digest without loading it fully into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_timestamp(epoch_seconds: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(epoch_seconds))


def build_manifest(paths: List[Path], bundle_name: str) -> Dict[str, object]:
    """SHA-256 digest, size in bytes, and modification time (UTC) for every
    packaged file, so the receiving server can verify each file's integrity
    and provenance before running anything from the bundle.
    """
    files: Dict[str, Dict[str, object]] = {}
    for path in paths:
        stat = path.stat()
        files[path.name] = {
            "sha256": _sha256_of(path),
            "size_bytes": stat.st_size,
            "mtime_utc": _utc_timestamp(stat.st_mtime),
        }
    return {
        "bundle_name": bundle_name,
        "generated_utc": _utc_timestamp(time.time()),
        "file_count": len(paths),
        "files": files,
    }


# ---------------------------------------------------------------------------
# Step 3: standardized, flat, gzip-compressed archive
# ---------------------------------------------------------------------------

def _flat_tarinfo(path: Path, *, mode: int) -> tarfile.TarInfo:
    """A ``TarInfo`` for ``path`` whose archive name is just the bare
    filename (no directory prefix, no local absolute path -- so the
    archive never leaks this machine's directory layout), with a fixed
    ownership/mode so the archive's contents are reproducible regardless
    of which local user account built it.
    """
    tarinfo = tarfile.TarInfo(name=path.name)
    stat = path.stat()
    tarinfo.size = stat.st_size
    tarinfo.mtime = int(stat.st_mtime)
    tarinfo.mode = mode
    tarinfo.uid = 0
    tarinfo.gid = 0
    tarinfo.uname = ""
    tarinfo.gname = ""
    tarinfo.type = tarfile.REGTYPE
    return tarinfo


def write_bundle(paths: List[Path], manifest: Dict[str, object], bundle_path: Path) -> None:
    """Write every required file plus the manifest into a gzip-compressed
    tar archive at ``bundle_path``. Every archive member is flattened to
    its bare filename. ``launch_server.sh`` is explicitly given Unix
    executable permission (0o755); every other file gets 0o644. The
    manifest is serialized straight into the archive from memory (never
    written as a loose file in the project directory first), so packaging
    never leaves stray byproducts behind.
    """
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")

    with tarfile.open(bundle_path, "w:gz") as tar:
        for path in paths:
            mode = LAUNCH_SCRIPT_MODE if path.name == LAUNCH_SCRIPT_NAME else DEFAULT_FILE_MODE
            tarinfo = _flat_tarinfo(path, mode=mode)
            with path.open("rb") as handle:
                tar.addfile(tarinfo, handle)

        manifest_tarinfo = tarfile.TarInfo(name=MANIFEST_NAME)
        manifest_tarinfo.size = len(manifest_bytes)
        manifest_tarinfo.mtime = int(time.time())
        manifest_tarinfo.mode = DEFAULT_FILE_MODE
        manifest_tarinfo.uid = 0
        manifest_tarinfo.gid = 0
        manifest_tarinfo.uname = ""
        manifest_tarinfo.gname = ""
        tar.addfile(manifest_tarinfo, io.BytesIO(manifest_bytes))


# ---------------------------------------------------------------------------
# Step 4: terminal feedback
# ---------------------------------------------------------------------------

def _human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0 or unit == "GB":
            return f"{size:.2f} {unit}" if unit != "B" else f"{int(size)} {unit}"
        size /= 1024.0
    return f"{size:.2f} GB"


def report(bundle_path: Path, paths: List[Path]) -> None:
    bundle_size = bundle_path.stat().st_size
    print("=== phd_research_server_bundle packaging complete ===")
    print(f"Archive:            {bundle_path}")
    print(f"Archive size:       {_human_size(bundle_size)} ({bundle_size} bytes)")
    print(f"Core files packed:  {len(paths)}")
    print(f"Archive members:    {len(paths) + 1} (core files + {MANIFEST_NAME})")
    print("")
    print("Server-side one-shot extract + launch:")
    print(f"  mkdir -p ./phd-research-server && "
          f"tar -xzf {bundle_path.name} -C ./phd-research-server && "
          f"cd ./phd-research-server && "
          f"bash launch_server.sh")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> int:
    root = Path(__file__).resolve().parent
    paths = verify_required_files(root, REQUIRED_FILES)
    manifest = build_manifest(paths, BUNDLE_NAME)
    bundle_path = root / BUNDLE_NAME
    write_bundle(paths, manifest, bundle_path)
    report(bundle_path, paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
