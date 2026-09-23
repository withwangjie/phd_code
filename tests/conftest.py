"""Make the src/ layout importable when pytest is run from the repository root.

scripts/formal_preflight.sh also exports PYTHONPATH=src; this keeps a bare
`python -m pytest tests` working without installing the package.
"""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
