"""Make the project root importable when running tests as plain scripts.

Python automatically imports `sitecustomize` (if present on `sys.path`) during
startup. Since running `python test_foo.py` from within `tests/` puts this folder
on `sys.path`, we can reliably add the repo root so `import a_gpt` works.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
