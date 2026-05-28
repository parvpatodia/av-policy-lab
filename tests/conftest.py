"""Pytest path setup for the planner-geometry suite.

planners.py lives in the repo's nuplan/ directory and imports the nuPlan
devkit (`from nuplan.common... import ...`) at module level. The devkit is
installed in the `nuplan` conda env, so it resolves on sys.path automatically.
We only need to add the repo's nuplan/ directory so `import planners` resolves
to the local module.
"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_NUPLAN_DIR = _REPO_ROOT / "nuplan"

# WHY append (not insert at 0): the installed `nuplan` devkit package must keep
# priority for `import nuplan`. We only want `import planners` to find the local
# module file, which lives inside nuplan/ but is itself a top-level module.
if str(_NUPLAN_DIR) not in sys.path:
    sys.path.append(str(_NUPLAN_DIR))
