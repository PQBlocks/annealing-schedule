"""conftest.py — adds src/ to sys.path so `import src` and `import annealing_types` work."""
import sys
import os as _os

# __file__ = tests/conftest.py → src/ is two levels up from conftest.py
# Actually: tests/conftest.py -> project_root/tests/conftest.py
# src/ is at project_root/src/
_project_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_src_dir = _os.path.join(_project_root, 'src')
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
# Also add project root so `from src import ...` works
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
