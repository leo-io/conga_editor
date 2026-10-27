import sys
from pathlib import Path

# Ensure the repository root is in sys.path
root_app = Path(__file__).resolve().parent.parent
if str(root_app) not in sys.path:
    sys.path.insert(0, str(root_app))
