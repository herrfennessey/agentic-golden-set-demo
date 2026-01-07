"""
Pytest configuration for the test suite.

This file ensures that the src directory is on the Python path,
which helps PyCharm and other IDEs discover test modules correctly.
"""

import sys
from pathlib import Path

# Add src to Python path for imports
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
