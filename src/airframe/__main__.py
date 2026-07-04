"""Enable ``python -m airframe`` as an alias for the ``airframe`` CLI."""

from __future__ import annotations

import sys

from airframe.cli import main

if __name__ == "__main__":
    sys.exit(main())
