"""Root pytest configuration shared by the unit and integration suites.

The directory split carries the meaning the ``integration`` marker used
to carry alone: anything under ``tests/integration/`` talks to a real
vendor endpoint and is marked automatically here, so a new integration
module cannot forget the marker and quietly join the default run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_INTEGRATION_DIR = Path(__file__).parent / "integration"


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Mark every test collected under ``tests/integration/``.

    Args:
        config: The active pytest configuration.
        items: Collected items, mutated in place.
    """
    for item in items:
        if _INTEGRATION_DIR in Path(str(item.fspath)).parents:
            item.add_marker(pytest.mark.integration)
