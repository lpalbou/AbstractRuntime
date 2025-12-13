"""abstractruntime.integrations.abstractcore.logging

Logging adapter for the AbstractCore-integrated runtime.

We prefer AbstractCore's structured logger for consistency across the stack.
"""

from __future__ import annotations

from typing import Any


def get_logger(name: str) -> Any:
    """Return a logger compatible with AbstractCore's structured logger.

    This is intentionally a thin wrapper to keep the integration layer small.
    """

    try:
        from abstractcore.utils.structured_logging import get_logger as _get_logger

        return _get_logger(name)
    except Exception:  # pragma: no cover
        import logging

        return logging.getLogger(name)

