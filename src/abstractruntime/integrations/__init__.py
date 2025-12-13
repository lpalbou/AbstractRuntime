"""abstractruntime.integrations

Integration modules live here.

Design rule (layered coupling):
- The **kernel** (`abstractruntime.core`, `abstractruntime.storage`, `abstractruntime.identity`) stays dependency-light.
- Optional integration packages may import heavier dependencies (e.g. AbstractCore) and provide effect handlers.

This package intentionally does not import any integration by default.
"""

