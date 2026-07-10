"""DEPRECATED SHIM — the writer lease moved to `abstractruntime.storage.lease`.

Re-homed under the 2026-07-10 vocabulary sign-off (a2a/fs/renaming.md,
maintainer-approved): the mechanism is a generic one-writer-per-directory
file lease; "home" only named its first consumer. This module re-exports the
storage implementation under the old names for the migration window and
DIES BEFORE RELEASE — no two spellings survive to publication.

Both spellings alias ONE implementation and ONE lock file
(`storage.lease.LEASE_FILENAME`), so mixed old/new callers keep excluding
each other correctly during the window.
"""

from __future__ import annotations

from ..storage.lease import (
    HOLDER_KINDS,
    LEASE_FILENAME,
    DirectoryLease,
    DirectoryLeaseHeld,
    acquire_directory_lease,
    read_directory_lease,
)

# Old spellings -> the one implementation (class identity preserved: an
# `except HomeLeaseHeld` catches exactly what the new code raises).
HomeLease = DirectoryLease
HomeLeaseHeld = DirectoryLeaseHeld
acquire_home_lease = acquire_directory_lease
read_home_lease = read_directory_lease

__all__ = [
    "HOLDER_KINDS",
    "LEASE_FILENAME",
    "DirectoryLease",
    "DirectoryLeaseHeld",
    "HomeLease",
    "HomeLeaseHeld",
    "acquire_directory_lease",
    "acquire_home_lease",
    "read_directory_lease",
    "read_home_lease",
]
