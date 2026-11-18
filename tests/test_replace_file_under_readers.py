"""A file is replaced while other threads read it (Windows refuses a plain os.replace then).

2026-11-18, a fresh Windows install: three of the first five runs ended `failed` with
"PermissionError: [WinError 5] Access is denied: ...run_<id>.json.<hex>.tmp -> ...run_<id>.json",
because the run file was open in a reader at the moment its save renamed over it.
"""

from __future__ import annotations

import threading
from pathlib import Path

from abstractruntime.utils.atomic_files import open_replaced, replace_file


def test_a_file_being_read_is_still_replaced(tmp_path: Path) -> None:
    target = tmp_path / "run.json"
    target.write_text("0", encoding="utf-8")
    stop = threading.Event()
    seen: list = []

    def read_forever() -> None:
        while not stop.is_set():
            with open_replaced(target, "rb") as fh:  # held open across the read, as every reader does
                seen.append(fh.read())
            stop.wait(0.0005)  # readers come and go; a file that is never closed cannot be replaced on Windows

    readers = [threading.Thread(target=read_forever, daemon=True) for _ in range(3)]
    for r in readers:
        r.start()
    try:
        for i in range(1, 301):
            tmp = tmp_path / f"run.json.{i}.tmp"
            tmp.write_text(str(i), encoding="utf-8")
            replace_file(tmp, target)
    finally:
        stop.set()
        for r in readers:
            r.join(5)
    assert not any(r.is_alive() for r in readers)
    assert target.read_text(encoding="utf-8") == "300"
    assert all(chunk.isdigit() for chunk in seen)  # whole old or whole new content, never a mixture
