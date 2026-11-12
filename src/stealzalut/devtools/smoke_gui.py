"""Drive the real window with the look method on one pair and grab it.

A visual check of the interface that needs no interaction: the window is
built as the application builds it, a pair is loaded, and a capture is
written once the preview is ready.  Nothing is written in the repository.

Usage: python -m stealzalut.devtools.smoke_gui SOURCE REFERENCE OUT_FOLDER
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from pathlib import Path

from stealzalut.config import load_settings
from stealzalut.gui.main_window import MainWindow
from stealzalut.main import create_application, load_stylesheet

#: Width the controls row must fit in.
WINDOW_SIZE = (1280, 800)


def main(argv: list[str]) -> int:
    """Load the pair, wait for the preview, write ``ui_look.png``."""
    if len(argv) != 3:
        print(__doc__)
        return 2
    source, reference, out = argv[0], argv[1], Path(argv[2])
    out.mkdir(parents=True, exist_ok=True)

    app = create_application([])
    load_stylesheet(app)
    window = MainWindow(settings=load_settings(None))
    window.resize(*WINDOW_SIZE)
    window.show()
    window._algo_combo.setCurrentIndex(window._algo_combo.findData("look"))
    window._source_zone.load_from_path(source)
    window._reference_zone.load_from_path(reference)

    def pump(done: Callable[[], bool], timeout: float = 60.0) -> None:
        end = time.monotonic() + timeout
        while time.monotonic() < end and not done():
            app.processEvents()
            time.sleep(0.01)

    def idle() -> bool:
        worker = window._transfer_worker
        return window._preview.image_rgb is not None and not (worker and worker.isRunning())

    pump(idle)
    pump(lambda: False, 0.5)  # let the debounced re-render start, if any
    pump(idle)

    preview = window._preview.image_rgb
    print("method:", window._current_method)
    print("preview:", None if preview is None else preview.shape)
    print("window:", window.width(), "x", window.height())
    window.grab().save(str(out / "ui_look.png"))
    window.stop_workers()
    return 0 if preview is not None else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
