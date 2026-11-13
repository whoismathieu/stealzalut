"""Drive the real window on one pair and grab it.

A visual check of the interface that needs no interaction: the window is
built as the application builds it, a pair is loaded, and a capture is
written once the preview is ready.  Nothing is written in the repository.

Usage: python -m stealzalut.devtools.smoke_gui SOURCE REFERENCE OUT_FOLDER [METHOD]

METHOD is ``look`` (default) or ``zones``; with ``zones`` the first source
zone is selected and the zone inspector is also written whole, as
``zone_inspector.png``.
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
    """Load the pair, wait for the preview, write ``ui_<method>.png``."""
    if len(argv) not in (3, 4):
        print(__doc__)
        return 2
    source, reference, out = argv[0], argv[1], Path(argv[2])
    method = argv[3] if len(argv) == 4 else "look"
    out.mkdir(parents=True, exist_ok=True)

    app = create_application([])
    load_stylesheet(app)
    window = MainWindow(settings=load_settings(None))
    window.resize(*WINDOW_SIZE)
    window.show()
    window._algo_combo.setCurrentIndex(window._algo_combo.findData(method))
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
    if method == "zones" and window._zone_model is not None:
        window._corr_view.select_zone(window._zone_model.source.ids[0], emit=True)
        pump(lambda: False, 0.3)

    preview = window._preview.image_rgb
    print("method:", window._current_method)
    print("preview:", None if preview is None else preview.shape)
    print("window:", window.width(), "x", window.height())
    window.grab().save(str(out / f"ui_{method}.png"))
    if method == "zones":
        # The inspector scrolls inside a short panel: grab it whole as well
        window._inspector.grab().save(str(out / "zone_inspector.png"))
    window.stop_workers()
    return 0 if preview is not None else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
