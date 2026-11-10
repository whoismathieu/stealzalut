"""Drive the real window with the look method on case A and grab it."""

import sys
import time
from pathlib import Path

import harness as h
from stealzalut.config import load_settings
from stealzalut.gui.main_window import MainWindow
from stealzalut.main import create_application, load_stylesheet

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
app = create_application([])
load_stylesheet(app)
w = MainWindow(settings=load_settings(None))
w.show()
w._algo_combo.setCurrentIndex(w._algo_combo.findData("look"))
w._source_zone.load_from_path(h.SOURCE)
w._reference_zone.load_from_path(h.REF_A)


def pump(cond, timeout=60.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end and not cond():
        app.processEvents()
        time.sleep(0.01)


def idle():
    return w._preview.image_rgb is not None and not (
        w._transfer_worker and w._transfer_worker.isRunning()
    )


pump(idle)
pump(lambda: False, 0.5)
pump(idle)
print("method:", w._current_method, "| preview:", w._preview.image_rgb.shape)
print("zones panel visible:", w._corr_group.isVisible())
w.grab().save(str(out / "ui_look.png"))
w.stop_workers()
