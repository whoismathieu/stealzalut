import sys
import time
from pathlib import Path

S = Path(sys.argv[1])
P = Path(r"C:\Users\mathi\Pictures")
from stealzalut.config import load_settings
from stealzalut.gui.main_window import MainWindow
from stealzalut.main import create_application, load_stylesheet

app = create_application([])
load_stylesheet(app)
w = MainWindow(settings=load_settings(None))
w.show()
w._algo_combo.setCurrentIndex(w._algo_combo.findData("zones"))
w._source_zone.load_from_path(P / "IMG_6295.DNG")
w._reference_zone.load_from_path(P / "841121141_18631647988026044_4768591964563047386_n.jpg")


def pump(cond, timeout=60.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end and not cond():
        app.processEvents()
        time.sleep(0.01)


def idle():
    return w._preview.image_rgb is not None and not (
        w._transfer_worker and w._transfer_worker.isRunning()
    )


pump(lambda: w._zone_model is not None and idle())
pump(lambda: False, 0.3)
print("hint visible:", not w._mode_hint.isHidden(), "|", w._mode_hint_label.text()[:90])
w.grab().save(str(S / "ui_hint.png"))
w._mode_hint_btn.click()
pump(lambda: False, 0.3)
pump(idle)
pump(lambda: False, 0.3)
print("method:", w._current_method, "hint visible:", not w._mode_hint.isHidden())
w.grab().save(str(S / "ui_ambiance.png"))
w.stop_workers()
