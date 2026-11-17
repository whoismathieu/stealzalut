"""Pictures dragged from a web page or pasted: no file on disk to start from."""

from __future__ import annotations

import base64
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
from PySide6.QtCore import QMimeData, QPoint, QUrl
from PySide6.QtGui import QGuiApplication, QImage
from PySide6.QtWidgets import QApplication

from stealzalut.config import Settings
from stealzalut.gui import incoming_image as inc
from stealzalut.gui.main_window import MainWindow
from stealzalut.gui.widgets.image_drop_zone import ImageDropZone
from stealzalut.main import create_application


@pytest.fixture
def qapp() -> QApplication:
    return create_application([])


@pytest.fixture(autouse=True)
def _own_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Nothing is written in the real temporary folder of the application."""
    folder = tmp_path / "incoming"
    monkeypatch.setattr(inc, "incoming_folder", lambda: folder)
    return folder


def _encoded(extension: str = ".jpg", colour: tuple[int, int, int] = (40, 90, 200)) -> bytes:
    picture = np.full((24, 32, 3), colour, dtype=np.uint8)
    ok, data = cv2.imencode(extension, picture)
    assert ok
    return bytes(data.tobytes())


def _response(content: bytes) -> MagicMock:
    response = MagicMock()
    response.read.side_effect = lambda limit=-1: content if limit < 0 else content[:limit]
    response.__enter__.return_value = response
    response.__exit__.return_value = None
    return response


def _web_drag(url: str, html: str | None = None) -> QMimeData:
    mime = QMimeData()
    mime.setUrls([QUrl(url)])
    mime.setText(url)
    mime.setHtml(html if html is not None else f'<img src="{url}" alt="x">')
    return mime


class TestOffer:
    def test_nothing_usable_is_an_empty_offer(self) -> None:
        mime = QMimeData()
        mime.setText("juste du texte")
        assert not inc.read_offer(mime)
        assert not inc.read_offer(QMimeData())

    def test_a_local_image_file_is_taken_as_it_is(self, tmp_path: Path) -> None:
        path = tmp_path / "photo.JPG"
        path.write_bytes(_encoded())
        note = tmp_path / "notes.txt"
        note.write_text("x")
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(str(note)), QUrl.fromLocalFile(str(path))])
        offer = inc.read_offer(mime)
        assert offer.local_file == path
        assert inc.fetch(offer) == path

    def test_the_address_of_the_img_wins_over_the_link_around_it(self) -> None:
        mime = _web_drag(
            "https://example.org/page.html",
            '<a href="https://example.org/page.html"><img alt="" '
            "src='https://cdn.example.org/a/photo%20one.jpg?w=1200&amp;q=80'></a>",
        )
        assert inc.read_offer(mime).url == "https://cdn.example.org/a/photo%20one.jpg?w=1200&q=80"

    def test_only_web_addresses_are_kept(self) -> None:
        for text in ("ftp://example.org/a.jpg", "file:///etc/passwd", "javascript:alert(1)"):
            mime = QMimeData()
            mime.setText(text)
            assert inc.read_offer(mime).url is None

    @pytest.mark.parametrize("extension", [".jpg", ".png", ".webp", ".bmp", ".tif"])
    def test_formats_are_recognised_from_their_first_bytes(self, extension: str) -> None:
        assert inc.sniff_extension(_encoded(extension)) == extension
        assert inc.sniff_extension(b"<html>not an image</html>") is None


class TestFetch:
    def test_an_address_is_downloaded_with_its_original_bytes(self, _own_folder: Path) -> None:
        content = _encoded(".jpg")
        offer = inc.read_offer(_web_drag("https://example.org/img/Tokyo%20night.jpg?x=1"))
        with patch("urllib.request.urlopen", return_value=_response(content)) as opened:
            path = inc.fetch(offer)
        request = opened.call_args.args[0]
        assert request.full_url == "https://example.org/img/Tokyo%20night.jpg?x=1"
        assert opened.call_args.kwargs["timeout"] == inc.TIMEOUT_S
        assert path.parent == _own_folder and path.suffix == ".jpg"
        assert path.name.startswith("Tokyo_night_")
        assert path.read_bytes() == content

    def test_the_extension_comes_from_the_content_not_from_the_address(self) -> None:
        offer = inc.Offer(url="https://example.org/image.php?id=3")
        with patch("urllib.request.urlopen", return_value=_response(_encoded(".png"))):
            assert inc.fetch(offer).suffix == ".png"

    def test_a_page_instead_of_a_picture_is_refused(self) -> None:
        offer = inc.Offer(url="https://example.org/photo")
        with (
            patch("urllib.request.urlopen", return_value=_response(b"<html>login</html>")),
            pytest.raises(inc.IncomingImageError, match="pas pris en charge"),
        ):
            inc.fetch(offer)

    def test_no_network_says_so_and_falls_back_on_the_decoded_picture(self) -> None:
        offer = inc.Offer(url="https://example.org/a.jpg")
        failing = patch("urllib.request.urlopen", side_effect=urllib.error.URLError("offline"))
        with failing, pytest.raises(inc.IncomingImageError, match="Impossible de récupérer"):
            inc.fetch(offer)
        image = QImage(20, 10, QImage.Format.Format_RGB32)
        image.fill(0x336699)
        with failing:
            path = inc.fetch(inc.Offer(url="https://example.org/a.jpg", image=image))
        assert path.suffix == ".png" and QImage(str(path)).size() == image.size()

    def test_too_large_a_download_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(inc, "MAX_BYTES", 100)
        with (
            patch("urllib.request.urlopen", return_value=_response(_encoded(".bmp"))),
            pytest.raises(inc.IncomingImageError, match="trop volumineuse"),
        ):
            inc.fetch(inc.Offer(url="https://example.org/a.bmp"))

    def test_a_picture_embedded_in_the_html_needs_no_network(self) -> None:
        content = _encoded(".png")
        mime = QMimeData()
        mime.setHtml(f'<img src="data:image/png;base64,{base64.b64encode(content).decode()}">')
        offer = inc.read_offer(mime)
        assert offer.data == content and offer.url is None
        with patch("urllib.request.urlopen", side_effect=AssertionError("no network")):
            assert inc.fetch(offer).read_bytes() == content


class TestDropZone:
    def test_a_picture_dragged_from_a_web_page_is_loaded(self, qapp: QApplication) -> None:
        zone = ImageDropZone()
        failures: list[str] = []
        zone.load_failed.connect(failures.append)
        with patch("urllib.request.urlopen", return_value=_response(_encoded(".jpg"))):
            assert zone.load_mime(_web_drag("https://example.org/a.jpg"))
        assert zone.has_image and zone.image_rgb.shape == (24, 32, 3)
        assert zone.image_path is not None and zone.image_path.suffix == ".jpg"
        assert failures == []
        # The temporary file goes away with the window, a file of the user never does
        fetched = zone.image_path
        zone.discard_fetched()
        assert not fetched.exists()

    def test_a_dropped_local_file_is_never_deleted(
        self, qapp: QApplication, tmp_path: Path
    ) -> None:
        path = tmp_path / "mine.png"
        path.write_bytes(_encoded(".png"))
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(str(path))])
        zone = ImageDropZone()
        assert zone.load_mime(mime) and zone.image_path == path
        zone.discard_fetched()
        assert path.exists()

    def test_a_failure_is_reported_and_leaves_the_zone_as_it_was(self, qapp: QApplication) -> None:
        zone = ImageDropZone()
        failures: list[str] = []
        zone.load_failed.connect(failures.append)
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("offline")):
            assert not zone.load_mime(_web_drag("https://example.org/a.jpg"))
        assert not zone.load_mime(QMimeData())
        assert not zone.has_image
        assert len(failures) == 2 and "Impossible de récupérer" in failures[0]
        assert QApplication.overrideCursor() is None

    def test_a_copied_picture_is_pasted(self, qapp: QApplication) -> None:
        image = QImage(30, 20, QImage.Format.Format_RGB32)
        image.fill(0xCC3311)
        QGuiApplication.clipboard().setImage(image)
        zone = ImageDropZone()
        assert zone.paste_from_clipboard()
        assert zone.image_rgb.shape == (20, 30, 3)
        r, g, b = (int(c) for c in zone.image_rgb[5, 5])
        assert (r, g, b) == (0xCC, 0x33, 0x11)
        QGuiApplication.clipboard().clear()

    def test_the_menu_of_the_zone_offers_to_paste(self, qapp: QApplication) -> None:
        zone = ImageDropZone()
        assert zone.view.context_menu().actions()[0].text().startswith("Coller une image")


class TestPasteInTheWindow:
    def test_ctrl_v_fills_the_first_empty_zone(self, qapp: QApplication, tmp_path: Path) -> None:
        window = MainWindow(settings=Settings(lut_output_dir=tmp_path))
        messages: list[str] = []
        try:
            window._status_bar.messageChanged.connect(messages.append)
            image = QImage(30, 20, QImage.Format.Format_RGB32)
            image.fill(0x204080)
            with (
                patch.object(window, "_run_transfer_preview"),
                # The pointer is away from both zones
                patch("stealzalut.gui.main_window.QCursor.pos", return_value=QPoint(-9000, -9000)),
            ):
                QGuiApplication.clipboard().setImage(image)
                window.paste_image()
                assert window._source_zone.has_image and not window._reference_zone.has_image
                window.paste_image()
                assert window._reference_zone.has_image
                QGuiApplication.clipboard().setText("rien d'utile")
                window.paste_image()
            assert any("Aucune image" in message for message in messages)
        finally:
            QGuiApplication.clipboard().clear()
            window.stop_workers()
