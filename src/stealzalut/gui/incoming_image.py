"""An image that arrives without being a file on disk: dragged from a web page, or pasted.

A browser does not hand over a file.  Depending on the browser and on the
gesture, a drop or a paste carries one or several of:

- the decoded picture itself (« Copy image », some drags);
- the address of the picture (``http://`` or ``https://``);
- a piece of HTML whose ``<img>`` holds that address, or the picture itself
  as a ``data:`` URL.

:func:`read_offer` copies what is useful out of the Qt mime data (which is
only valid during the event), and :func:`fetch` turns it into a local file
the normal loader can read: the engine reads files, with their colour
profile, and the original bytes are kept whenever they are available.

Fetching an address is the only moment the application uses the network,
and only for the address the user dropped or pasted.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import logging
import re
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlsplit

from PySide6.QtCore import QMimeData
from PySide6.QtGui import QImage

from stealzalut.utils.image_io import SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)

#: Largest picture accepted from the network or from a ``data:`` URL.
MAX_BYTES = 80 * 1024 * 1024
#: Seconds allowed to reach the server, then between two packets.
TIMEOUT_S = 12.0
_USER_AGENT = "Mozilla/5.0 (StealZaLut image fetch)"

_IMG_SRC = re.compile(r"<img\b[^>]*?\bsrc\s*=\s*([\"'])(.*?)\1", re.IGNORECASE | re.DOTALL)
_DATA_URL = re.compile(r"data:image/[\w.+-]+;base64,([A-Za-z0-9+/=\s]+)", re.IGNORECASE)
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


class IncomingImageError(Exception):
    """The offer held no usable picture; the message is shown to the user (French)."""


@dataclass(frozen=True, slots=True)
class Offer:
    """What a drop or a paste carries, copied out of the mime data.

    Attributes:
        local_file: A supported image file on disk.
        image: The decoded picture.
        data: The bytes of a picture embedded as a ``data:`` URL.
        url: The address of the picture on the web.
    """

    local_file: Path | None = None
    image: QImage | None = None
    data: bytes | None = None
    url: str | None = None

    def __bool__(self) -> bool:
        return any(part is not None for part in (self.local_file, self.image, self.data, self.url))


def _web_address(text: str) -> str | None:
    text = text.strip()
    return text if urlsplit(text).scheme in ("http", "https") and " " not in text else None


def read_offer(mime: QMimeData) -> Offer:
    """Copy out of *mime* whatever can become an image (an empty offer if nothing can)."""
    local, url, data, image = None, None, None, None
    for candidate in mime.urls() if mime.hasUrls() else []:
        if candidate.isLocalFile():
            path = Path(candidate.toLocalFile())
            if local is None and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                local = path
        elif url is None:
            url = _web_address(candidate.toString())
    if mime.hasHtml():
        html = mime.html()
        found = _IMG_SRC.search(html)
        source = found.group(2).replace("&amp;", "&") if found else ""
        embedded = _DATA_URL.match(source)
        if embedded:
            try:
                data = base64.b64decode(embedded.group(1), validate=False)
            except (binascii.Error, ValueError):
                data = None
        elif source:
            # The <img> is the picture; a plain URL next to it may be the page or a link
            url = _web_address(source) or url
    if url is None and mime.hasText():
        url = _web_address(mime.text())
    if mime.hasImage():
        candidate_image = mime.imageData()
        if isinstance(candidate_image, QImage) and not candidate_image.isNull():
            image = QImage(candidate_image)  # detached from the event's data
    return Offer(local, image, data, url)


def incoming_folder() -> Path:
    """Folder where fetched and pasted pictures are written (system temporary directory)."""
    return Path(tempfile.gettempdir()) / "stealzalut" / "incoming"


def sniff_extension(content: bytes) -> str | None:
    """File extension matching the first bytes of *content*, among the formats read here."""
    head = content[:16]
    if head.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return ".webp"
    if head[:4] in (b"II*\x00", b"MM\x00*"):
        return ".tif"
    if head[:2] == b"BM":
        return ".bmp"
    if head[4:8] == b"ftyp" and head[8:12] in (b"heic", b"heix", b"mif1", b"msf1", b"hevc"):
        return ".heic"
    return None


def _stem(name: str, content: bytes) -> str:
    """A safe, unique file stem: a readable part, then a few characters of the content's hash."""
    readable = _UNSAFE.sub("_", Path(name).stem).strip("._")[:60] or "image"
    return f"{readable}_{hashlib.sha1(content).hexdigest()[:8]}"


def _write(content: bytes, name: str, folder: Path) -> Path:
    extension = sniff_extension(content)
    if extension is None:
        raise IncomingImageError(
            "Ce format d'image n'est pas pris en charge (JPEG, PNG, WebP, TIFF, BMP ou HEIC)."
        )
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (_stem(name, content) + extension)
    path.write_bytes(content)
    return path


def download(url: str) -> bytes:
    """The bytes at *url* (``http`` or ``https`` only), at most :data:`MAX_BYTES`.

    Raises:
        IncomingImageError: Unreachable address, refusal, or too large.
    """
    if urlsplit(url).scheme not in ("http", "https"):
        raise IncomingImageError("Cette adresse n'est pas une adresse web.")
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT, "Accept": "image/*"})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
            content: bytes = response.read(MAX_BYTES + 1)
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        logger.info("Could not fetch %s: %s", url, exc)
        raise IncomingImageError(
            "Impossible de récupérer l'image à cette adresse (pas de connexion, ou le site "
            "refuse). Enregistre-la puis dépose le fichier."
        ) from exc
    if len(content) > MAX_BYTES:
        raise IncomingImageError("Cette image est trop volumineuse (plus de 80 Mo).")
    return content


def fetch(offer: Offer, folder: Path | None = None) -> Path:
    """A local file holding the picture of *offer*.

    In order: the file already on disk; the embedded bytes; the address,
    fetched (the original file, with its colour profile); and last the
    decoded picture, saved as PNG.  When the address cannot be fetched but a
    decoded picture came with it, that picture is used.

    Raises:
        IncomingImageError: Nothing in the offer gives a picture.
    """
    if offer.local_file is not None:
        return offer.local_file
    folder = incoming_folder() if folder is None else folder
    if offer.data is not None:
        return _write(offer.data, "image_collee", folder)
    failure: IncomingImageError | None = None
    if offer.url is not None:
        try:
            name = unquote(urlsplit(offer.url).path.rsplit("/", 1)[-1]) or "image_web"
            return _write(download(offer.url), name, folder)
        except IncomingImageError as exc:
            failure = exc
    if offer.image is not None:
        folder.mkdir(parents=True, exist_ok=True)
        size = f"{offer.image.width()}x{offer.image.height()}"
        raw = bytes(offer.image.constBits())
        path = folder / (_stem("image_collee", size.encode() + raw) + ".png")
        if offer.image.save(str(path)):
            return path
        failure = IncomingImageError("L'image n'a pas pu être enregistrée temporairement.")
    raise failure or IncomingImageError("Aucune image dans ce qui a été déposé ou collé.")
