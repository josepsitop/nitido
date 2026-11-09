"""Nítido · catálogo, descarga e importación de modelos ONNX."""
import hashlib
import json
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import urlparse

from . import core


def load_catalog():
    """Lee models/catalog.json (puedes editarlo y añadir tus propias entradas)."""
    f = core.models_dir() / "catalog.json"
    if not f.exists():
        return []
    return json.loads(f.read_text(encoding="utf-8"))


def import_file(path):
    """Copia un .onnx local a la carpeta models/."""
    src = Path(path)
    if src.suffix.lower() != ".onnx":
        raise RuntimeError("Solo se pueden importar archivos .onnx (los .pth hay que convertirlos, ver MODELOS.md).")
    dst = core.models_dir() / src.name
    if src.resolve() != dst.resolve():
        shutil.copyfile(src, dst)
    return dst.name


def download(entry, progress=lambda done, total: None, cancel=None):
    """Descarga una entrada del catálogo. Si es .zip, extrae todos los .onnx. Devuelve los nombres."""
    url = entry["url"]
    tmp = Path(tempfile.mkdtemp(prefix="nitido_"))
    try:
        part = tmp / "download.bin"
        req = urllib.request.Request(url, headers={"User-Agent": "Nitido/0.2"})
        sha = hashlib.sha256()
        with urllib.request.urlopen(req, timeout=60) as r, open(part, "wb") as f:
            total, done = int(r.headers.get("Content-Length") or 0), 0
            while True:
                if cancel is not None and cancel.is_set():
                    raise core.Cancelled()
                chunk = r.read(1 << 16)
                if not chunk:
                    break
                f.write(chunk)
                sha.update(chunk)
                done += len(chunk)
                progress(done, total)
        want = (entry.get("sha256") or "").lower()
        if want and sha.hexdigest() != want:
            raise RuntimeError("El archivo descargado no coincide con el sha256 del catálogo.")
        models = core.models_dir()
        if urlparse(url).path.lower().endswith(".zip"):
            names = []
            with zipfile.ZipFile(part) as z:
                for member in z.namelist():
                    if member.lower().endswith(".onnx"):
                        target = models / Path(member).name
                        with z.open(member) as src, open(target, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                        names.append(target.name)
            if not names:
                raise RuntimeError("El zip no contiene archivos .onnx.")
            return names
        name = entry.get("file") or Path(urlparse(url).path).name
        shutil.move(str(part), str(models / name))
        return [name]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
