"""Nítido · núcleo: presets, ffmpeg, motor ONNX y pipeline de render."""
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

DEFAULT = {
    "name": "Nuevo", "model": "", "provider": "auto", "tile": 0,
    "prefilter": "", "max_height": 0,
    "encoder": "libx265", "crf": 20, "speed": "slow", "ten_bit": True,
    "container": "mkv", "audio": "copy", "image_format": "png",
}

SPEEDS = {
    "libx265": {"fast": "fast", "balanced": "medium", "slow": "slow"},
    "libx264": {"fast": "fast", "balanced": "medium", "slow": "slow"},
    "hevc_nvenc": {"fast": "p3", "balanced": "p5", "slow": "p7"},
    "libsvtav1": {"fast": "10", "balanced": "7", "slow": "5"},
}


class Cancelled(Exception):
    pass


# ---------------------------------------------------------------- rutas / presets
def base_dir():
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def _sub(name):
    d = base_dir() / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def presets_dir():
    return _sub("presets")


def models_dir():
    return _sub("models")


def list_models():
    return sorted(f.name for f in models_dir().glob("*.onnx"))


def list_presets():
    out = {}
    for f in sorted(presets_dir().glob("*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            out[d.get("name", f.stem)] = {**DEFAULT, **d}
        except Exception:
            pass
    return out


def delete_preset(name):
    for f in presets_dir().glob("*.json"):
        try:
            if json.loads(f.read_text(encoding="utf-8")).get("name") == name:
                f.unlink()
        except Exception:
            pass


def save_preset(p):
    name = p["name"].strip() or "Preset"
    delete_preset(name)
    slug = "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_") or "preset"
    data = {**p, "name": name}
    (presets_dir() / f"{slug}.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def find_tool(name):
    exe = name + (".exe" if os.name == "nt" else "")
    for d in (base_dir() / "bin", base_dir()):
        if (d / exe).exists():
            return str(d / exe)
    return shutil.which(name)


def output_path(src, outdir, p):
    src = Path(src)
    d = Path(outdir) if outdir else src.parent
    d.mkdir(parents=True, exist_ok=True)
    is_img = src.suffix.lower() in IMAGE_EXTS
    ext = "." + (p.get("image_format", "png") if is_img else p.get("container", "mkv"))
    out, i = d / f"{src.stem}_nitido{ext}", 1
    while out.exists():
        out = d / f"{src.stem}_nitido_{i}{ext}"
        i += 1
    return out


# ---------------------------------------------------------------- ffmpeg
def _kw():
    return {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}


def _num(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _frac(x):
    try:
        a, b = str(x).split("/")
        return float(a) / float(b)
    except Exception:
        return 0.0


def probe(path):
    r = subprocess.run(
        [find_tool("ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration,color_space:format=duration",
         "-of", "json", str(path)],
        capture_output=True, text=True, **_kw())
    if r.returncode != 0 or not json.loads(r.stdout or "{}").get("streams"):
        raise RuntimeError(f"ffprobe no pudo leer el archivo:\n{r.stderr.strip()}")
    j = json.loads(r.stdout)
    s = j["streams"][0]
    fs = s.get("avg_frame_rate", "0/0")
    if _frac(fs) <= 0:
        fs = s.get("r_frame_rate", "25/1")
    if _frac(fs) <= 0:
        fs = "25/1"
    fps = _frac(fs)
    dur = _num(s.get("duration")) or _num(j.get("format", {}).get("duration"))
    nb = str(s.get("nb_frames", ""))
    total = int(nb) if nb.isdigit() else int(dur * fps)
    return {"w": int(s["width"]), "h": int(s["height"]), "fps": fps, "fps_str": fs,
            "total": max(total, 1), "cs": s.get("color_space", "")}


def src_matrix(info):
    m = {"bt709": "bt709", "smpte170m": "bt601", "bt470bg": "bt601", "bt2020nc": "bt2020"}.get(info["cs"])
    return m or ("bt601" if info["h"] < 720 else "bt709")


def encoder_args(p):
    e = p["encoder"]
    if e not in SPEEDS:
        raise RuntimeError(f"Encoder desconocido: {e}")
    sp = SPEEDS[e].get(p.get("speed", "balanced"), SPEEDS[e]["balanced"])
    crf = str(p["crf"])
    ten = bool(p.get("ten_bit")) and e != "libx264"
    if e == "libx265":
        a = ["-c:v", "libx265", "-preset", sp, "-crf", crf, "-x265-params", "log-level=error"]
    elif e == "libx264":
        a = ["-c:v", "libx264", "-preset", sp, "-crf", crf]
    elif e == "hevc_nvenc":
        a = ["-c:v", "hevc_nvenc", "-preset", sp, "-rc", "vbr", "-cq", crf, "-b:v", "0"]
        if ten:
            a += ["-profile:v", "main10"]
    else:
        a = ["-c:v", "libsvtav1", "-preset", sp, "-crf", crf]
    if p.get("container") == "mp4" and e in ("libx265", "hevc_nvenc"):
        a += ["-tag:v", "hvc1"]
    pix = "p010le" if (e == "hevc_nvenc" and ten) else ("yuv420p10le" if ten else "yuv420p")
    return a, pix


def reader_cmd(ff, src, info, p, test_s):
    chain = []
    if (p.get("prefilter") or "").strip():
        chain.append(p["prefilter"].strip())
    chain += [f"scale=in_color_matrix={src_matrix(info)}", "format=rgb24"]
    cmd = [ff, "-hide_banner", "-loglevel", "error"]
    if test_s:
        cmd += ["-t", str(test_s)]
    cmd += ["-i", str(src), "-map", "0:v:0", "-vf", ",".join(chain), "-r", info["fps_str"],
            "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    return cmd


def writer_cmd(ff, src, dst, info, p, ow, oh, is_img, test_s):
    cmd = [ff, "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{ow}x{oh}", "-framerate", info["fps_str"], "-i", "pipe:0"]
    if is_img:
        extra = {"png": [], "jpg": ["-q:v", "2"], "webp": ["-quality", "92"]}[p.get("image_format", "png")]
        return cmd + ["-frames:v", "1"] + extra + [str(dst)]
    mp4 = p.get("container") == "mp4"
    enc, pix = encoder_args(p)
    cmd += ["-i", str(src), "-map", "0:v:0", "-map", "1:a?"]
    if not mp4:
        cmd += ["-map", "1:s?", "-map", "1:t?"]
    cmd += ["-map_chapters", "1"]
    cmd += ["-c:a", "copy"] if p.get("audio", "copy") == "copy" else ["-c:a", "aac", "-b:a", "192k"]
    if not mp4:
        cmd += ["-c:s", "copy"]
    mh = int(p.get("max_height") or 0)
    mh -= mh % 2
    if mh and oh > mh:
        sc = f"scale=-2:{mh}:flags=lanczos:"
    else:
        sc = "scale=trunc(iw/2)*2:trunc(ih/2)*2:"
    cmd += ["-vf", f"{sc}out_color_matrix=bt709:out_range=tv,format={pix}"] + enc
    cmd += ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv"]
    if test_s:
        cmd += ["-t", str(test_s)]
    return cmd + [str(dst)]


# ---------------------------------------------------------------- motor ONNX
class Engine:
    def __init__(self, model_path, provider="auto", log=lambda s: None):
        import onnxruntime as ort

        av = ort.get_available_providers()
        cache = _sub("cache") / "trt"
        names = {"tensorrt": "TensorrtExecutionProvider", "cuda": "CUDAExecutionProvider",
                 "directml": "DmlExecutionProvider"}
        table = {"tensorrt": ("TensorrtExecutionProvider", {
                    "trt_fp16_enable": True, "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": str(cache)}),
                 "cuda": "CUDAExecutionProvider", "directml": "DmlExecutionProvider"}
        prefs = {"auto": ["tensorrt", "cuda", "directml"], "tensorrt": ["tensorrt", "cuda"],
                 "cuda": ["cuda"], "directml": ["directml"], "cpu": []}[provider]
        provs = [table[k] for k in prefs if names[k] in av] + ["CPUExecutionProvider"]
        so = ort.SessionOptions()
        so.log_severity_level = 3
        if any("Dml" in (x if isinstance(x, str) else x[0]) for x in provs):
            so.enable_mem_pattern = False
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        log(f"Cargando modelo {Path(model_path).name} (puede tardar la primera vez)…")
        self.sess = ort.InferenceSession(str(model_path), so, providers=provs)
        log(f"Backend activo: {self.sess.get_providers()[0]}")
        inp = self.sess.get_inputs()[0]
        self.name = inp.name
        self.dtype = np.float16 if "float16" in inp.type else np.float32
        h, w = inp.shape[2], inp.shape[3]
        fixed_h, fixed_w = isinstance(h, int) and h > 0, isinstance(w, int) and w > 0
        if fixed_h != fixed_w:
            raise RuntimeError("Modelo ONNX con una sola dimensión fija: no soportado.")
        self.fixed = (h, w) if fixed_h else None
        ph, pw = self.fixed or (64, 64)
        if self.fixed:
            log(f"Modelo de tamaño fijo ({w}x{h}): se procesa por mosaicos y será más lento. "
                "Si puedes, conviértelo con ejes dinámicos (ver MODELOS.md).")
        out = self.sess.run(None, {self.name: np.zeros((1, 3, ph, pw), self.dtype)})[0]
        self.scale = out.shape[3] // pw
        log(f"Escala del modelo: {self.scale}x")

    def _run(self, img):
        x = np.ascontiguousarray(img.transpose(2, 0, 1)[None], dtype=self.dtype)
        x *= 1.0 / 255.0
        y = self.sess.run(None, {self.name: x})[0][0].astype(np.float32, copy=False)
        y = np.clip(y * 255.0 + 0.5, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(y.transpose(1, 2, 0))

    def _window(self, img, ya, xa, fh, fw):
        h, w = img.shape[:2]
        y0, x0, y1, x1 = max(ya, 0), max(xa, 0), min(ya + fh, h), min(xa + fw, w)
        return np.pad(img[y0:y1, x0:x1], ((y0 - ya, ya + fh - y1), (x0 - xa, xa + fw - x1), (0, 0)), mode="edge")

    def _upscale_fixed(self, img, pad=8):
        fh, fw = self.fixed
        s, (h, w, _) = self.scale, img.shape
        sh, sw = max(fh - 2 * pad, 1), max(fw - 2 * pad, 1)
        out = np.empty((h * s, w * s, 3), np.uint8)
        for y0 in range(0, h, sh):
            for x0 in range(0, w, sw):
                y1, x1 = min(y0 + sh, h), min(x0 + sw, w)
                t = self._run(self._window(img, y0 - pad, x0 - pad, fh, fw))
                out[y0 * s:y1 * s, x0 * s:x1 * s] = t[pad * s:(pad + y1 - y0) * s, pad * s:(pad + x1 - x0) * s]
        return out

    def upscale(self, img, tile=0, pad=16):
        if self.fixed:
            return self._upscale_fixed(img)
        h, w, _ = img.shape
        if tile <= 0 or (h <= tile and w <= tile):
            return self._run(img)
        s = self.scale
        out = np.empty((h * s, w * s, 3), np.uint8)
        for y0 in range(0, h, tile):
            for x0 in range(0, w, tile):
                y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
                ya, yb = max(y0 - pad, 0), min(y1 + pad, h)
                xa, xb = max(x0 - pad, 0), min(x1 + pad, w)
                t = self._run(img[ya:yb, xa:xb])
                out[y0 * s:y1 * s, x0 * s:x1 * s] = t[(y0 - ya) * s:(y1 - ya) * s, (x0 - xa) * s:(x1 - xa) * s]
        return out


_ENGINES = {}


def get_engine(p, log):
    name = p.get("model") or (list_models() or [""])[0]
    if not name:
        raise RuntimeError("No hay modelos .onnx en la carpeta models/. Mira models/README.md.")
    path = models_dir() / name
    if not path.exists():
        raise RuntimeError(f"No existe el modelo: {name}")
    key = (name, p.get("provider", "auto"))
    if key not in _ENGINES:
        _ENGINES.clear()
        _ENGINES[key] = Engine(path, p.get("provider", "auto"), log)
    return _ENGINES[key]


# ---------------------------------------------------------------- pipeline
def _drain(proc, buf):
    for line in iter(proc.stderr.readline, b""):
        buf.append(line.decode(errors="replace"))


def run_job(src, dst, p, progress, log, cancel, test_s=0):
    ff = find_tool("ffmpeg")
    if not ff or not find_tool("ffprobe"):
        raise RuntimeError("No encuentro ffmpeg/ffprobe (ponlos en la carpeta bin/ o en el PATH).")
    is_img = Path(src).suffix.lower() in IMAGE_EXTS
    info = probe(src)
    eng = get_engine(p, log)
    ow, oh = info["w"] * eng.scale, info["h"] * eng.scale
    total = 1 if is_img else info["total"]
    if test_s and not is_img:
        total = min(total, max(1, int(test_s * info["fps"])))
    fsz = info["w"] * info["h"] * 3
    tile = int(p.get("tile") or 0)

    rd = subprocess.Popen(reader_cmd(ff, src, info, p, 0 if is_img else test_s),
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, **_kw())
    wr = subprocess.Popen(writer_cmd(ff, src, dst, info, p, ow, oh, is_img, test_s),
                          stdin=subprocess.PIPE, stderr=subprocess.PIPE, **_kw())
    errs = {"r": [], "w": []}
    for proc, k in ((rd, "r"), (wr, "w")):
        threading.Thread(target=_drain, args=(proc, errs[k]), daemon=True).start()

    stop = threading.Event()
    rq, wq = queue.Queue(maxsize=6), queue.Queue(maxsize=6)
    werr = []

    def put(q, item):
        while not stop.is_set():
            try:
                q.put(item, timeout=0.5)
                return True
            except queue.Full:
                pass
        return False

    def reader():
        try:
            while not stop.is_set():
                buf = rd.stdout.read(fsz)
                if len(buf) < fsz or not put(rq, buf):
                    break
        finally:
            put(rq, None)

    def writer():
        try:
            while True:
                item = wq.get()
                if item is None:
                    break
                wr.stdin.write(item)
        except Exception as e:  # tubería rota, etc.
            werr.append(str(e))
            stop.set()
        finally:
            try:
                wr.stdin.close()
            except Exception:
                pass

    tr = threading.Thread(target=reader, daemon=True)
    tw = threading.Thread(target=writer, daemon=True)
    tr.start()
    tw.start()
    done, t0 = 0, time.time()
    try:
        while True:
            if cancel.is_set():
                raise Cancelled()
            if werr:
                raise RuntimeError("Falló el encoder:\n" + "".join(errs["w"][-30:]))
            try:
                buf = rq.get(timeout=0.5)
            except queue.Empty:
                continue
            if buf is None:
                break
            frame = np.frombuffer(buf, np.uint8).reshape(info["h"], info["w"], 3)
            if not put(wq, eng.upscale(frame, tile).tobytes()):
                raise RuntimeError("Falló el encoder:\n" + "".join(errs["w"][-30:]))
            done += 1
            if done == 1 or done % 3 == 0:
                progress(done, total, done / max(time.time() - t0, 1e-6))
        put(wq, None)
        tw.join()
        rd.wait()
        wr.wait()
        if done == 0 or rd.returncode != 0:
            raise RuntimeError("No se pudieron leer frames del origen:\n" + "".join(errs["r"][-30:]))
        if wr.returncode != 0 or werr:
            raise RuntimeError("ffmpeg falló al codificar:\n" + "".join(errs["w"][-30:]))
        progress(total, total, done / max(time.time() - t0, 1e-6))
    except BaseException:
        stop.set()
        for pr in (rd, wr):
            try:
                pr.kill()
            except Exception:
                pass
        Path(dst).unlink(missing_ok=True)
        raise
