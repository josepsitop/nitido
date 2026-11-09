#!/usr/bin/env python3
"""Convierte modelos .pth/.safetensors a ONNX con ejes dinámicos (alto/ancho variables).

Uso local:
    pip install torch spandrel onnx onnxruntime
    python tools/convert_to_onnx.py modelo.pth [salida.onnx] [--fp16]

Uso por lote (lo que hace el workflow de GitHub):
    python tools/convert_to_onnx.py --list tools/models_to_convert.json --out onnx_out
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch
from spandrel import ModelLoader


def convert(src, dst, fp16=False, opset=17):
    desc = ModelLoader().load_from_file(str(src))
    device, dtype = "cpu", torch.float32
    if fp16:
        if not torch.cuda.is_available():
            raise RuntimeError("--fp16 necesita una GPU con CUDA")
        device, dtype = "cuda", torch.float16
    model = desc.model.eval().to(device=device, dtype=dtype)
    dummy = torch.rand(1, desc.input_channels, 64, 64, device=device, dtype=dtype)
    kw = dict(opset_version=opset, input_names=["input"], output_names=["output"],
              dynamic_axes={"input": {0: "batch", 2: "height", 3: "width"},
                            "output": {0: "batch", 2: "out_height", 3: "out_width"}})
    with torch.no_grad():
        try:
            torch.onnx.export(model, dummy, str(dst), dynamo=False, **kw)
        except TypeError:  # versiones de PyTorch sin el argumento dynamo
            torch.onnx.export(model, dummy, str(dst), **kw)
    return desc.scale


def verify(path, scale):
    """Comprueba que el ONNX acepta tamaños distintos (ejes dinámicos) y que la escala coincide."""
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    dt = np.float16 if "float16" in inp.type else np.float32
    for h, w in ((64, 64), (80, 96)):
        out = sess.run(None, {inp.name: np.random.rand(1, 3, h, w).astype(dt)})[0]
        if out.shape[2] != h * scale or out.shape[3] != w * scale:
            raise RuntimeError(f"forma inesperada {out.shape} para entrada {h}x{w} (escala {scale})")


def fetch(url, dst):
    req = urllib.request.Request(url, headers={"User-Agent": "Nitido-convert"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dst, "wb") as f:
        f.write(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?")
    ap.add_argument("output", nargs="?")
    ap.add_argument("--list")
    ap.add_argument("--out", default="onnx_out")
    ap.add_argument("--fp16", action="store_true")
    a = ap.parse_args()

    if a.list:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        tmp = out / "_tmp"
        tmp.mkdir(exist_ok=True)
        ok, bad, notes = 0, 0, []
        for e in json.loads(Path(a.list).read_text(encoding="utf-8")):
            dst = out / e["file"]
            try:
                src = tmp / Path(e["url"]).name
                print(f"↓ {e['url']}", flush=True)
                fetch(e["url"], src)
                scale = convert(src, dst)
                verify(dst, scale)
                notes.append(f"{e['file']}\t{scale}x\t{e['license']}\t{e['use']}\t{e['url']}")
                ok += 1
                print(f"✔ {dst.name} ({scale}x)", flush=True)
            except Exception as ex:
                bad += 1
                dst.unlink(missing_ok=True)
                print(f"✖ {e['file']}: {ex}", flush=True)
        for f in tmp.glob("*"):
            f.unlink()
        tmp.rmdir()
        (out / "LEEME.txt").write_text(
            "Modelos convertidos con Nítido (archivo, escala, licencia, uso, origen):\n" + "\n".join(notes) + "\n",
            encoding="utf-8")
        print(f"Convertidos: {ok} · fallidos: {bad}")
        sys.exit(0 if ok else 1)

    if not a.model:
        ap.error("indica un modelo o usa --list")
    src = Path(a.model)
    dst = Path(a.output) if a.output else src.with_suffix(".onnx")
    scale = convert(src, dst, a.fp16)
    verify(dst, scale)
    print(f"✔ {dst} ({scale}x). Cópialo a la carpeta models/ de Nítido.")


if __name__ == "__main__":
    main()
