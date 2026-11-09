# Nítido

App personal para **reescalar y limpiar video y fotos con IA** en Windows. Usa modelos ONNX
(Real-ESRGAN compact, SPAN, etc.), ffmpeg para decodificar/codificar y presets editables.

## Compilar en GitHub (sin instalar nada)

1. Crea un repositorio y sube todo este contenido.
2. Ve a **Actions → Build Windows → Run workflow**.
3. Cuando termine, descarga el artefacto **Nitido-win64** y descomprímelo.
4. Para publicar un release: `git tag v0.1.0 && git push --tags`.

## Uso

1. Consigue modelos: botón **Descargar modelos…** en la app, o mira **MODELOS.md** (enlaces, licencias, qué elegir
   y cómo convertir los `.pth` con el workflow *Convertir modelos a ONNX*).
2. Abre `Nitido.exe`, arrastra tus videos/fotos, elige un preset y pulsa **Procesar**.
3. Usa **Probar solo los primeros N s** para comparar modelos y presets antes de renderizar una película entera.

La app es portable: `presets/`, `models/` y `bin/` (ffmpeg) están junto al `.exe`.

## Presets

Son archivos JSON en `presets/`; se editan en la interfaz y se guardan con **Guardar como…**.

| Campo | Qué hace |
|---|---|
| `model` | Archivo `.onnx` en `models/` (vacío = el primero) |
| `provider` | `auto`, `tensorrt`, `cuda`, `directml`, `cpu` |
| `tile` | 0 = frame completo; 256–512 si falta VRAM |
| `prefilter` | Filtro ffmpeg antes de la IA (`hqdn3d=…`, `deblock`, etc.) |
| `max_height` | Reduce la salida a esta altura tras el escalado (controla el peso) |
| `encoder` / `crf` / `speed` / `ten_bit` | Codec y calidad (`libx265`, `hevc_nvenc`, `libsvtav1`, `libx264`) |
| `container` / `audio` | `mkv` (conserva subtítulos, fuentes y capítulos) o `mp4`; audio `copy` o `aac` |

Puntos de partida para el peso: x265 CRF 18–22, NVENC cq 22–26, AV1 CRF 26–32. Ajusta con un clip de prueba.

## Backends

- El `.exe` trae **DirectML** (funciona en cualquier GPU, incluida tu RTX).
- Para **TensorRT/CUDA** (más rápido en NVIDIA) ejecuta desde código fuente con `onnxruntime-gpu`
  y las librerías CUDA/TensorRT instaladas: `pip install PySide6 numpy onnxruntime-gpu` y `python run.py`.
  La primera ejecución con TensorRT construye un engine y se cachea en `cache/`.

## Limitaciones actuales

- Modelos de tamaño fijo funcionan por mosaicos (lento); lo ideal son ONNX con ejes dinámicos.
- Un solo modelo por trabajo (aún no se pueden encadenar). Sin restauración de caras.
- Sin HDR/Dolby Vision (convierte a SDR antes) ni video con fps variable (se fuerza fps constante).
- Aún sin interpolación de frames (RIFE) ni backend VapourSynth.

## Desarrollo

```
pip install -r requirements.txt
python run.py
```

## Hoja de ruta

- Android (APK universal arm64-v8a + armeabi-v7a con ncnn) — fase 2.
- Interpolación RIFE, cola persistente, vista previa lado a lado.
