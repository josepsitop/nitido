# Modelos para Nítido

Nítido usa modelos **ONNX** (entrada RGB 0–1, salida RGB). Hay tres formas de tenerlos en la carpeta `models/`:

1. **Descargar modelos…** (botón en la app): baja los que están en `models/catalog.json` (solo ONNX ya listos).
2. **Convertir con GitHub** (para los `.pth`): *Actions → Convertir modelos a ONNX → Run workflow*, descarga el
   artefacto **modelos-onnx**, descomprímelo y copia los `.onnx` a `models/` (o usa **Importar .onnx…**).
3. **A mano**: baja cualquier modelo de [OpenModelDB](https://openmodeldb.info), conviértelo (abajo) y pulsa **Importar .onnx…**.

La escala (1x/2x/4x) se detecta sola. Un modelo 1x limpia sin cambiar el tamaño.

## ONNX listos (botón «Descargar modelos…»)

| Modelo | Para qué | Escala | Licencia | Fuente |
|---|---|---|---|---|
| 2xLiveActionV1 SPAN (1.6 MB) | Película / live action / video digital: quita compresión y halos sin suavizar de más | 2x | CC-BY-NC-SA-4.0 | [OpenModelDB](https://openmodeldb.info/models/2x-LiveActionV1-SPAN) · [.onnx directo](https://github.com/jcj83429/upscaling/raw/f73a3a02874360ec6ced18f8bdd8e43b5d7bba57/2xLiveActionV1_SPAN/2xLiveActionV1_SPAN_490000.onnx) |
| AnimeJaNai HD V3 (Compact / UltraCompact / SuperUltraCompact) | Anime HD 720p–1080p | 2x | CC-BY-NC-SA-4.0 | [zip](https://github.com/the-database/mpv-upscale-2x_animejanai/releases/download/3.0.0/2x_AnimeJaNai_HD_V3_ModelsOnly.zip) · [proyecto](https://github.com/the-database/mpv-AnimeJaNai) |
| AnimeJaNai V2 | Anime, generación anterior | 2x | CC-BY-NC-SA-4.0 | [zip](https://github.com/the-database/mpv-upscale-2x_animejanai/releases/download/2.0.0/2x_AnimeJaNai_V2_ModelsOnly.zip) |
| AnimeJaNai v1 (Compact recomendado para SD) | Anime SD/medio | 2x | CC-BY-NC-SA-4.0 | [zip](https://github.com/the-database/mpv-upscale-2x_animejanai/releases/download/1.0.0/mpv-upscale-2x_animejanai_v1.zip) · [OpenModelDB](https://openmodeldb.info/models/2x-AnimeJaNai-Standard-v1-Compact) |

Dentro de cada familia, **SuperUltraCompact/UltraCompact** son los más rápidos y **Compact** el de más calidad (y más lento).
Los AnimeJaNai V3.1 más nuevos (SPAN) solo vienen dentro de los paquetes grandes de [mpv-AnimeJaNai](https://github.com/the-database/mpv-AnimeJaNai/releases).

## Se convierten con el workflow (`.pth` → ONNX)

| Archivo que obtendrás | Para qué | Licencia | `.pth` original |
|---|---|---|---|
| 2x_RealESRGANv2_AnimeVideo_xs.onnx | Anime video 2x, muy ligero | BSD-3-Clause | [descarga](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.3.0/RealESRGANv2-animevideo-xsx2.pth) |
| 4x_RealESR_AnimeVideo_v3.onnx | Anime SD/480p, 4x Compact | BSD-3-Clause | [descarga](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth) |
| 2x_HFA2kAVCCompact.onnx | Anime con compresión H.264 | CC-BY-4.0 | [descarga](https://github.com/Phhofm/models/raw/main/2xHFA2kAVCCompact/2xHFA2kAVCCompact.pth) |
| 2x_AniScale2S_Compact.onnx | Anime WEB/DVD, buena limpieza de compresión | CC-BY-NC-4.0 | [descarga](https://github.com/Sirosky/Upscale-Hub/releases/download/AniScale2/2x_AniScale2S_Compact_i8_60K.pth) |
| 2x_ModernSpanimationV1.onnx | Anime 2x SPAN, rápido | MIT | [descarga](https://github.com/TNTwise/Models/releases/download/2x_ModernSpanimationV1/2x_ModernSpanimationV1.pth) |
| 1x_BroadcastToStudio_Compact.onnx | Limpieza 1x de material de TV | CC-BY-NC-SA-4.0 | [descarga](https://github.com/Kim2091/Kim2091-Models/releases/download/1x-BroadcastToStudio_Compact/1x-BroadcastToStudio_Compact.pth) |
| 4x_PurePhoto_SPAN.onnx | Fotos 4x, rápido | CC-BY-SA-4.0 | [descarga](https://github.com/starinspace/StarinspaceUpscale/releases/download/Models/4xPurePhoto-span.pth) |
| 2x_RealESRGAN_x2plus.onnx | Fotos 2x, calidad alta pero pesado | BSD-3-Clause | [descarga](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth) |

La lista vive en `tools/models_to_convert.json`: añade ahí tus propios `.pth` (campo `url` y `file`) y vuelve a ejecutar el workflow.

### Convertir uno a mano

```
pip install torch spandrel onnx onnxruntime
python tools/convert_to_onnx.py modelo.pth            # genera modelo.onnx con ejes dinámicos
python tools/convert_to_onnx.py modelo.pth --fp16     # (opcional, requiere GPU CUDA) más rápido
```

Funciona bien con arquitecturas Compact, SPAN, RealPLKSR y ESRGAN. Las de tipo transformer (SwinIR, DAT, HAT…) suelen fallar o ir muy lentas.
Alternativa: chaiNNer → nodo de conversión a ONNX.

## Qué elegir (RTX 3060 Ti, render offline)

| Tu material | Empieza con | Preset sugerido |
|---|---|---|
| Anime HD (720p/1080p) | AnimeJaNai HD V3 UltraCompact → prueba Compact | Anime 720p a 1080p |
| Anime SD, DVD o WEB con compresión | AniScale2 Compact, HFA2kAVCCompact o AnimeJaNai v1 Compact | Anime 720p a 1080p |
| Anime viejo/pequeño, prisa | RealESRGANv2 AnimeVideo xs o ModernSpanimationV1 | Borrador rápido |
| Película / live action | 2xLiveActionV1 SPAN | Película DVD/HD a 1080p |
| Fotos | PurePhoto SPAN (rápido) o RealESRGAN x2plus (más calidad) | Foto: restaurar |

Consejos: prueba siempre con **«Probar solo los primeros N s»**; no subas a 4K a menos que lo necesites (el peso se dispara);
si te quedas sin VRAM usa `tile` 256–512.

Para live action, los modelos abiertos cuidan la fidelidad, pero un producto comercial como Topaz puede dar mejores resultados en ese tipo de material.

## Licencias

La mayoría son **uso personal / no comercial** (CC-BY-NC…): perfecto para tu uso, pero no los uses en un servicio de pago.
Cada modelo conserva su licencia; consulta su página antes de redistribuirlo.

## Problemas típicos

- *«El archivo no contiene .onnx»*: ese paquete trae `.pth`; usa el workflow de conversión.
- *Error con DirectML*: prueba `Backend = CPU` para descartar el modelo, o convierte el modelo en fp32.
- *Modelo de tamaño fijo*: funciona (por mosaicos) pero es lento; conviértelo con ejes dinámicos.
- *Colores raros o bandas*: activa 10 bits y baja el CRF; evita pre-limpiezas fuertes.

## Dónde buscar más

- [OpenModelDB](https://openmodeldb.info) (filtra por arquitectura Compact/SPAN y por etiqueta *Video Frame*, *Anime*, *Restoration*).
- [Upscale-Hub (Sirosky)](https://github.com/Sirosky/Upscale-Hub/releases): AniScale, AniSD (DVD) y más.
- [Guía de Zarxrax sobre upscaling con IA](https://sites.google.com/view/amv101/guides/ai-upscaling).
