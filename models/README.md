# Modelos de IA

Pon aquí tus modelos `.onnx`. Nítido los detecta solo (botón **Recargar**).

- Búscalos en **OpenModelDB** (openmodeldb.info). Para anime/video elige modelos "compact" o "SPAN" (rápidos);
  para fotos, modelos más grandes. Revisa la licencia de cada uno.
- Si solo hay `.pth`/`.safetensors`, conviértelos a ONNX con **chaiNNer** (nodo "Save Model" → ONNX)
  usando **ejes dinámicos** (alto y ancho variables). Los ONNX de tamaño fijo no funcionan.
- La escala (2x, 4x…) se detecta automáticamente.
