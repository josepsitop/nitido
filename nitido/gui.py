"""Nítido · interfaz (PySide6)."""
import sys
import threading
from pathlib import Path

from PySide6.QtCore import QThread, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
    QInputDialog, QLabel, QLineEdit, QListWidget, QMainWindow, QMessageBox, QPlainTextEdit,
    QProgressBar, QPushButton, QSpinBox, QVBoxLayout, QWidget)

from . import __version__, core

MEDIA = ("Video/foto (*.mp4 *.mkv *.avi *.mov *.webm *.m4v *.ts *.wmv *.flv *.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff);;Todos (*.*)")


class Worker(QThread):
    progress = Signal(int, int, float)
    log = Signal(str)
    file_started = Signal(int, int)

    def __init__(self, files, preset, outdir, test_s):
        super().__init__()
        self.files, self.p, self.outdir, self.test_s = files, preset, outdir, test_s
        self.cancel = threading.Event()

    def run(self):
        for i, src in enumerate(self.files):
            if self.cancel.is_set():
                break
            self.file_started.emit(i + 1, len(self.files))
            try:
                dst = core.output_path(src, self.outdir, self.p)
                self.log.emit(f"▶ {Path(src).name}  →  {dst}")
                core.run_job(src, dst, self.p, lambda d, t, f: self.progress.emit(d, t, f),
                             self.log.emit, self.cancel, self.test_s)
                self.log.emit("✔ Listo")
            except core.Cancelled:
                self.log.emit("⏹ Cancelado")
                break
            except Exception as e:
                self.log.emit(f"✖ Error: {e}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Nítido {__version__} · reescalado y limpieza con IA")
        self.setAcceptDrops(True)
        self.worker = None
        self.presets = {}
        self._build()
        self.refresh_models()
        self.refresh_presets()
        self._startup_log()

    # ------------------------------------------------------------ UI
    def _build(self):
        root = QWidget()
        self.setCentralWidget(root)
        main = QVBoxLayout(root)
        cols = QHBoxLayout()
        main.addLayout(cols, 1)

        # izquierda: cola de archivos
        g1 = QGroupBox("Archivos (también puedes arrastrarlos aquí)")
        l1 = QVBoxLayout(g1)
        self.lst = QListWidget()
        self.lst.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        l1.addWidget(self.lst)
        row = QHBoxLayout()
        for text, fn in (("Añadir…", self.pick_files), ("Quitar", self.remove_files),
                         ("Vaciar", self.lst.clear)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        l1.addLayout(row)
        self.ed_out = QLineEdit()
        self.ed_out.setPlaceholderText("Carpeta de salida (vacío = junto al original)")
        b_out = QPushButton("…")
        b_out.setMaximumWidth(32)
        b_out.clicked.connect(self.pick_outdir)
        row = QHBoxLayout()
        row.addWidget(self.ed_out)
        row.addWidget(b_out)
        l1.addLayout(row)
        cols.addWidget(g1, 2)

        # derecha: preset + parámetros
        right = QVBoxLayout()
        cols.addLayout(right, 3)

        g2 = QGroupBox("Preset")
        l2 = QHBoxLayout(g2)
        self.cb_preset = QComboBox()
        self.cb_preset.setMinimumWidth(220)
        self.cb_preset.currentTextChanged.connect(self.on_preset)
        b_save, b_del = QPushButton("Guardar como…"), QPushButton("Eliminar")
        b_save.clicked.connect(self.save_preset)
        b_del.clicked.connect(self.delete_preset)
        l2.addWidget(self.cb_preset, 1)
        l2.addWidget(b_save)
        l2.addWidget(b_del)
        right.addWidget(g2)

        g3 = QGroupBox("Ajustes del preset (editables)")
        f = QFormLayout(g3)
        self.cb_model = QComboBox()
        b_mod, b_ref = QPushButton("Abrir carpeta"), QPushButton("Recargar")
        b_mod.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(core.models_dir()))))
        b_ref.clicked.connect(self.refresh_models)
        row = QHBoxLayout()
        row.addWidget(self.cb_model, 1)
        row.addWidget(b_mod)
        row.addWidget(b_ref)
        f.addRow("Modelo IA (.onnx)", row)

        self.cb_prov = QComboBox()
        for t, v in (("Auto (TensorRT → CUDA → DirectML)", "auto"), ("TensorRT", "tensorrt"),
                     ("CUDA", "cuda"), ("DirectML (cualquier GPU)", "directml"), ("CPU (lento)", "cpu")):
            self.cb_prov.addItem(t, v)
        f.addRow("Backend", self.cb_prov)

        self.sp_tile = QSpinBox()
        self.sp_tile.setRange(0, 4096)
        self.sp_tile.setSingleStep(64)
        self.sp_tile.setSpecialValueText("Frame completo")
        self.sp_tile.setToolTip("Si te quedas sin VRAM, usa tiles (p. ej. 256 o 384).")
        f.addRow("Tile", self.sp_tile)

        self.ed_pre = QLineEdit()
        self.ed_pre.setPlaceholderText("filtro ffmpeg previo, p. ej. hqdn3d=1.5:1.5:6:6")
        f.addRow("Pre-limpieza", self.ed_pre)

        self.sp_mh = QSpinBox()
        self.sp_mh.setRange(0, 4320)
        self.sp_mh.setSingleStep(2)
        self.sp_mh.setSpecialValueText("Sin límite")
        self.sp_mh.setToolTip("Si la salida es más alta, se reduce a esta altura (ahorra peso).")
        f.addRow("Altura máxima", self.sp_mh)

        self.cb_enc = QComboBox()
        self.cb_enc.addItems(["libx265", "hevc_nvenc", "libsvtav1", "libx264"])
        f.addRow("Codec", self.cb_enc)
        self.sp_crf = QSpinBox()
        self.sp_crf.setRange(0, 63)
        self.sp_crf.setToolTip("Menor = más calidad y más peso. x265: 18-22. NVENC (cq): 22-26. AV1: 26-32.")
        f.addRow("CRF / CQ", self.sp_crf)
        self.cb_speed = QComboBox()
        self.cb_speed.addItems(["fast", "balanced", "slow"])
        f.addRow("Velocidad de codificación", self.cb_speed)
        self.ck_10 = QCheckBox("10 bits (menos bandas y mejor compresión)")
        f.addRow("", self.ck_10)
        self.cb_cont = QComboBox()
        self.cb_cont.addItems(["mkv", "mp4"])
        f.addRow("Contenedor", self.cb_cont)
        self.cb_audio = QComboBox()
        self.cb_audio.addItems(["copy", "aac"])
        f.addRow("Audio", self.cb_audio)
        self.cb_img = QComboBox()
        self.cb_img.addItems(["png", "jpg", "webp"])
        f.addRow("Formato de fotos", self.cb_img)
        right.addWidget(g3)

        g4 = QGroupBox("Esta ejecución")
        l4 = QHBoxLayout(g4)
        self.sp_test = QSpinBox()
        self.sp_test.setRange(0, 600)
        self.sp_test.setSuffix(" s")
        self.sp_test.setSpecialValueText("Todo el video")
        self.sp_test.setToolTip("Procesa solo los primeros N segundos para probar un preset.")
        l4.addWidget(QLabel("Probar solo los primeros:"))
        l4.addWidget(self.sp_test)
        l4.addStretch(1)
        right.addWidget(g4)
        right.addStretch(1)

        # abajo: control y log
        row = QHBoxLayout()
        self.btn_go, self.btn_stop = QPushButton("▶  Procesar"), QPushButton("⏹  Cancelar")
        self.btn_go.clicked.connect(self.start)
        self.btn_stop.clicked.connect(self.cancel)
        self.btn_stop.setEnabled(False)
        row.addWidget(self.btn_go)
        row.addWidget(self.btn_stop)
        main.addLayout(row)
        self.bar = QProgressBar()
        self.lbl = QLabel("Listo.")
        main.addWidget(self.bar)
        main.addWidget(self.lbl)
        self.txt = QPlainTextEdit()
        self.txt.setReadOnly(True)
        self.txt.setMaximumBlockCount(2000)
        self.txt.setMinimumHeight(130)
        main.addWidget(self.txt)

    # ------------------------------------------------------------ archivos
    def add_file(self, path):
        if path and Path(path).is_file():
            self.lst.addItem(str(path))

    def pick_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Elegir archivos", "", MEDIA)
        for f in files:
            self.add_file(f)

    def remove_files(self):
        for it in self.lst.selectedItems():
            self.lst.takeItem(self.lst.row(it))

    def pick_outdir(self):
        d = QFileDialog.getExistingDirectory(self, "Carpeta de salida")
        if d:
            self.ed_out.setText(d)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        for u in e.mimeData().urls():
            self.add_file(u.toLocalFile())

    # ------------------------------------------------------------ presets
    def collect(self):
        return {
            "name": self.cb_preset.currentText(), "model": self.cb_model.currentText(),
            "provider": self.cb_prov.currentData(), "tile": self.sp_tile.value(),
            "prefilter": self.ed_pre.text().strip(), "max_height": self.sp_mh.value(),
            "encoder": self.cb_enc.currentText(), "crf": self.sp_crf.value(),
            "speed": self.cb_speed.currentText(), "ten_bit": self.ck_10.isChecked(),
            "container": self.cb_cont.currentText(), "audio": self.cb_audio.currentText(),
            "image_format": self.cb_img.currentText(),
        }

    def apply(self, d):
        d = {**core.DEFAULT, **d}
        if d["model"] and self.cb_model.findText(d["model"]) >= 0:
            self.cb_model.setCurrentText(d["model"])
        self.cb_prov.setCurrentIndex(max(0, self.cb_prov.findData(d["provider"])))
        self.sp_tile.setValue(int(d["tile"]))
        self.ed_pre.setText(d["prefilter"])
        self.sp_mh.setValue(int(d["max_height"]))
        self.cb_enc.setCurrentText(d["encoder"])
        self.sp_crf.setValue(int(d["crf"]))
        self.cb_speed.setCurrentText(d["speed"])
        self.ck_10.setChecked(bool(d["ten_bit"]))
        self.cb_cont.setCurrentText(d["container"])
        self.cb_audio.setCurrentText(d["audio"])
        self.cb_img.setCurrentText(d["image_format"])

    def refresh_models(self):
        cur = self.cb_model.currentText()
        self.cb_model.clear()
        self.cb_model.addItems(core.list_models())
        if cur:
            self.cb_model.setCurrentText(cur)

    def refresh_presets(self, select=None):
        self.presets = core.list_presets()
        self.cb_preset.blockSignals(True)
        self.cb_preset.clear()
        self.cb_preset.addItems(list(self.presets))
        if select in self.presets:
            self.cb_preset.setCurrentText(select)
        self.cb_preset.blockSignals(False)
        if self.presets:
            self.apply(self.presets[self.cb_preset.currentText()])

    def on_preset(self, name):
        if name in self.presets:
            self.apply(self.presets[name])

    def save_preset(self):
        name, ok = QInputDialog.getText(self, "Guardar preset", "Nombre del preset:",
                                        QLineEdit.EchoMode.Normal, self.cb_preset.currentText())
        if ok and name.strip():
            p = self.collect()
            p["name"] = name.strip()
            core.save_preset(p)
            self.refresh_presets(p["name"])

    def delete_preset(self):
        name = self.cb_preset.currentText()
        if name and QMessageBox.question(self, "Eliminar", f"¿Eliminar el preset «{name}»?") == QMessageBox.StandardButton.Yes:
            core.delete_preset(name)
            self.refresh_presets()

    # ------------------------------------------------------------ ejecución
    def log(self, s):
        self.txt.appendPlainText(s)

    def _startup_log(self):
        self.log(f"Carpeta base: {core.base_dir()}")
        self.log("ffmpeg: " + (core.find_tool("ffmpeg") or "NO ENCONTRADO (carpeta bin/ o PATH)"))
        self.log(f"Modelos .onnx encontrados: {len(core.list_models())}  (carpeta models/)")
        try:
            import onnxruntime as ort
            self.log("Backends disponibles: " + ", ".join(ort.get_available_providers()))
        except Exception as e:
            self.log(f"onnxruntime no disponible: {e}")

    def set_busy(self, busy):
        self.btn_go.setEnabled(not busy)
        self.btn_stop.setEnabled(busy)

    def start(self):
        files = [self.lst.item(i).text() for i in range(self.lst.count())]
        if not files:
            self.log("Añade al menos un archivo.")
            return
        if not core.find_tool("ffmpeg"):
            self.log("✖ No encuentro ffmpeg. Ponlo en la carpeta bin/ junto a la app.")
            return
        self.set_busy(True)
        self.worker = Worker(files, self.collect(), self.ed_out.text().strip(), self.sp_test.value())
        self.worker.progress.connect(self.on_progress)
        self.worker.log.connect(self.log)
        self.worker.file_started.connect(self.on_file)
        self.worker.finished.connect(lambda: (self.set_busy(False), self.lbl.setText("Terminado.")))
        self.worker.start()

    def cancel(self):
        if self.worker:
            self.worker.cancel.set()

    def on_file(self, i, n):
        self.bar.setValue(0)
        self.lbl.setText(f"Archivo {i}/{n}…")

    def on_progress(self, done, total, fps):
        self.bar.setMaximum(max(total, 1))
        self.bar.setValue(min(done, total))
        eta = int((total - done) / fps) if fps > 0 else 0
        h, m, s = eta // 3600, eta % 3600 // 60, eta % 60
        self.lbl.setText(f"Frame {done}/{total} · {fps:.2f} fps · quedan ≈ {h:d}:{m:02d}:{s:02d}")

    def closeEvent(self, e):
        if self.worker and self.worker.isRunning():
            self.worker.cancel.set()
            self.worker.wait(5000)
        e.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = MainWindow()
    w.resize(1020, 780)
    w.show()
    sys.exit(app.exec())
