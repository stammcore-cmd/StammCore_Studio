"""GUI page for testing trained models on new documents."""

from __future__ import annotations

import os
import threading
import json
from typing import Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QHBoxLayout,
    QFileDialog,
    QTextEdit,
    QScrollArea,
)

from ...core.inference import run_inference


class TestPage(QWidget):
    """Page allowing users to run inference with a selected model on a new image."""

    log_signal = Signal(str)
    finished_signal = Signal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_ui()
        self.log_signal.connect(self._append_log)
        self.finished_signal.connect(self._on_finished)

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        form_layout = QFormLayout()
        # Model directory
        self.model_edit = QLineEdit()
        model_btn = QPushButton("Durchsuchen…")
        model_btn.clicked.connect(self._browse_model)  # type: ignore[arg-type]
        h1 = QHBoxLayout()
        h1.addWidget(self.model_edit)
        h1.addWidget(model_btn)
        form_layout.addRow(QLabel("Modellordner:"), h1)
        # Image file
        self.image_edit = QLineEdit()
        image_btn = QPushButton("Durchsuchen…")
        image_btn.clicked.connect(self._browse_image)  # type: ignore[arg-type]
        h2 = QHBoxLayout()
        h2.addWidget(self.image_edit)
        h2.addWidget(image_btn)
        form_layout.addRow(QLabel("Bilddatei:"), h2)
        layout.addLayout(form_layout)
        # Inference button
        self.infer_button = QPushButton("Inference ausführen")
        self.infer_button.clicked.connect(self._start_inference)  # type: ignore[arg-type]
        layout.addWidget(self.infer_button)
        # Log and output
        layout.addWidget(QLabel("Ausgabe:"))
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        layout.addWidget(self.output_text)

    def _browse_model(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Modellordner auswählen", "")
        if directory:
            self.model_edit.setText(directory)

    def _browse_image(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(self, "Bild auswählen", "", "Image Files (*.png *.jpg *.jpeg *.bmp);;All Files (*)")
        if filename:
            self.image_edit.setText(filename)

    def _start_inference(self) -> None:
        model_dir = self.model_edit.text().strip()
        image_path = self.image_edit.text().strip()
        if not model_dir or not os.path.isdir(model_dir):
            self._append_log("Bitte einen gültigen Modellordner auswählen.")
            return
        if not image_path or not os.path.isfile(image_path):
            self._append_log("Bitte eine gültige Bilddatei auswählen.")
            return
        self.infer_button.setEnabled(False)
        self.output_text.clear()
        self._append_log("Starte Inferenz...")
        def run():
            try:
                result = run_inference(model_dir, image_path, log_callback=self.log_signal.emit)
            except Exception as exc:
                self.log_signal.emit(f"Fehler während der Inferenz: {exc}")
                self.finished_signal.emit({})
                return
            self.finished_signal.emit(result)
        threading.Thread(target=run, daemon=True).start()

    def _append_log(self, message: str) -> None:
        self.output_text.append(message)

    def _on_finished(self, result: dict) -> None:
        self.infer_button.setEnabled(True)
        if result:
            formatted = json.dumps(result, indent=2, ensure_ascii=False)
            self.output_text.append("Extrahierte Entitäten:\n" + formatted)
        else:
            self.output_text.append("Inferenz abgeschlossen, keine Entitäten gefunden oder Fehler.")