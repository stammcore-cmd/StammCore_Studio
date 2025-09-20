"""GUI page for training LayoutLMv3 models on custom datasets."""

from __future__ import annotations

import os
import threading
from datetime import datetime
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
    QTextEdit,
    QFileDialog,
    QSpinBox,
    QDoubleSpinBox,
)

from ...core.training import train_layoutlm_model
import json


class TrainingPage(QWidget):
    """Page that lets the user configure and run training jobs."""

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
        # Dataset JSON
        self.dataset_edit = QLineEdit()
        ds_btn = QPushButton("Durchsuchen…")
        ds_btn.clicked.connect(self._browse_dataset)  # type: ignore[arg-type]
        h1 = QHBoxLayout()
        h1.addWidget(self.dataset_edit)
        h1.addWidget(ds_btn)
        form_layout.addRow(QLabel("Dataset JSON:"), h1)
        # Images directory
        self.images_edit = QLineEdit()
        images_btn = QPushButton("Durchsuchen…")
        images_btn.clicked.connect(self._browse_images)  # type: ignore[arg-type]
        h2 = QHBoxLayout()
        h2.addWidget(self.images_edit)
        h2.addWidget(images_btn)
        form_layout.addRow(QLabel("Bildordner:"), h2)
        # Output directory
        self.output_edit = QLineEdit()
        output_btn = QPushButton("Durchsuchen…")
        output_btn.clicked.connect(self._browse_output)  # type: ignore[arg-type]
        h3 = QHBoxLayout()
        h3.addWidget(self.output_edit)
        h3.addWidget(output_btn)
        form_layout.addRow(QLabel("Ausgabeordner:"), h3)
        # Hyperparameters
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 100)
        self.epochs_spin.setValue(3)
        form_layout.addRow(QLabel("Epochen:"), self.epochs_spin)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 16)
        self.batch_spin.setValue(2)
        form_layout.addRow(QLabel("Batchgröße:"), self.batch_spin)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(1e-6, 1e-3)
        self.lr_spin.setDecimals(6)
        self.lr_spin.setSingleStep(1e-5)
        self.lr_spin.setValue(1e-5)
        form_layout.addRow(QLabel("Lernrate:"), self.lr_spin)
        self.warmup_spin = QDoubleSpinBox()
        self.warmup_spin.setRange(0.0, 0.9)
        self.warmup_spin.setSingleStep(0.05)
        self.warmup_spin.setValue(0.1)
        form_layout.addRow(QLabel("Warmup Verhältnis:"), self.warmup_spin)
        layout.addLayout(form_layout)
        # Train button
        self.train_button = QPushButton("Training starten")
        self.train_button.clicked.connect(self._start_training)  # type: ignore[arg-type]
        layout.addWidget(self.train_button)
        # Log output
        layout.addWidget(QLabel("Training Log:"))
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)
        # Save metrics placeholder (could be used later)
        self.metrics: Optional[dict] = None

    def _browse_dataset(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(self, "Dataset JSON auswählen", "", "JSON Files (*.json);;All Files (*)")
        if filename:
            self.dataset_edit.setText(filename)

    def _browse_images(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Bildordner auswählen", "")
        if directory:
            self.images_edit.setText(directory)

    def _browse_output(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Ausgabeordner auswählen", "")
        if directory:
            self.output_edit.setText(directory)

    def _start_training(self) -> None:
        dataset_path = self.dataset_edit.text().strip()
        images_dir = self.images_edit.text().strip()
        output_dir = self.output_edit.text().strip()
        if not dataset_path or not os.path.isfile(dataset_path):
            self._append_log("Bitte ein gültiges Dataset JSON auswählen.")
            return
        if not images_dir or not os.path.isdir(images_dir):
            self._append_log("Bitte einen gültigen Bildordner auswählen.")
            return
        if not output_dir:
            self._append_log("Bitte einen Ausgabeordner auswählen.")
            return
        os.makedirs(output_dir, exist_ok=True)
        # disable button
        self.train_button.setEnabled(False)
        self.log_text.clear()
        self._append_log(f"[{datetime.now().isoformat()}] Training gestartet...")
        num_epochs = self.epochs_spin.value()
        batch_size = self.batch_spin.value()
        lr = self.lr_spin.value()
        warmup = self.warmup_spin.value()
        # Start training in background thread
        def run_training():
            try:
                metrics = train_layoutlm_model(
                    dataset_json=dataset_path,
                    images_dir=images_dir,
                    output_dir=output_dir,
                    num_epochs=num_epochs,
                    batch_size=batch_size,
                    learning_rate=lr,
                    warmup_ratio=warmup,
                    log_callback=self.log_signal.emit,
                )
            except Exception as exc:
                self.log_signal.emit(f"Fehler während des Trainings: {exc}")
                self.finished_signal.emit({})
                return
            self.finished_signal.emit(metrics)

        threading.Thread(target=run_training, daemon=True).start()

    def _append_log(self, message: str) -> None:
        self.log_text.append(message)

    def _on_finished(self, metrics: dict) -> None:
        self.train_button.setEnabled(True)
        if metrics:
            self.metrics = metrics
            self._append_log(f"Training abgeschlossen.\nMetriken:\n{json.dumps(metrics, indent=2)}")
        else:
            self._append_log("Training abgebrochen oder fehlgeschlagen.")