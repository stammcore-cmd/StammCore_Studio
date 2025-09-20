"""GUI page for browsing and inspecting trained models."""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import List, Optional

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QLineEdit,
    QFileDialog,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
)


class LibraryPage(QWidget):
    """Page that displays a library of trained models and their metrics."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        # Path to models directory
        path_layout = QHBoxLayout()
        self.models_dir_edit = QLineEdit()
        # Default to the 'models' folder within the package if it exists;
        # fall back to models folder in current working directory otherwise.
        package_models = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models")
        default_dir = package_models if os.path.isdir(package_models) else os.path.join(os.getcwd(), "models")
        self.models_dir_edit.setText(default_dir)
        browse_btn = QPushButton("Durchsuchen…")
        browse_btn.clicked.connect(self._browse_models_dir)  # type: ignore[arg-type]
        path_layout.addWidget(QLabel("Modelle Ordner:"))
        path_layout.addWidget(self.models_dir_edit)
        path_layout.addWidget(browse_btn)
        layout.addLayout(path_layout)
        # Refresh button
        refresh_btn = QPushButton("Aktualisieren")
        refresh_btn.clicked.connect(self._refresh_models)  # type: ignore[arg-type]
        layout.addWidget(refresh_btn)
        # Table
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Modell", "F1", "Precision", "Recall"])
        self.table.cellClicked.connect(self._on_row_selected)  # type: ignore[arg-type]
        layout.addWidget(self.table)
        # Details box
        self.details = QTextEdit()
        self.details.setReadOnly(True)
        layout.addWidget(QLabel("Details:"))
        layout.addWidget(self.details)
        # Populate initial list
        self._refresh_models()

    def _browse_models_dir(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Modelle Ordner auswählen", "")
        if directory:
            self.models_dir_edit.setText(directory)
            self._refresh_models()

    def _refresh_models(self) -> None:
        models_dir = self.models_dir_edit.text().strip()
        self.table.setRowCount(0)
        self.details.clear()
        if not models_dir or not os.path.isdir(models_dir):
            return
        for subdir in sorted(os.listdir(models_dir)):
            model_path = os.path.join(models_dir, subdir)
            metrics_path = os.path.join(model_path, "metrics.json")
            if os.path.isdir(model_path) and os.path.isfile(metrics_path):
                try:
                    with open(metrics_path, "r", encoding="utf-8") as f:
                        metrics = json.load(f)
                except Exception:
                    metrics = {}
                row = self.table.rowCount()
                self.table.insertRow(row)
                self.table.setItem(row, 0, QTableWidgetItem(subdir))
                self.table.setItem(row, 1, QTableWidgetItem(f"{metrics.get('f1', 0):.4f}"))
                self.table.setItem(row, 2, QTableWidgetItem(f"{metrics.get('precision', 0):.4f}"))
                self.table.setItem(row, 3, QTableWidgetItem(f"{metrics.get('recall', 0):.4f}"))
        self.table.resizeColumnsToContents()

    def _on_row_selected(self, row: int, _column: int) -> None:
        models_dir = self.models_dir_edit.text().strip()
        model_name_item = self.table.item(row, 0)
        if not model_name_item:
            return
        model_name = model_name_item.text()
        model_path = os.path.join(models_dir, model_name)
        metrics_path = os.path.join(model_path, "metrics.json")
        details_text = []
        details_text.append(f"Modell: {model_name}")
        # modification time as training date
        try:
            mtime = os.path.getmtime(model_path)
            details_text.append(f"Trainiert am: {datetime.fromtimestamp(mtime).isoformat()}")
        except Exception:
            pass
        if os.path.isfile(metrics_path):
            try:
                with open(metrics_path, "r", encoding="utf-8") as f:
                    metrics = json.load(f)
                details_text.append("Metriken:")
                details_text.append(json.dumps(metrics, indent=2))
            except Exception:
                details_text.append("Metriken konnten nicht geladen werden.")
        else:
            details_text.append("Keine Metrikendatei gefunden.")
        self.details.setPlainText("\n".join(details_text))