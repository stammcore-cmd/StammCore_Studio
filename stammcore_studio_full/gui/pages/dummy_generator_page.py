"""GUI page for generating dummy data from an Excel file and template."""

from __future__ import annotations

import os
import threading
from functools import partial

from PySide6.QtCore import Qt, Signal, QObject
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
)

from ...core.dummy_generator import generate_dummy_documents


class DummyGeneratorPage(QWidget):
    """Page allowing the user to generate synthetic forms for training."""

    log_signal = Signal(str)
    finished_signal = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_ui()
        self.log_signal.connect(self._append_log)
        self.finished_signal.connect(self._on_finished)

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        form_layout = QFormLayout()
        # Excel file
        self.excel_edit = QLineEdit()
        excel_btn = QPushButton("Durchsuchen…")
        excel_btn.clicked.connect(self._browse_excel)  # type: ignore[arg-type]
        h1 = QHBoxLayout()
        h1.addWidget(self.excel_edit)
        h1.addWidget(excel_btn)
        form_layout.addRow(QLabel("Excel Datei:"), h1)
        # Template file
        self.template_edit = QLineEdit()
        template_btn = QPushButton("Durchsuchen…")
        template_btn.clicked.connect(self._browse_template)  # type: ignore[arg-type]
        h2 = QHBoxLayout()
        h2.addWidget(self.template_edit)
        h2.addWidget(template_btn)
        form_layout.addRow(QLabel("Vorlagen-Datei:"), h2)
        # Output directory
        self.output_edit = QLineEdit()
        output_btn = QPushButton("Durchsuchen…")
        output_btn.clicked.connect(self._browse_output)  # type: ignore[arg-type]
        h3 = QHBoxLayout()
        h3.addWidget(self.output_edit)
        h3.addWidget(output_btn)
        form_layout.addRow(QLabel("Ausgabeordner:"), h3)
        layout.addLayout(form_layout)
        # Generate button
        self.generate_button = QPushButton("Dummy Daten erzeugen")
        self.generate_button.clicked.connect(self._start_generation)  # type: ignore[arg-type]
        layout.addWidget(self.generate_button)
        # Log output
        layout.addWidget(QLabel("Protokoll:"))
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)

    def _browse_excel(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(self, "Excel Datei wählen", "", "Excel Files (*.xlsx *.xls);;All Files (*)")
        if filename:
            self.excel_edit.setText(filename)

    def _browse_template(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(self, "Vorlage wählen", "", "Image/PDF Files (*.png *.jpg *.jpeg *.pdf);;All Files (*)")
        if filename:
            self.template_edit.setText(filename)

    def _browse_output(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Ausgabeordner wählen", "")
        if directory:
            self.output_edit.setText(directory)

    def _start_generation(self) -> None:
        excel_file = self.excel_edit.text().strip()
        template_file = self.template_edit.text().strip()
        output_dir = self.output_edit.text().strip()
        if not excel_file or not os.path.isfile(excel_file):
            self._append_log("Bitte eine gültige Excel-Datei auswählen.")
            return
        if not template_file or not os.path.isfile(template_file):
            self._append_log("Bitte eine gültige Vorlage auswählen.")
            return
        if not output_dir:
            self._append_log("Bitte einen Ausgabeordner auswählen.")
            return
        os.makedirs(output_dir, exist_ok=True)
        # Disable button
        self.generate_button.setEnabled(False)
        self.log_text.clear()
        self._append_log("Starte Generierung...")
        # Start generation in a background thread
        thread = threading.Thread(
            target=generate_dummy_documents,
            args=(excel_file, template_file, output_dir),
            kwargs={"log_callback": self.log_signal.emit},
            daemon=True,
        )
        thread.start()
        # When thread finishes, call finished_signal; we can't easily detect finish
        # inside generate_dummy_documents, so we wrap it with a small helper
        def monitor():
            thread.join()
            self.finished_signal.emit()

        threading.Thread(target=monitor, daemon=True).start()

    def _append_log(self, message: str) -> None:
        self.log_text.append(message)

    def _on_finished(self) -> None:
        self.generate_button.setEnabled(True)
        self._append_log("Fertig.")