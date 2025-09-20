"""System check page for StammCore Studio.

This page performs a quick environment check to verify that the key
dependencies required by the Studio (e.g. PyTorch, Transformers,
PaddleOCR) are installed and whether GPU acceleration is available.
It reports each check with a green or red indicator and can be
refreshed by the user.  Running these checks helps users diagnose
missing packages before starting training or inference tasks.
"""

from __future__ import annotations

import importlib

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton
from PySide6.QtGui import QColor
from PySide6.QtCore import Qt


class SystemCheckPage(QWidget):
    """Page that checks availability of dependencies and GPU.

    This implementation avoids importing heavy modules in the UI thread.  It
    uses ``importlib.util.find_spec`` to detect whether a package can be
    imported and lazily queries for CUDA support via PyTorch if present.
    The environment checks are deferred until after the widget has been
    constructed so that the GUI appears responsive immediately.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.labels = {}
        self._init_ui()
        # Defer environment checks to avoid blocking the UI thread on heavy imports.
        from PySide6.QtCore import QTimer
        QTimer.singleShot(0, self.run_checks)

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Systemprüfung"))
        # placeholders for results
        for key in ["torch", "transformers", "paddleocr", "cuda"]:
            lbl = QLabel()
            self.labels[key] = lbl
            layout.addWidget(lbl)
        refresh_btn = QPushButton("Erneut prüfen")
        refresh_btn.clicked.connect(self.run_checks)  # type: ignore[arg-type]
        layout.addWidget(refresh_btn)
        layout.addStretch(1)

    def run_checks(self) -> None:
        """Perform the environment checks and update the UI."""
        # Check whether core modules are available without importing them fully
        torch_ok = self._module_available("torch")
        self._set_status("torch", torch_ok)
        transformers_ok = self._module_available("transformers")
        self._set_status("transformers", transformers_ok)
        paddle_ok = self._module_available("paddleocr")
        self._set_status("paddleocr", paddle_ok)
        # Determine CUDA availability via torch if present
        cuda_available = False
        if torch_ok:
            try:
                import importlib
                torch_mod = importlib.import_module("torch")
                cuda_available = bool(getattr(torch_mod, "cuda", None)) and torch_mod.cuda.is_available()
            except Exception:
                cuda_available = False
        self._set_status("cuda", cuda_available)

    def _module_available(self, module_name: str) -> bool:
        """Check if a module is available without importing it fully.

        This uses ``importlib.util.find_spec`` which performs a lightweight
        lookup instead of fully importing the module.  It avoids side
        effects and long import times when checking optional dependencies.
        """
        import importlib.util
        try:
            return importlib.util.find_spec(module_name) is not None
        except Exception:
            return False

    def _set_status(self, key: str, ok: bool) -> None:
        """Update a status label with a coloured indicator."""
        label = self.labels.get(key)
        if not label:
            return
        colour = "#4CAF50" if ok else "#F44336"  # green vs red
        if key == "cuda" and not ok:
            text = "CUDA-Unterstützung: nicht verfügbar"
        elif key == "cuda" and ok:
            text = "CUDA-Unterstützung: verfügbar"
        else:
            text = f"{key}: {'OK' if ok else 'fehlt'}"
        label.setText(text)
        label.setStyleSheet(f"color: {colour}; font-weight: bold;")