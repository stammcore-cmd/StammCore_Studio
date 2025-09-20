"""Application entry point for StammCore Studio Full.

This module creates and launches the PySide6 based GUI.  It does not
perform any heavy machine learning imports at import time to allow the
program to start quickly and to fail gracefully if dependencies are
missing.  All heavy work is deferred to worker threads spawned by
individual pages.
"""

import sys

try:
    from PySide6.QtWidgets import QApplication
except Exception as exc:  # pragma: no cover - runtime import guard
    raise RuntimeError(
        "PySide6 must be installed to run the StammCore Studio. "
        "Please install the required dependencies and try again."
    ) from exc

from .gui.main_window import MainWindow


def main() -> None:
    """Launch the StammCore Studio application."""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()