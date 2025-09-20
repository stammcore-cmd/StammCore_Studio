"""StammCore Studio Full package.

This package contains a fully featured desktop application for training and
inference using the LayoutLMv3 model on scanned forms.  It exposes a
modular GUI built on PySide6 that covers the entire workflow from
synthetic data generation through model training, library management and
document testing.  Heavy machine learning imports (PyTorch, transformers,
PaddleOCR, etc.) are only performed inside worker threads so that the
application can start even if some dependencies are missing.  To launch
the app simply run ``python -m stammcore_studio_full`` in a properly
configured Python environment.
"""

from .main import main  # noqa: F401

__all__ = ["main"]