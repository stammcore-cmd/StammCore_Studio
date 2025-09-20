"""Main application window for StammCore Studio Full.

This module defines the ``MainWindow`` class which composes the
navigation sidebar, top bar with logo/header and a stack of pages.
Individual pages implement the functionality for dummy data
generation, model training, library management and testing.  The
window applies a dark theme loaded from a QSS stylesheet located in
the assets directory and attempts to display user provided logo and
header images if present.
"""

from __future__ import annotations

import os
from typing import Dict, Type

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QStackedWidget,
    QFrame,
    QSizePolicy,
)

from .pages.dummy_generator_page import DummyGeneratorPage
from .pages.training_page import TrainingPage
from .pages.library_page import LibraryPage
from .pages.test_page import TestPage
from .pages.annotation_page import AnnotationPage
from .pages.system_check_page import SystemCheckPage


class MainWindow(QMainWindow):
    """Top level window containing sidebar, top bar and content pages."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("StammCore Studio")
        self.resize(1200, 800)
        # internal storage for page instances
        self._pages: Dict[str, QWidget] = {}
        self._init_ui()
        # apply dark stylesheet
        self._apply_stylesheet()

    def _init_ui(self) -> None:
        """Construct the UI layout with sidebar, top bar and stacked pages."""
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # Sidebar
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(200)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(0)

        # Navigation buttons
        nav_buttons = [
            ("Dummy Generator", self.show_dummy_generator),
            ("Trainings Center", self.show_training_center),
            ("Bibliothek", self.show_library),
            ("Test Center", self.show_test_center),
            ("Annotation Studio", self.show_annotation_studio),
            ("System-Check", self.show_system_check),
        ]
        for text, handler in nav_buttons:
            btn = QPushButton(text)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            btn.setObjectName("navButton")
            btn.clicked.connect(handler)  # type: ignore[arg-type]
            sidebar_layout.addWidget(btn)
        # stretch to bottom
        sidebar_layout.addStretch(1)

        # Main area: top bar + stacked pages
        main_area = QFrame()
        main_area_layout = QVBoxLayout(main_area)
        main_area_layout.setContentsMargins(0, 0, 0, 0)
        main_area_layout.setSpacing(0)

        # Top bar
        top_bar = QFrame()
        top_bar.setObjectName("topBar")
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(10, 5, 10, 5)
        top_layout.setSpacing(10)
        # Logo
        logo_label = QLabel()
        logo_pixmap = self._load_pixmap("logo/logo.png")
        if logo_pixmap is not None:
            logo_label.setPixmap(logo_pixmap.scaledToHeight(40, Qt.SmoothTransformation))
        else:
            logo_label.setText("StammCore")
            logo_label.setStyleSheet("color: white; font-weight: bold; font-size: 20px;")
        top_layout.addWidget(logo_label)
        # Header
        header_label = QLabel()
        header_pixmap = self._load_pixmap("header/header.png")
        if header_pixmap is not None:
            header_label.setPixmap(header_pixmap.scaledToHeight(40, Qt.SmoothTransformation))
        else:
            header_label.setText("Studio")
            header_label.setStyleSheet("color: white; font-size: 16px;")
        # align header left but take available space
        header_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        top_layout.addWidget(header_label)
        # optional filler for alignment
        top_layout.addStretch(1)

        main_area_layout.addWidget(top_bar)

        # Stacked widget for pages
        self.stacked = QStackedWidget()
        main_area_layout.addWidget(self.stacked)

        # add sidebar and main area to root layout
        root_layout.addWidget(sidebar)
        root_layout.addWidget(main_area)

        # Create pages and add to stack
        self._pages = {
            "dummy": DummyGeneratorPage(parent=self),
            "training": TrainingPage(parent=self),
            "library": LibraryPage(parent=self),
            "test": TestPage(parent=self),
            "annotation": AnnotationPage(parent=self),
            "system": SystemCheckPage(parent=self),
        }
        for page in self._pages.values():
            self.stacked.addWidget(page)
        # show default page
        self.show_dummy_generator()

    def _apply_stylesheet(self) -> None:
        """Apply dark theme QSS from assets if available."""
        # Path relative to this file
        qss_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../assets/dark.qss")
        try:
            with open(qss_path, "r", encoding="utf-8") as f:
                self.setStyleSheet(f.read())
        except Exception:
            # fallback to a simple dark palette if QSS missing
            self.setStyleSheet(
                """
                QWidget { background-color: #121212; color: #EEEEEE; }
                QPushButton#navButton { background-color: #1E1E1E; color: #CCCCCC; border: none; padding: 10px; }
                QPushButton#navButton:hover { background-color: #2E2E2E; }
                QPushButton#navButton:checked { background-color: #3F51B5; color: white; }
                QFrame#topBar { background-color: #1E1E1E; }
                QTextEdit { background-color: #1E1E1E; color: #CCCCCC; }
                QLineEdit { background-color: #2E2E2E; color: white; border: 1px solid #444444; padding: 4px; }
                QLabel { color: #CCCCCC; }
                """
            )

    def _load_pixmap(self, relative_path: str) -> QPixmap | None:
        """Load a QPixmap from the assets folder if it exists."""
        # assets folder is ../assets relative to this file
        assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../assets")
        path = os.path.join(assets_dir, relative_path)
        if os.path.exists(path):
            pixmap = QPixmap(path)
            if not pixmap.isNull():
                return pixmap
        return None

    # Navigation slots
    def show_dummy_generator(self) -> None:
        self.stacked.setCurrentWidget(self._pages["dummy"])

    def show_training_center(self) -> None:
        self.stacked.setCurrentWidget(self._pages["training"])

    def show_library(self) -> None:
        self.stacked.setCurrentWidget(self._pages["library"])

    def show_test_center(self) -> None:
        self.stacked.setCurrentWidget(self._pages["test"])

    def show_annotation_studio(self) -> None:
        """Switch to the annotation studio page."""
        self.stacked.setCurrentWidget(self._pages["annotation"])

    def show_system_check(self) -> None:
        """Switch to the system check page."""
        self.stacked.setCurrentWidget(self._pages["system"])