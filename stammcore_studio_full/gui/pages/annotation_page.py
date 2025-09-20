"""GUI page for annotating documents to create training JSON files.

This page allows the user to load an image, draw bounding boxes around
regions of interest, assign a canonical label to each region and
optionally edit the recognised text extracted with PaddleOCR.  The
resulting annotations can be exported as a JSON file compatible with
the training pipeline in ``core.training``.

The design follows the same dark theme used throughout the Studio and
aims to minimise blocking operations by running OCR in a background
thread.  Labels are loaded from ``label_registry.yaml`` located in the
package root; if the file is missing, a small default set of labels
is used.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import Qt, QRectF, QSize
from PySide6.QtGui import QPixmap, QPainter, QPen, QBrush
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsRectItem,
    QGraphicsPixmapItem,
    QListWidget,
    QListWidgetItem,
    QComboBox,
    QLineEdit,
    QTextEdit,
    QMessageBox,
    QFormLayout,
    QSplitter,
)

try:
    from paddleocr import PaddleOCR  # type: ignore
except Exception:
    PaddleOCR = None  # fall back if not installed


class AnnotatorCanvas(QGraphicsView):
    """A canvas for displaying an image and drawing bounding boxes."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self._pixmap_item: Optional[QGraphicsPixmapItem] = None
        self._current_rect_item: Optional[QGraphicsRectItem] = None
        self._origin = None
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        # allow dragging to pan when not drawing
        self.setDragMode(QGraphicsView.ScrollHandDrag)

    def load_image(self, image_path: str) -> None:
        """Load an image into the scene."""
        pixmap = QPixmap(image_path)
        self.scene().clear()
        self._pixmap_item = self.scene().addPixmap(pixmap)
        self.setSceneRect(pixmap.rect())
        self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)
        self._current_rect_item = None
        self._origin = None

    def mousePressEvent(self, event):  # type: ignore[override]
        if event.button() == Qt.LeftButton and self._pixmap_item is not None:
            # start drawing a rectangle
            self._origin = self.mapToScene(event.pos())
            rect = QRectF(self._origin, self._origin)
            pen = QPen(Qt.red)
            pen.setWidth(2)
            self._current_rect_item = self.scene().addRect(rect, pen, QBrush(Qt.transparent))
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):  # type: ignore[override]
        if self._current_rect_item is not None and self._origin is not None:
            # update rectangle as the mouse moves
            current_pos = self.mapToScene(event.pos())
            rect = QRectF(self._origin, current_pos).normalized()
            self._current_rect_item.setRect(rect)
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):  # type: ignore[override]
        if (self._current_rect_item is not None and self._origin is not None
                and event.button() == Qt.LeftButton):
            # finalise rectangle and emit signal via parent page
            rect = self._current_rect_item.rect()
            # clamp rect inside scene rect
            scene_rect = self.sceneRect()
            rect = rect.intersected(scene_rect)
            if rect.width() > 5 and rect.height() > 5:
                # call parent callback to handle annotation creation
                parent: AnnotationPage = self.parent()  # type: ignore[assignment]
                parent.handle_new_rectangle(rect)
            # remove temporary rectangle
            self.scene().removeItem(self._current_rect_item)
            self._current_rect_item = None
            self._origin = None
        else:
            super().mouseReleaseEvent(event)


class AnnotationPage(QWidget):
    """Annotation studio for creating labelled training data."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.image_path: Optional[str] = None
        self.annotations: List[Dict] = []
        self.label_options: List[str] = self._load_labels()
        self._ocr: Optional[PaddleOCR] = None
        self._init_ui()

    def _load_labels(self) -> List[str]:
        """Load available labels from ``label_registry.yaml``.

        Returns a list of label keys.  If the registry file is not found or
        cannot be parsed, a default set is returned.
        """
        import yaml  # local import to avoid dependency if not needed

        registry_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                     "label_registry.yaml")
        if os.path.exists(registry_path):
            try:
                with open(registry_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                return [entry["key"] for entry in data.get("labels", [])]
            except Exception:
                pass
        # fallback labels
        return ["NAME", "LAST_NAME", "ADDRESS", "POSTAL_CODE", "CITY", "BIRTH_DATE"]

    def _init_ocr(self) -> None:
        """Initialise PaddleOCR instance lazily."""
        if self._ocr is None and PaddleOCR is not None:
            # use German language as default; fallback to English
            try:
                self._ocr = PaddleOCR(lang="de", use_angle_cls=False, det=True, rec=True, show_log=False)
            except Exception:
                try:
                    self._ocr = PaddleOCR(lang="en", use_angle_cls=False, det=True, rec=True, show_log=False)
                except Exception:
                    self._ocr = None

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        # Top controls: load image and export
        control_layout = QHBoxLayout()
        load_btn = QPushButton("Bild laden…")
        load_btn.clicked.connect(self._browse_image)  # type: ignore[arg-type]
        control_layout.addWidget(load_btn)
        self.export_btn = QPushButton("Exportieren…")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self._export_json)  # type: ignore[arg-type]
        control_layout.addWidget(self.export_btn)
        layout.addLayout(control_layout)
        # Splitter to separate canvas and annotation list
        splitter = QSplitter(Qt.Horizontal)
        # Canvas area
        self.canvas = AnnotatorCanvas(self)
        splitter.addWidget(self.canvas)
        # Right side: annotation list and form
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(5)
        right_layout.addWidget(QLabel("Annotationen"))
        self.annotation_list = QListWidget()
        right_layout.addWidget(self.annotation_list, 1)
        # Annotation details form
        form_layout = QFormLayout()
        self.label_combo = QComboBox()
        self.label_combo.addItems(self.label_options)
        form_layout.addRow("Label:", self.label_combo)
        self.text_edit = QLineEdit()
        form_layout.addRow("Text:", self.text_edit)
        save_btn = QPushButton("Speichern")
        save_btn.clicked.connect(self._save_current_annotation)  # type: ignore[arg-type]
        form_layout.addRow(save_btn)
        right_layout.addLayout(form_layout)
        splitter.addWidget(right_widget)
        splitter.setSizes([700, 300])
        layout.addWidget(splitter, 1)
        # Info text
        layout.addWidget(QLabel("Zeichne ein Rechteck mit der Maus, wähle ein Label und passe den Text an."))

    def _browse_image(self) -> None:
        """Open a file dialog to select an image."""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Bild auswählen",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff);;All Files (*)",
        )
        if filename:
            self.load_image(filename)

    def load_image(self, image_path: str) -> None:
        """Load the selected image and reset annotations."""
        self.image_path = image_path
        self.annotations.clear()
        self.annotation_list.clear()
        self.canvas.load_image(image_path)
        self.export_btn.setEnabled(True)

    def handle_new_rectangle(self, rect: QRectF) -> None:
        """Handle a new rectangle drawn on the canvas.

        This method triggers OCR on the selected region (in a thread) and
        populates the label and text fields for the user to confirm.
        """
        if not self.image_path:
            return
        # convert rect to pixel coordinates
        image = QPixmap(self.image_path)
        img_w, img_h = image.width(), image.height()
        x0 = max(0, min(img_w, int(rect.left())))
        y0 = max(0, min(img_h, int(rect.top())))
        x1 = max(0, min(img_w, int(rect.right())))
        y1 = max(0, min(img_h, int(rect.bottom())))

        # Prepare new annotation structure with uuid
        annotation_id = str(uuid.uuid4())
        annotation = {
            "id": annotation_id,
            "label": self.label_combo.currentText(),
            "text": "",
            "box": [x0, y0, x1, y1],
        }

        def run_ocr():
            self._init_ocr()
            if self._ocr is None:
                return ""
            from PIL import Image as PILImage
            import numpy as np
            # crop image for OCR
            pil = PILImage.open(self.image_path).convert("RGB")
            crop = pil.crop((x0, y0, x1, y1))
            try:
                result = self._ocr.ocr(np.array(crop), cls=False)
                # result structure: [[ (poly, (text, score)), ... ]]
                texts = []
                for line in result[0]:
                    text, score = line[1][0], line[1][1]
                    if score is not None and score >= 0.5 and text.strip():
                        texts.append(text)
                return " ".join(texts).strip()
            except Exception:
                return ""

        # run OCR in background thread and update text field
        def finish(result_text: str) -> None:
            # fill fields
            annotation["text"] = result_text
            self.label_combo.setCurrentIndex(0)
            self.text_edit.setText(result_text)
            # store current rect for saving
            self._pending_annotation = annotation

        # Start OCR thread
        self._pending_annotation: Optional[Dict] = None
        def worker():
            text_result = run_ocr()
            # update UI in main thread
            def update_ui():
                finish(text_result)
            try:
                # use Qt's single shot timer to schedule call on GUI thread
                from PySide6.QtCore import QTimer
                QTimer.singleShot(0, update_ui)
            except Exception:
                finish(text_result)

        threading.Thread(target=worker, daemon=True).start()

    def _save_current_annotation(self) -> None:
        """Save the annotation currently in the form fields to the list."""
        # ensure there is a pending annotation from the last drawn rectangle
        if not hasattr(self, "_pending_annotation") or self._pending_annotation is None:
            QMessageBox.information(self, "Hinweis", "Bitte zuerst ein Rechteck zeichnen.")
            return
        annotation = self._pending_annotation
        # update with user inputs
        annotation["label"] = self.label_combo.currentText().strip()
        annotation["text"] = self.text_edit.text().strip()
        self.annotations.append(annotation)
        # display in list
        item = QListWidgetItem(f"{annotation['label']}: {annotation['text']}")
        item.setData(Qt.UserRole, annotation)
        self.annotation_list.addItem(item)
        # clear pending
        self._pending_annotation = None
        self.text_edit.clear()

    def _export_json(self) -> None:
        """Export the current image annotations to a JSON file."""
        if not self.image_path or not self.annotations:
            QMessageBox.warning(self, "Warnung", "Keine Annotationen zum Exportieren.")
            return
        dataset = [
            {
                "file_name": os.path.basename(self.image_path),
                "annotations": [
                    {
                        "text": ann["text"],
                        "box": ann["box"],
                        "label": ann["label"],
                    }
                    for ann in self.annotations
                ],
            }
        ]
        # save file
        outfile, _ = QFileDialog.getSaveFileName(
            self,
            "JSON-Datei speichern",
            "annotations.json",
            "JSON Files (*.json);;All Files (*)",
        )
        if outfile:
            try:
                with open(outfile, "w", encoding="utf-8") as f:
                    json.dump(dataset, f, indent=2, ensure_ascii=False)
                QMessageBox.information(self, "Erfolg", f"Annotationen wurden nach {outfile} exportiert.")
            except Exception as exc:
                QMessageBox.critical(self, "Fehler", f"Konnte Datei nicht speichern: {exc}")