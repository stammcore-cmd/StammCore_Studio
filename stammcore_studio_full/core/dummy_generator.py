"""Functions to generate synthetic form images or PDFs from example data.

The dummy generator allows users to generate large numbers of realistic
filled forms based on a single template (PNG/PDF) and an Excel file
containing example records.  This is useful for training and testing
document understanding models without exposing any personal data.

The generator operates entirely locally.  It reads the template image
using PIL and writes each generated document back to disk.  If a
template PDF is provided the first page is rasterised and used as
background.  When the output file ends with ``.pdf`` the PIL Image
save method is used to write a PDF (no external libraries required).
"""

from __future__ import annotations

import os
import random
import string
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union, Callable

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# Type for the callback used to stream log messages during generation
LogCallback = Callable[[str], None]


def _load_template(template_path: str) -> Image.Image:
    """Load a template image.  Supports PNG/JPEG directly.  For PDFs
    only the first page is rasterised via PIL's PDF support.

    Parameters
    ----------
    template_path: str
        Path to the template image or PDF.
    Returns
    -------
    Image.Image
        PIL image of the template.
    """
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template not found: {template_path}")
    template = Image.open(template_path)
    # For multi-page PDFs only use first page
    if getattr(template, "n_frames", 1) > 1:
        template.seek(0)
    return template.convert("RGB")


def _default_positions(columns: List[str], template_size: Tuple[int, int]) -> Dict[str, Tuple[int, int]]:
    """Compute default positions for each column on the template.

    This simple heuristic lays out fields vertically with a fixed
    offset.  Users should adjust positions manually if more control
    is required.
    """
    width, height = template_size
    x_start = int(width * 0.05)
    y_start = int(height * 0.2)
    y_step = int((height * 0.6) / max(1, len(columns)))
    positions = {}
    for i, col in enumerate(columns):
        positions[col] = (x_start, y_start + i * y_step)
    return positions


def generate_dummy_documents(
    excel_file: str,
    template_file: str,
    output_dir: str,
    positions: Optional[Dict[str, Tuple[int, int]]] = None,
    font_path: Optional[str] = None,
    font_size: int = 12,
    variation: bool = True,
    log_callback: Optional[LogCallback] = None,
) -> None:
    """Generate dummy filled documents from the given template and Excel data.

    Parameters
    ----------
    excel_file: str
        Path to an Excel file (.xlsx) with one row per dummy document.  Each
        column corresponds to a field name.  The header row is required.
    template_file: str
        Path to a template image (PNG, JPEG) or PDF (first page used).
    output_dir: str
        Directory where generated documents are saved.  Will be created if
        it does not exist.  Generated filenames take the form
        ``dummy_<row>.png`` or ``dummy_<row>.pdf`` depending on the
        template extension.
    positions: dict[str, tuple[int,int]], optional
        Mapping from column names to (x,y) positions in pixels where
        text should be drawn.  If omitted a simple vertical layout is
        computed automatically based on template size.  Coordinates are
        relative to the top-left of the template.
    font_path: str, optional
        Path to a .ttf font file used for drawing text.  If omitted
        PIL's default font is used.
    font_size: int, default=12
        Size of the font used to draw text.
    variation: bool, default=True
        When True each row will be slightly perturbed in position
        (±2 pixels) to simulate human filled forms.
    log_callback: callable, optional
        Callback used to emit log messages.  If provided the function
        will be called with a single string argument for each log
        message.
    """
    def log(msg: str) -> None:
        if log_callback:
            log_callback(msg)
        else:
            print(msg)

    df = pd.read_excel(excel_file, engine="openpyxl")
    if df.empty:
        log("No data found in the Excel file.")
        return
    template = _load_template(template_file)
    # Determine output format
    ext = os.path.splitext(template_file)[1].lower()
    use_pdf_output = ext == ".pdf"
    # Compute default positions if none provided
    cols = list(df.columns)
    pos = positions or _default_positions(cols, template.size)
    # Load font
    if font_path and os.path.exists(font_path):
        try:
            font = ImageFont.truetype(font_path, font_size)
        except Exception:
            font = ImageFont.load_default()
            log(f"Could not load font '{font_path}', falling back to default.")
    else:
        font = ImageFont.load_default()
    os.makedirs(output_dir, exist_ok=True)
    # Iterate over rows and generate documents
    for idx, row in df.iterrows():
        img = template.copy()
        draw = ImageDraw.Draw(img)
        for col, value in row.items():
            text = str(value) if not pd.isna(value) else ""
            if not text:
                continue
            x, y = pos.get(col, (50, 50))
            dx = random.randint(-2, 2) if variation else 0
            dy = random.randint(-2, 2) if variation else 0
            draw.text((x + dx, y + dy), text, font=font, fill=(0, 0, 0))
        # Construct filename
        out_filename = f"dummy_{idx+1}.{ 'pdf' if use_pdf_output else 'png' }"
        out_path = os.path.join(output_dir, out_filename)
        if use_pdf_output:
            # PIL can save a single image as PDF
            img.save(out_path, format="PDF")
        else:
            img.save(out_path)
        log(f"Generated {out_path}")