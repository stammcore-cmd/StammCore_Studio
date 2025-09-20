"""Inference pipeline for extracting entities from a scanned form.

This module exposes a function to run OCR and a LayoutLMv3 model on a
single image.  It performs lazy imports for heavy dependencies and
supports streaming log messages via a callback.  The returned
dictionary maps entity labels to lists of strings extracted from the
document.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Callable

LogCallback = Callable[[str], None]


def run_inference(
    model_dir: str,
    image_path: str,
    log_callback: Optional[LogCallback] = None,
) -> Dict[str, List[str]]:
    """Run OCR and LayoutLMv3 model inference on a single image.

    Parameters
    ----------
    model_dir: str
        Directory containing a fine‑tuned LayoutLMv3 model.  Must include
        the files ``config.json`` and ``pytorch_model.bin`` as well as
        the ``processor`` files.
    image_path: str
        Path to an image file (.png/.jpg) to process.
    log_callback: callable, optional
        Callback for streaming log messages.  Called with a single
        string argument.  If omitted logs are printed to stdout.

    Returns
    -------
    dict
        Mapping of entity labels to lists of extracted words.
    """
    def log(msg: str) -> None:
        if log_callback:
            log_callback(msg)
        else:
            print(msg)

    try:
        import torch
        from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
        from PIL import Image
        import numpy as np
        from paddleocr import PaddleOCR
    except ImportError as exc:
        log(f"Missing dependency during inference: {exc}")
        raise

    # Validate paths
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load processor and model
    processor = LayoutLMv3Processor.from_pretrained(model_dir, apply_ocr=False)
    model = LayoutLMv3ForTokenClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    # Initialize OCR (prefer GPU if available) and set reasonable confidence filtering.
    # We wrap OCR initialisation in a try/catch because different versions of
    # PaddleOCR expose slightly different arguments.  When GPU is available we
    # enable it to speed up inference; otherwise CPU is used.  The
    # ``drop_score`` parameter filters out low confidence recognition results
    # before they reach our own threshold.  According to the PaddleOCR
    # documentation the default drop_score is 0.5【753011927532575†L556-L566】, which is a more
    # conservative default than the 0.1 used previously.  We will leave
    # drop_score at its default and apply our own threshold below.
    ocr = None
    try:
        use_gpu = torch.cuda.is_available()
        ocr = PaddleOCR(
            use_angle_cls=False,
            use_gpu=use_gpu,
            lang="de",
        )
    except Exception:
        # Fallback to CPU-only OCR without specifying language
        try:
            ocr = PaddleOCR(
                use_angle_cls=False,
                use_gpu=False,
                lang="en",
            )
        except Exception:
            ocr = None

    # Load image
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    # Run OCR.  We use a two‑stage approach: first try the ``predict`` API
    # available in recent PaddleOCR versions.  If that fails or returns no
    # recognised text we fall back to the more stable ``ocr`` method.  The
    # recognition results are filtered with a configurable confidence
    # threshold (default 0.5), as recommended by the PaddleOCR docs【753011927532575†L556-L566】.
    confidence_threshold: float = 0.5
    words: List[str] = []
    boxes: List[List[int]] = []
    if ocr is not None:
        try:
            result = ocr.predict(np.array(image))
        except Exception:
            result = None
        texts: List[str] = []
        scores: List[float] = []
        polys: List[List[List[float]]] = []
        if result and isinstance(result, list) and result and result[0].get("rec_texts"):
            # New API format: dict with rec_texts, rec_scores, rec_polys
            texts = result[0].get("rec_texts", [])
            scores = result[0].get("rec_scores", [])
            polys = result[0].get("rec_polys", [])
        else:
            # Fallback to ocr.ocr which returns a list of lines with poly & (text, score)
            try:
                ocr_out = ocr.ocr(np.array(image), cls=False)
                if ocr_out:
                    lines = ocr_out[0]  # type: ignore[index]
                    for line in lines:
                        poly, (txt, score) = line
                        texts.append(txt or "")
                        scores.append(float(score) if score is not None else 0.0)
                        polys.append(poly)
            except Exception:
                pass
        # Convert OCR results to words and bounding boxes
        for text, score, poly in zip(texts, scores, polys):
            if not text or not text.strip() or score < confidence_threshold:
                continue
            # compute bounding box in normalized coordinates
            x_coords = [float(p[0]) for p in poly]
            y_coords = [float(p[1]) for p in poly]
            x0, y0, x1, y1 = min(x_coords), min(y_coords), max(x_coords), max(y_coords)
            bbox = [
                max(0, min(1000, int(1000 * x0 / width))),
                max(0, min(1000, int(1000 * y0 / height))),
                max(0, min(1000, int(1000 * x1 / width))),
                max(0, min(1000, int(1000 * y1 / height))),
            ]
            # avoid zero area boxes
            if bbox[2] == bbox[0]:
                bbox[2] = min(1000, bbox[0] + 1)
            if bbox[3] == bbox[1]:
                bbox[3] = min(1000, bbox[1] + 1)
            words.append(text.strip())
            boxes.append(bbox)
    if not words:
        log("No words detected in image using OCR.")
        return {}

    # Prepare model input.  We need to retain the ``BatchEncoding`` instance
    # returned by the processor in order to access the ``word_ids`` method
    # later.  Converting the encoding to a plain dict would drop that
    # method and force us into a fallback alignment strategy.  Therefore we
    # keep the full encoding, extract the mapping, then move tensors to
    # the selected device.
    batch_encoding = processor(
        image,
        words,
        boxes=boxes,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt",
    )
    # Attempt to retrieve the word id mapping before converting to a dict
    try:
        word_id_mapping = batch_encoding.word_ids(batch_index=0)  # type: ignore[call-arg]
    except Exception:
        word_id_mapping = None
    # Move tensors to the selected device
    encoding_tensors = {k: v.to(device) for k, v in batch_encoding.items()}
    # Run model inference
    with torch.no_grad():
        outputs = model(**encoding_tensors)
        logits = outputs.logits  # shape: (1, seq_len, num_labels)
        pred_ids: List[int] = logits.argmax(-1).cpu().numpy()[0].tolist()

    # Determine id2label mapping
    id2label = getattr(model.config, "id2label", None)
    if not id2label:
        unique_ids = sorted(set(pred_ids))
        id2label = {i: f"LABEL_{i}" for i in unique_ids}

    # Map predictions to words.  We align tokens to words using the
    # ``word_id_mapping`` if available.  For each word we assign the label
    # predicted for its first subtoken; special tokens and subsequent
    # subwords are ignored.  See HuggingFace docs for details on token
    # alignment【177870977359215†L299-L329】.
    entities: Dict[str, List[str]] = {}
    if word_id_mapping:
        word_to_label: Dict[int, int] = {}
        seen: set[int] = set()
        for tok_idx, w_id in enumerate(word_id_mapping):
            # skip special tokens and already processed words
            if w_id is None or w_id in seen:
                continue
            seen.add(w_id)
            if tok_idx < len(pred_ids):
                word_to_label[w_id] = pred_ids[tok_idx]
        for w_idx, word in enumerate(words):
            label_id = word_to_label.get(w_idx)
            if label_id is None:
                continue
            label = id2label.get(label_id, str(label_id))
            # skip 'O' (outside) label
            if label == "O":
                continue
            entities.setdefault(label, []).append(word)
    else:
        # Fallback: naive one-to-one mapping
        for word, pred_id in zip(words, pred_ids):
            label = id2label.get(pred_id, str(pred_id))
            if label == "O":
                continue
            entities.setdefault(label, []).append(word)
    return entities