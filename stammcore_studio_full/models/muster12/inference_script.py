"""Reference inference script imported from the Muster12 repository.

This file contains a simplified copy of the inference logic used by the
third‑party Muster12 model.  It defines the label list and provides
placeholder code for loading the trained LayoutLMv3 model and
running inference.  The actual weights and configuration files are
not included in this repository; to use this script you must place
the corresponding ``layoutlmv3_final_model`` directory (containing
``config.json``, ``pytorch_model.bin`` and ``preprocessor``) in the
same folder as this script.

The canonical label list for this model is reproduced here for
reference.  See the official Muster12 repository for the full
implementation.
"""

import torch
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from paddleocr import PaddleOCR

# Canonical label list extracted from Muster12
label_list = [
    'O',
    'arzt_nr',
    'betriebsstaetten_nr',
    'birth_date',
    'datum',
    'doctor_address',
    'doctor_city',
    'doctor_name',
    'first_name',
    'icd_codes',
    'kostentraeger_kennzeichen',
    'krankenkasse',
    'last_name',
    'patient_address_city',
    'patient_address_street',
    'status',
    'versicherten_nr',
]

id2label = {id: label for id, label in enumerate(label_list)}
label2id = {label: id for id, label in id2label.items()}


def load_model(model_path: str):
    """Load the model and processor from the given directory."""
    processor = LayoutLMv3Processor.from_pretrained(model_path, apply_ocr=False)
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        model_path,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )
    return processor, model


def run_inference(model_dir: str, image_path: str):
    """Placeholder function for running inference using the Muster12 model.

    This function demonstrates how one would load the processor and model,
    initialise PaddleOCR and perform inference on a single image.  The
    actual inference logic, including token-to-word alignment and
    confidence filtering, should follow the pattern implemented in
    ``core/inference.py`` of the StammCore Studio.
    """
    processor, model = load_model(model_dir)
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    # Initialise OCR
    ocr = PaddleOCR(use_angle_cls=False, lang='de', use_gpu=torch.cuda.is_available())
    # Load and process image, run OCR and model inference...
    raise NotImplementedError("This is a reference script; see core.inference.run_inference for implementation.")