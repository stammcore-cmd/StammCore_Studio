
# stammcore_studio_full/core/training.py
# -*- coding: utf-8 -*-
"""
Robuste Trainings-Pipeline für LayoutLMv3 mit Versionskompatibilität für
Hugging Face `transformers` **und** mit Backwards-Compat bei Funktionsparametern.

- Akzeptiert jetzt sowohl `dataset_json` (alt) als auch `dataset_json_path` (neu).
- Akzeptiert `num_epochs` alias `epochs` sowie weitere optionale Alias-Felder.
"""

from __future__ import annotations

import os
import json
import math
import inspect
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Any

from PIL import Image


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _normalize_box(box: Iterable[int], img_w: int, img_h: int) -> List[int]:
    x0, y0, x1, y1 = map(int, box)
    x0 = max(0, min(1000, int(1000 * x0 / max(1, img_w))))
    y0 = max(0, min(1000, int(1000 * y0 / max(1, img_h))))
    x1 = max(0, min(1000, int(1000 * x1 / max(1, img_w))))
    y1 = max(0, min(1000, int(1000 * y1 / max(1, img_h))))
    if x1 <= x0:
        x1 = min(1000, x0 + 1)
    if y1 <= y0:
        y1 = min(1000, y0 + 1)
    return [x0, y0, x1, y1]


def _collect_labels(dataset_items: List[Dict[str, Any]]) -> List[str]:
    labels = set(["O"])
    for item in dataset_items:
        for ann in item.get("annotations", []):
            lab = (ann.get("label") or "").strip()
            if lab:
                labels.add(lab)
    return sorted(labels)


def _build_training_args(output_dir: str,
                         batch_size: int,
                         epochs: int,
                         learning_rate: float,
                         warmup_ratio: float,
                         train_len: int) -> "TrainingArguments":
    from transformers import TrainingArguments
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except Exception:
        has_cuda = False

    import inspect as _inspect
    sig = _inspect.signature(TrainingArguments.__init__)
    allowed = set(sig.parameters.keys())

    steps_per_epoch = max(1, math.ceil(train_len / max(1, batch_size)))
    total_steps = int(steps_per_epoch * max(1, epochs))

    args = {
        "output_dir": output_dir,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "num_train_epochs": float(epochs),
        "learning_rate": float(learning_rate),
        "weight_decay": 0.01,
    }
    if "fp16" in allowed:
        args["fp16"] = bool(has_cuda)
    if "logging_steps" in allowed:
        args["logging_steps"] = max(1, steps_per_epoch // 5)
    if "save_steps" in allowed:
        args["save_steps"] = steps_per_epoch
    if "save_total_limit" in allowed:
        args["save_total_limit"] = 2

    if "evaluation_strategy" in allowed:
        args["evaluation_strategy"] = "epoch"
        if "save_strategy" in allowed:
            args["save_strategy"] = "epoch"
        if "load_best_model_at_end" in allowed:
            args["load_best_model_at_end"] = True
        if "metric_for_best_model" in allowed:
            args["metric_for_best_model"] = "f1"
    else:
        if "evaluate_during_training" in allowed:
            args["evaluate_during_training"] = True
        if "eval_steps" in allowed:
            args["eval_steps"] = steps_per_epoch

    if "warmup_ratio" in allowed:
        args["warmup_ratio"] = float(warmup_ratio)
    elif "warmup_steps" in allowed:
        args["warmup_steps"] = int(total_steps * float(warmup_ratio))

    if "report_to" in allowed:
        args["report_to"] = []

    return TrainingArguments(**args)


@dataclass
class JsonItem:
    file_name: str
    annotations: List[Dict[str, Any]]
    doc_type: Optional[str] = None


class FormDataset:
    def __init__(self,
                 items: List[JsonItem],
                 images_dir: str,
                 processor: "LayoutLMv3Processor",
                 label2id: Dict[str, int]):
        self.items = items
        self.images_dir = images_dir
        self.processor = processor
        self.label2id = label2id
        self.id2label = {v: k for k, v in label2id.items()}

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        image_path = os.path.join(self.images_dir, item.file_name)
        image = Image.open(image_path).convert("RGB")
        img_w, img_h = image.size

        words: List[str] = []
        boxes: List[List[int]] = []
        word_labels: List[int] = []

        for ann in item.annotations:
            text = (ann.get("text") or "").strip()
            if not text:
                continue
            box = ann.get("box") or [0, 0, img_w, img_h]
            nbox = _normalize_box(box, img_w, img_h)
            lab_id = self.label2id.get((ann.get("label") or "O").strip() or "O", self.label2id["O"])
            for w in text.split():
                words.append(w)
                boxes.append(nbox)
                word_labels.append(lab_id)

        if not words:
            words = ["[PAD]"]
            boxes = [[0, 0, 1, 1]]
            word_labels = [self.label2id["O"]]

        enc = self.processor(
            image,
            words,
            boxes=boxes,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        )

        # Label-Alignment über word_ids()
        word_ids = enc.word_ids(batch_index=0)
        labels = []
        for tok_i, w_id in enumerate(word_ids):
            if w_id is None:
                labels.append(-100)
            else:
                labels.append(int(word_labels[w_id]))

        item_out = {k: v.squeeze(0) for k, v in enc.items()}
        import torch
        item_out["labels"] = torch.tensor(labels, dtype=torch.long)
        return item_out


def _compute_metrics_builder(id2label: Dict[int, str]):
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    def _compute_metrics(eval_pred) -> Dict[str, float]:
        logits, labels = eval_pred
        import numpy as np
        preds = np.argmax(logits, axis=-1)
        true_labels: List[str] = []
        true_preds:  List[str] = []
        for pred_seq, lab_seq in zip(preds, labels):
            for p, l in zip(pred_seq, lab_seq):
                if l == -100:
                    continue
                true_labels.append(id2label.get(int(l), "O"))
                true_preds.append(id2label.get(int(p), "O"))
        if not true_labels:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, true_preds, average="weighted", zero_division=0
        )
        acc = accuracy_score(true_labels, true_preds)
        return {"precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "accuracy": float(acc)}
    return _compute_metrics


def train_layoutlm_model(
    dataset_json_path: Optional[str] = None,
    images_dir: str = "",
    output_dir: str = "",
    doc_type: Optional[str] = None,
    epochs: int = 3,
    batch_size: int = 2,
    learning_rate: float = 5e-5,
    warmup_ratio: float = 0.1,
    log_callback: Optional[Callable[[str], None]] = None,
    stop_flag: Optional["threading.Event"] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Führt ein LayoutLMv3-Training aus.

    Backwards-Compat:
      - `dataset_json` (alt) oder `dataset_json_path` (neu)
      - `num_epochs` (alt) alias `epochs`
    """
    # ---- Aliase/Back-Compat ----
    if dataset_json_path is None:
        dataset_json_path = kwargs.pop("dataset_json", None) or kwargs.pop("dataset_path", None)
    epochs = int(kwargs.pop("num_epochs", epochs))
    # optional: alternative LR-Feldnamen tolerieren
    learning_rate = float(kwargs.pop("lr", learning_rate))

    if not dataset_json_path:
        raise TypeError("train_layoutlm_model(): Missing required dataset_json_path (oder dataset_json).")
    if not images_dir or not output_dir:
        raise TypeError("train_layoutlm_model(): images_dir und output_dir sind erforderlich.")

    _ensure_dir(output_dir)

    # Lazy Imports
    import random
    try:
        import transformers as hf
        from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor, Trainer
    except Exception as e:
        raise RuntimeError("transformers nicht installiert oder inkompatibel: " + str(e))

    _log(f"Transformers-Version: {hf.__version__}", log_callback)

    # Daten laden
    with open(dataset_json_path, "r", encoding="utf-8") as f:
        raw_items = json.load(f)
    items: List[JsonItem] = []
    for it in raw_items:
        items.append(JsonItem(
            file_name=it.get("file_name") or it.get("image") or "",
            annotations=it.get("annotations", []),
            doc_type=it.get("doc_type"),
        ))
    if not items:
        raise RuntimeError("Dataset JSON enthält keine Items.")

    labels = _collect_labels([{"annotations": it.annotations} for it in items])
    label2id = {lab: i for i, lab in enumerate(labels)}
    id2label = {i: lab for lab, i in label2id.items()}

    processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        "microsoft/layoutlmv3-base",
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    # Split 85/15
    rng = random.Random(42)
    idxs = list(range(len(items)))
    rng.shuffle(idxs)
    split = max(1, int(0.85 * len(items)))
    train_items = [items[i] for i in idxs[:split]]
    val_items   = [items[i] for i in idxs[split:]]

    train_ds = FormDataset(train_items, images_dir, processor, label2id)
    val_ds   = FormDataset(val_items,   images_dir, processor, label2id)

    # TrainingArguments (versionssicher)
    training_args = _build_training_args(
        output_dir=output_dir,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        train_len=len(train_ds),
    )

    compute_metrics = _compute_metrics_builder(id2label)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=processor,
        compute_metrics=compute_metrics,
    )

    # Train & final evaluate
    try:
        trainer.train()
        metrics = trainer.evaluate()
    except Exception as e:
        _log(f"Fehler während des Trainings: {e}", log_callback)
        return {"metrics": {}, "model_dir": ""}

    f1 = float(metrics.get("eval_f1") or metrics.get("f1") or 0.0)
    precision = float(metrics.get("eval_precision") or metrics.get("precision") or 0.0)
    recall = float(metrics.get("eval_recall") or metrics.get("recall") or 0.0)
    accuracy = float(metrics.get("eval_accuracy") or metrics.get("accuracy") or 0.0)

    model_dir = os.path.join(output_dir, "layoutlmv3_final_model")
    _ensure_dir(model_dir)
    try:
        trainer.save_model(model_dir)
        processor.save_pretrained(model_dir)
    except Exception as e:
        _log(f"Warnung: Modell konnte nicht vollständig gespeichert werden: {e}", log_callback)

    metrics_payload = {
        "f1": f1, "precision": precision, "recall": recall, "accuracy": accuracy,
        "doc_type": doc_type or "", "dataset": os.path.basename(dataset_json_path),
    }
    try:
        with open(os.path.join(model_dir, "metrics.json"), "w", encoding="utf-8") as mf:
            json.dump(metrics_payload, mf, indent=2, ensure_ascii=False)
    except Exception as e:
        _log(f"Warnung: metrics.json konnte nicht geschrieben werden: {e}", log_callback)

    return {"metrics": metrics_payload, "model_dir": model_dir}
