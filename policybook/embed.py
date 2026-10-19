"""Local CPU embedding encoder: vendored quantized all-MiniLM-L6-v2 (ONNX).

Why vendored: the reviewer's reproduction environment restricts egress to a
metered gateway (EVALUATION_PROTOCOL.md), so the model ships inside the repo
under models/all-MiniLM-L6-v2 and reproduce.sh downloads nothing. Quantized
ONNX keeps the repo small (~23 MB) and needs no torch install. Same model
family the approach named (bge-small / MiniLM); MiniLM won on artifact size.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

MODEL_NAME = "all-MiniLM-L6-v2 (quantized ONNX, Xenova export, vendored)"


class MiniLMEncoder:
    def __init__(self, model_dir, max_tokens: int = 256):
        model_dir = Path(model_dir)
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=max_tokens)
        self.tokenizer.enable_padding()
        self.session = ort.InferenceSession(
            str(model_dir / "onnx" / "model_quantized.onnx"),
            providers=["CPUExecutionProvider"])
        self._inputs = {i.name for i in self.session.get_inputs()}
        self.dim = 384

    def encode(self, texts, batch_size: int = 32) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            encs = self.tokenizer.encode_batch(list(texts[i:i + batch_size]))
            ids = np.array([e.ids for e in encs], dtype=np.int64)
            mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
            feeds = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self._inputs:
                feeds["token_type_ids"] = np.zeros_like(ids)
            hidden = self.session.run(None, feeds)[0]  # [B, T, 384]
            m = mask[..., None].astype(np.float32)
            pooled = (hidden * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)
            norm = np.linalg.norm(pooled, axis=1, keepdims=True)
            out.append(pooled / np.clip(norm, 1e-12, None))
        return np.vstack(out).astype(np.float32)
