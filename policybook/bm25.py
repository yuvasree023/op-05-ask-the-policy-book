"""Okapi BM25 over in-memory chunk texts (stdlib only).

The corpus is ~400 chunks, so a from-scratch implementation beats adding a
dependency: no install weight, no version surface, and the ranking code is
small enough to explain line by line in review.
"""
from __future__ import annotations

import math
import re

TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list:
    return TOKEN.findall(text.lower())


class BM25:
    def __init__(self, texts, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(texts)
        self.tf = []   # per-doc {term: count}
        self.dl = []   # doc lengths
        df = {}
        for text in texts:
            counts = {}
            for tok in tokenize(text):
                counts[tok] = counts.get(tok, 0) + 1
            self.tf.append(counts)
            self.dl.append(sum(counts.values()))
            for tok in counts:
                df[tok] = df.get(tok, 0) + 1
        self.avgdl = sum(self.dl) / self.n if self.n else 0.0
        self.idf = {t: math.log(1 + (self.n - c + 0.5) / (c + 0.5))
                    for t, c in df.items()}

    def scores(self, query: str) -> list:
        out = [0.0] * self.n
        for tok in set(tokenize(query)):
            idf = self.idf.get(tok)
            if idf is None:
                continue
            for i in range(self.n):
                f = self.tf[i].get(tok)
                if not f:
                    continue
                norm = 1 - self.b + self.b * self.dl[i] / (self.avgdl or 1)
                out[i] += idf * f * (self.k1 + 1) / (f + self.k1 * norm)
        return out
