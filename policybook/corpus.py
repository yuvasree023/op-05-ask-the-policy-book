"""Read and validate an OP-05 corpus directory.

The corpus is an input, not a constant (see freshness_contract.md): a corpus
directory contains manifest.jsonl, passages.jsonl and the referenced text
files in the published formats. Nothing here is tied to the shipped Season-1
bytes; any directory in the same format (e.g. the synthetic update pack)
loads the same way.

Authoritative quote source: the text/<doc_id>.txt bytes, exactly as shipped.
The grader's verbatim checker reads those files, so chunk text must remain an
exact substring of them.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

ROLES = ("public", "ops", "legal")  # public < ops < legal, in that access order
SECTION_MARKER = re.compile(r"^\[([A-Za-z0-9._-]+#s\d+)\]\s*$")
DOC_ID = re.compile(r"^[A-Za-z0-9._-]+$")
SHA256 = re.compile(r"^[a-f0-9]{64}$")
REQUIRED_FIELDS = ("doc_id", "title", "source_url", "effective_date",
                   "supersedes", "role", "text_path", "sha256")


class CorpusError(ValueError):
    """The corpus directory does not follow the published format."""


@dataclass
class DocMeta:
    doc_id: str
    title: str
    source_url: str
    effective_date: str
    supersedes: list
    role: str
    text_path: str
    sha256: str


@dataclass
class Section:
    passage_id: str  # "DOC#s00001"; this is the citation "section" field
    doc_id: str
    text: str        # exact substring of the shipped text file


@dataclass
class Corpus:
    root: Path
    docs: dict
    sections: list
    corpus_hash: str
    validation: list = field(default_factory=list)


def _parse_sections(doc_id: str, text: str, validation: list) -> list:
    """Split a text file on its [DOC#sNNNNN] markers, keeping exact text."""
    sections, current, buf = [], None, []
    preamble = []

    def flush():
        if current is None:
            return
        body = "\n".join(buf).strip("\n")
        if not body.strip():
            validation.append(f"{doc_id}: empty section {current}")
        if not current.startswith(doc_id + "#"):
            validation.append(
                f"{doc_id}: section id {current} does not start with doc id")
        sections.append(Section(passage_id=current, doc_id=doc_id, text=body))

    for line in text.split("\n"):
        m = SECTION_MARKER.match(line)
        if m:
            flush()
            current, buf = m.group(1), []
        elif current is None:
            preamble.append(line)
        else:
            buf.append(line)
    flush()
    if any(p.strip() for p in preamble):
        validation.append(f"{doc_id}: non-empty text before first section marker")
    return sections


def read_corpus(corpus_dir) -> Corpus:
    root = Path(corpus_dir)
    if not root.is_dir():
        raise CorpusError(f"corpus directory not found: {root}")
    manifest_path = root / "manifest.jsonl"
    if not manifest_path.is_file():
        raise CorpusError(
            f"{root} has no manifest.jsonl; the published corpus format is "
            "manifest.jsonl + passages.jsonl + text/<doc_id>.txt")
    manifest_bytes = manifest_path.read_bytes()

    docs, seen = {}, set()
    for lineno, raw in enumerate(manifest_bytes.decode("utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as e:
            raise CorpusError(f"manifest.jsonl line {lineno}: invalid JSON: {e}")
        missing = [k for k in REQUIRED_FIELDS if k not in rec]
        if missing:
            raise CorpusError(f"manifest.jsonl line {lineno}: missing {missing}")
        if not DOC_ID.match(rec["doc_id"]):
            raise CorpusError(f"manifest.jsonl line {lineno}: bad doc_id {rec['doc_id']!r}")
        if rec["role"] not in ROLES:
            raise CorpusError(f"manifest.jsonl line {lineno}: bad role {rec['role']!r}")
        if not SHA256.match(rec["sha256"]):
            raise CorpusError(f"manifest.jsonl line {lineno}: bad sha256")
        if not isinstance(rec["supersedes"], list):
            raise CorpusError(f"manifest.jsonl line {lineno}: supersedes must be a list")
        if rec["doc_id"] in seen:
            raise CorpusError(f"manifest.jsonl line {lineno}: duplicate doc_id")
        seen.add(rec["doc_id"])
        docs[rec["doc_id"]] = DocMeta(
            doc_id=rec["doc_id"], title=rec["title"], source_url=rec["source_url"],
            effective_date=rec["effective_date"], supersedes=list(rec["supersedes"]),
            role=rec["role"], text_path=rec["text_path"], sha256=rec["sha256"])
    if not docs:
        raise CorpusError("manifest.jsonl has no documents")

    validation, sections = [], []
    root_resolved = root.resolve()
    for doc_id, meta in docs.items():
        text_file = root / meta.text_path
        try:
            resolved = text_file.resolve()
            if not resolved.is_relative_to(root_resolved):
                raise CorpusError(f"{doc_id}: text_path escapes corpus dir")
        except FileNotFoundError:
            raise CorpusError(f"{doc_id}: text file missing: {meta.text_path}")
        if not text_file.is_file():
            raise CorpusError(f"{doc_id}: text file missing: {meta.text_path}")
        data = text_file.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != meta.sha256:
            raise CorpusError(
                f"{doc_id}: sha256 mismatch (manifest {meta.sha256[:12]}..., "
                f"file {digest[:12]}...); corpus bytes are corrupted or stale")
        sections.extend(_parse_sections(doc_id, data.decode("utf-8"), validation))

    # Cross-check against passages.jsonl when shipped. The text files stay
    # authoritative (the verbatim quote checker reads them); a mismatch is
    # reported loudly, not silently absorbed.
    passages_path = root / "passages.jsonl"
    if passages_path.is_file():
        pbytes = passages_path.read_bytes()
        expected = {}
        for raw in pbytes.decode("utf-8").splitlines():
            if raw.strip():
                rec = json.loads(raw)
                expected[rec["passage_id"]] = (rec["doc_id"], rec["text"])
        got = {s.passage_id: (s.doc_id, s.text) for s in sections}
        if set(expected) != set(got):
            only_e = sorted(set(expected) - set(got))[:5]
            only_g = sorted(set(got) - set(expected))[:5]
            validation.append(
                f"passage id mismatch: {len(set(expected) - set(got))} only in "
                f"passages.jsonl {only_e}, {len(set(got) - set(expected))} only "
                f"in parsed text {only_g}")
        text_mismatch = [pid for pid in set(expected) & set(got)
                         if expected[pid] != got[pid]]
        if text_mismatch:
            validation.append(
                f"passage text mismatch on {len(text_mismatch)} sections, "
                f"e.g. {text_mismatch[:5]}")

    h = hashlib.sha256()
    for line in sorted(manifest_bytes.splitlines()):
        if line.strip():
            h.update(line.strip() + b"\n")
    if passages_path.is_file():
        h.update(b"passages:" + hashlib.sha256(pbytes).hexdigest().encode())
    return Corpus(root=root, docs=docs, sections=sections,
                  corpus_hash=h.hexdigest(), validation=validation)
