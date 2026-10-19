"""Split corpus sections into embedding-sized chunks without losing provenance.

Chunk boundaries never rewrite bytes: every chunk's text is an exact substring
of the source file, so a later citation quote can be checked verbatim. The
citation "section" (passage id) survives splitting via the part counter.

all-MiniLM-L6-v2 truncates at 256 word-pieces, so chunks target ~1000 chars.
"""
from __future__ import annotations

from dataclasses import dataclass

MAX_CHARS = 1000


@dataclass
class Chunk:
    chunk_id: str    # passage_id, or passage_id + "#pNN" when a section splits
    passage_id: str  # citation "section" field
    doc_id: str
    part: int
    text: str        # exact substring of the shipped text file


def _cut_points(text: str, max_chars: int) -> list:
    """Offset-based split; returned parts are exact substrings of text."""
    if len(text) <= max_chars:
        return [text]
    parts, start = [], 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            window = text[start:end]
            floor = max_chars // 2
            cut = window.rfind("\n\n", floor)
            if cut == -1:
                cut = window.rfind(". ", floor)
                if cut != -1:
                    cut += 1  # keep the period with its sentence
            if cut == -1:
                cut = window.rfind(" ", floor)
            if cut == -1:
                cut = max_chars  # no boundary at all: hard cut
            end = start + cut
        part = text[start:end].strip("\n ")
        if part:
            parts.append(part)
        start = end
        while start < len(text) and text[start] in "\n ":
            start += 1
    return parts


def chunk_section(section, max_chars: int = MAX_CHARS) -> list:
    pieces = _cut_points(section.text, max_chars)
    if len(pieces) == 1:
        return [Chunk(chunk_id=section.passage_id, passage_id=section.passage_id,
                      doc_id=section.doc_id, part=1, text=pieces[0])]
    return [Chunk(chunk_id=f"{section.passage_id}#p{i:02d}",
                  passage_id=section.passage_id, doc_id=section.doc_id,
                  part=i + 1, text=piece)
            for i, piece in enumerate(pieces)]
