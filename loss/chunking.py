"""Deterministic, discourse-aware chunking for ChunkRPO.

Implements the chunker C(x, y) described in Section 3.1 of the ChunkRPO paper:
given a response (its text and the token spans the tokenizer assigned to it), it
returns a *unique* partition of the response tokens into contiguous chunks. The
partition is a function of the text alone -- it does not depend on the policy
pi_theta -- and is intended to be computed once per example and frozen for the
whole training run.

Design notes
------------
* Boundaries are placed at *natural discourse units* using lightweight,
  parser-free cues: sentence/clause terminators, discourse connectives, list and
  enumeration markers, and structural delimiters (line breaks, Markdown headers,
  code fences). When the text is code or chain-of-thought, the same delimiters
  (line breaks / fences / numbered steps) act as step boundaries.
* Two regularizers keep chunks numerically comparable (cf. the paper's
  min/max-length rule):
    - ``min_chunk_tokens``: adjacent sub-floor fragments are merged forward;
    - ``max_chunk_tokens``: over-long chunks are split into pieces of at most
      this many tokens.
* Every boundary is *snapped to a token boundary* consistent with the model
  tokenizer by operating on the tokenizer's character offset mapping, so the
  emitted spans ``c_i = (a_i, ..., b_i)`` index exactly into the response tokens.

The output of :func:`chunk_response` is a list of ``(start, end)`` half-open
token-index intervals over the *response* tokens (index 0 == first response
token). The caller is responsible for shifting these onto the full
prompt+response sequence and turning them into a per-token chunk-id array.
"""

from __future__ import annotations

import re
from typing import List, Sequence, Tuple

# A boundary is allowed to *end* a chunk when the character immediately at/after
# the cue marks a discourse unit. We detect cues on the decoded response text and
# then map character positions back to token indices via the offset mapping.

# Sentence / clause terminators (followed by whitespace or end-of-text).
_SENTENCE_END = re.compile(r"[.!?。！？;:](?=\s|$)")

# Structural delimiters: one or more newlines (paragraph / line / list breaks).
_NEWLINE = re.compile(r"\n+")

# Markdown / code-fence structural markers at a line start.
_FENCE = re.compile(r"^\s*```")
_MD_HEADER = re.compile(r"^\s*#{1,6}\s")

# List / enumeration markers at a line start: "1.", "1)", "-", "*", "+".
_LIST_MARKER = re.compile(r"^\s*(?:\d+[.)]|[-*+])\s")

# Discourse connectives that typically *begin* a new discourse unit. We cut
# *before* these when they start a clause (preceded by a comma/space, lowercased
# match on a word boundary). Kept short and high-precision to avoid over-cutting.
_CONNECTIVES = (
    "however",
    "therefore",
    "moreover",
    "furthermore",
    "consequently",
    "nevertheless",
    "nonetheless",
    "meanwhile",
    "additionally",
    "thus",
    "hence",
    "instead",
    "otherwise",
    "finally",
    "first",
    "second",
    "third",
    "next",
    "then",
)
_CONNECTIVE_RE = re.compile(
    r"(?:(?<=\.\s)|(?<=,\s)|(?<=\n))(?:" + "|".join(_CONNECTIVES) + r")\b",
    re.IGNORECASE,
)


def _char_boundaries(text: str) -> List[int]:
    """Return a sorted list of *character* offsets at which a chunk may end.

    Each returned offset ``p`` means "a chunk boundary sits immediately before
    character ``p``" (i.e. characters ``[prev, p)`` form a candidate unit). The
    set always implicitly includes 0 and ``len(text)``; we return the interior
    cut points only.
    """
    cuts = set()

    # Sentence/clause terminators: cut *after* the terminator (and any following
    # run of whitespace is left to the next unit's leading trim downstream).
    for m in _SENTENCE_END.finditer(text):
        cuts.add(m.end())

    # Newlines / paragraph breaks: cut after the newline run.
    for m in _NEWLINE.finditer(text):
        cuts.add(m.end())

    # Per-line structural markers (fences, headers, list items): cut *before* the
    # line so the marker starts a fresh unit.
    pos = 0
    for line in text.splitlines(keepends=True):
        stripped_start = pos
        if (
            _FENCE.match(line)
            or _MD_HEADER.match(line)
            or _LIST_MARKER.match(line)
        ):
            if stripped_start > 0:
                cuts.add(stripped_start)
        pos += len(line)

    # Discourse connectives that begin a new clause: cut *before* the connective.
    for m in _CONNECTIVE_RE.finditer(text):
        if m.start() > 0:
            cuts.add(m.start())

    cuts.discard(0)
    cuts.discard(len(text))
    return sorted(cuts)


def _char_cuts_to_token_cuts(
    char_cuts: Sequence[int],
    offsets: Sequence[Tuple[int, int]],
) -> List[int]:
    """Snap character cut points to token boundaries.

    ``offsets[k] = (c_start, c_end)`` is the character span of response token
    ``k`` (relative to the response text). A character cut at position ``p``
    becomes a *token* cut before the first token whose start offset is ``>= p``.
    Returns a sorted list of unique token indices in ``(0, n_tokens)``.
    """
    n = len(offsets)
    if n == 0:
        return []
    # token start offsets; some fast tokenizers emit (0,0) for special tokens --
    # those collapse onto neighbours and are harmless here.
    starts = [o[0] for o in offsets]
    token_cuts = set()
    j = 0
    for p in char_cuts:
        # advance to the first token starting at/after p
        while j < n and starts[j] < p:
            j += 1
        if 0 < j < n:
            token_cuts.add(j)
        if j >= n:
            break
    return sorted(token_cuts)


def _segments_from_cuts(n_tokens: int, token_cuts: Sequence[int]) -> List[Tuple[int, int]]:
    """Turn interior token cut indices into half-open ``(start, end)`` segments."""
    bounds = [0] + list(token_cuts) + [n_tokens]
    segs = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b > a:
            segs.append((a, b))
    return segs


def _apply_length_regularizers(
    segs: List[Tuple[int, int]],
    min_chunk_tokens: int,
    max_chunk_tokens: int,
) -> List[Tuple[int, int]]:
    """Merge sub-floor fragments forward, then split over-long chunks."""
    if not segs:
        return segs

    # ---- merge: fold fragments shorter than the floor into the previous chunk;
    # if the very first chunk is too short, fold it into the next one instead.
    merged: List[Tuple[int, int]] = []
    for start, end in segs:
        if merged and (end - start) < min_chunk_tokens:
            ps, _ = merged[-1]
            merged[-1] = (ps, end)
        else:
            merged.append((start, end))
    # handle a too-short leading chunk (no previous to merge into at append time)
    if len(merged) >= 2 and (merged[0][1] - merged[0][0]) < min_chunk_tokens:
        s0, _ = merged[0]
        _, e1 = merged[1]
        merged = [(s0, e1)] + merged[2:]

    # ---- split: break chunks longer than the ceiling into <= max pieces.
    if max_chunk_tokens and max_chunk_tokens > 0:
        out: List[Tuple[int, int]] = []
        for start, end in merged:
            length = end - start
            if length <= max_chunk_tokens:
                out.append((start, end))
                continue
            # even-ish split into ceil(length / max) pieces
            n_pieces = (length + max_chunk_tokens - 1) // max_chunk_tokens
            base = length // n_pieces
            rem = length % n_pieces
            cur = start
            for k in range(n_pieces):
                step = base + (1 if k < rem else 0)
                out.append((cur, cur + step))
                cur += step
        merged = out

    return merged


def chunk_response(
    text: str,
    offsets: Sequence[Tuple[int, int]],
    min_chunk_tokens: int = 4,
    max_chunk_tokens: int = 64,
) -> List[Tuple[int, int]]:
    """Deterministically partition a response into discourse-aware chunks.

    Args:
        text: the decoded *response* text (assistant turn only, no prompt).
        offsets: per-token ``(char_start, char_end)`` offsets for the response
            tokens, as returned by a fast tokenizer's ``return_offsets_mapping``.
            ``len(offsets)`` is the number of response tokens.
        min_chunk_tokens: minimum chunk size; shorter fragments are merged.
        max_chunk_tokens: maximum chunk size; longer chunks are split. Set to 0
            to disable splitting.

    Returns:
        A list of half-open ``(start, end)`` intervals over the response token
        indices that exactly tile ``[0, len(offsets))``.
    """
    n_tokens = len(offsets)
    if n_tokens == 0:
        return []
    if n_tokens <= min_chunk_tokens:
        # too short to partition meaningfully -> single chunk
        return [(0, n_tokens)]

    char_cuts = _char_boundaries(text)
    token_cuts = _char_cuts_to_token_cuts(char_cuts, offsets)
    segs = _segments_from_cuts(n_tokens, token_cuts)
    segs = _apply_length_regularizers(segs, min_chunk_tokens, max_chunk_tokens)
    # safety: guarantee a non-empty tiling
    if not segs:
        segs = [(0, n_tokens)]
    return segs
