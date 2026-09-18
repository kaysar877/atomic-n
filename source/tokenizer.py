"""Byte-level BPE tokenizer (GPT-style), trained purely from the buffer corpus.

Byte-level = lossless: ``decode(encode(s)) == s`` for ANY string, so no
tokenizer can produce unknown-token holes. Compact enough to be trained
locally on a few MB of text in seconds.
"""
from __future__ import annotations

import base64
import json
import re
from collections import defaultdict

GPT2_PAT = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?\d+| ?[^\sA-Za-z\d]+|\s+(?!\S)|\s+"""
)


class ByteLevelBPE:
    def __init__(self, vocab_size: int = 4096):
        self.base = 256
        self.vocab_size = max(vocab_size, self.base + 1)
        self.merges: dict[tuple[int, int], int] = {}
        self.ids: dict[bytes, int] = {bytes([i]): i for i in range(self.base)}
        self.chunks: dict[int, bytes] = {i: bytes([i]) for i in range(self.base)}
        self.next_id = self.base

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def _alloc(self, chunk: bytes) -> int:
        if chunk in self.ids:
            return self.ids[chunk]
        i, self.next_id = self.next_id, self.next_id + 1
        self.ids[chunk] = i
        self.chunks[i] = chunk
        return i

    def train(self, texts, max_merges: int | None = None) -> int:
        """Deterministically build merge rules. Returns number of merges made."""
        if max_merges is None:
            max_merges = self.vocab_size - self.base
        counts: dict[bytes, int] = defaultdict(int)
        for text in texts:
            for word in GPT2_PAT.findall(text):
                counts[word.encode("utf-8")] += 1
        words: dict[bytes, list[int]] = {wb: list(wb) for wb in counts}
        symbols = [words[wb] for wb in words]
        weights = list(counts.values())

        made = 0
        for _ in range(max_merges):
            pairs: dict[tuple[int, int], int] = defaultdict(int)
            for sym, w in zip(symbols, weights):
                for a, b in zip(sym, sym[1:]):
                    pairs[(a, b)] += w
            if not pairs:
                break
            (a, b), best_count = max(pairs.items(), key=lambda kv: kv[1])
            if best_count < 2:          # spurious-pair guard: nothing left to learn
                break
            mid = self._alloc(self.chunks[a] + self.chunks[b])
            self.merges[(a, b)] = mid
            made += 1
            symbols = [self._replace(s, a, b, mid) for s in symbols]
        return made

    @staticmethod
    def _replace(sym, a: int, b: int, mid: int) -> list[int]:
        out = []
        i, L = 0, len(sym)
        while i < L:
            if i + 1 < L and sym[i] == a and sym[i + 1] == b:
                out.append(mid)
                i += 2
            else:
                out.append(sym[i])
                i += 1
        return out

    # ------------------------------------------------------------------ #
    # Encode / decode (greedy left-to-right over merge rules)
    # ------------------------------------------------------------------ #
    def encode(self, text: str) -> list[int]:
        ids = list(text.encode("utf-8"))
        changed = True
        while changed:
            changed = False
            out = []
            i, L = 0, len(ids)
            while i < L:
                if i + 1 < L and (ids[i], ids[i + 1]) in self.merges:
                    out.append(self.merges[(ids[i], ids[i + 1])])
                    i += 2
                    changed = True
                else:
                    out.append(ids[i])
                    i += 1
            ids = out
        return ids

    def decode(self, ids) -> str:
        raw = b"".join(self.chunks[int(i)] for i in ids)
        return raw.decode("utf-8", errors="replace")

    def vocabulary_size(self) -> int:
        return self.next_id

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "merges": {f"{a},{b}": m for (a, b), m in self.merges.items()},
            "chunks": [base64.b64encode(self.chunks[i]).decode()
                       for i in range(self.next_id)],
        }

    def save(self, path) -> None:
        import hashlib

        blob = json.dumps(self.to_dict(), sort_keys=True).encode("utf-8")
        path.write_bytes(blob)
        self.fingerprint = hashlib.sha256(blob).hexdigest()[:16]

    @classmethod
    def load(cls, path) -> "ByteLevelBPE":
        import hashlib

        data = json.loads(path.read_text(encoding="utf-8"))
        tok = cls(data["vocab_size"])
        chunks = data["chunks"]
        for i, ch in enumerate(chunks):
            raw = base64.b64decode(ch)
            tok.ids[raw] = i
            tok.chunks[i] = raw
        tok.next_id = len(chunks)          # chunks already include the 256 base
        for key, m in data["merges"].items():
            a, b = (int(x) for x in key.split(","))
            tok.merges[(a, b)] = m
        tok.fingerprint = hashlib.sha256(
            json.dumps(tok.to_dict(), sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        return tok