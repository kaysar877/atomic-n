"""Asynchronous web scraper, quality cleaner, chunker, and binary .pt writer.

Pipeline: fetch_targets() -> clean_html() -> chunk_texts() -> quality_filter()
        -> BPE encode -> write_block()

All tokenization happens on the CPU; tensors spill straight to
``buffer_data/knowledge_block_{N}.pt``. Blocks carry a tokenizer fingerprint so
stale blocks from an older vocabulary are detected and skipped, never crashing
the training loop.
"""
from __future__ import annotations

import hashlib
import html
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch
from torch import LongTensor

import config
from tokenizer import ByteLevelBPE

log = logging.getLogger("data_engine")

_TOKENIZER_PATH = config.DATA_DIR / "tokenizer.json"
_DISCOVERY_STATE = config.LOGS_DIR / ".discovery_state.json"

_STYLE_SCRIPT = re.compile(r"<(script|style|noscript)[^>]*>.*?</\1>", re.I | re.S)
_TAGS = re.compile(r"<[^>]+>")
_MULTI_WS = re.compile(r"[ \t\r\f\v]+")
_NEWLINES = re.compile(r"\n{3,}")
_WORD_SPLIT = re.compile(r"\s+")


# --------------------------------------------------------------------------- #
# Tokenizer lifecycle
# --------------------------------------------------------------------------- #
def current_tokenizer() -> ByteLevelBPE:
    if _TOKENIZER_PATH.exists():
        tok = ByteLevelBPE.load(_TOKENIZER_PATH)
    else:
        tok = ByteLevelBPE(config.VOCAB_SIZE)
        tok.fingerprint = "untrained"
    return tok


def ensure_tokenizer(texts: list[str]) -> ByteLevelBPE:
    """Load the saved tokenizer, or train+persist one if missing."""
    if _TOKENIZER_PATH.exists():
        tok = ByteLevelBPE.load(_TOKENIZER_PATH)
        log.info("tokenizer loaded: vocab=%d fp=%s", tok.vocabulary_size(),
                 tok.fingerprint)
        return tok
    tok = ByteLevelBPE(config.VOCAB_SIZE)
    merges = tok.train(texts)
    tok.save(_TOKENIZER_PATH)
    log.info("tokenizer trained: %d merges, vocab=%d fp=%s",
             merges, tok.vocabulary_size(), tok.fingerprint)
    return tok


# --------------------------------------------------------------------------- #
# Fetch layer (async, non-blocking via a thread pool)
# --------------------------------------------------------------------------- #
def _fetch_one(url: str, timeout: float = config.REQUEST_TIMEOUT_S) -> str:
    import urllib.error
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": config.USER_AGENT})
    last_err: Exception | None = None
    for attempt in range(config.REQUEST_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read(config.MAX_FETCH_BYTES + 1)
            if len(raw) > config.MAX_FETCH_BYTES:      # oversized -> discard
                log.warning("source too large: %s", url)
                return ""
            ctype = resp.headers.get("Content-Type", "") or ""
            if "json" in ctype:
                import json as _json

                data = _json.loads(raw.decode("utf-8", "ignore"))
                if isinstance(data, dict) and "extract" in data:
                    return data["extract"]
                return str(data)
            return raw.decode("utf-8", "ignore")
        except (urllib.error.URLError, OSError, ValueError) as exc:
            last_err = exc
            if attempt < config.REQUEST_RETRIES:
                continue
    log.warning("fetch failed for %s: %s", url, last_err)
    return ""


def fetch_targets(urls=None) -> list[str]:
    urls = urls or list(config.TARGET_URLS)
    results: list[str] = []
    with ThreadPoolExecutor(max_workers=config.MAX_CONCURRENCY) as pool:
        futures = {pool.submit(_fetch_one, u): u for u in urls}
        for fut in as_completed(futures):
            text = fut.result()
            if text:
                results.append(text)
    log.info("fetched %d/%d targets", len(results), len(urls))
    return results


# --------------------------------------------------------------------------- #
# Cleaner + chunker
# --------------------------------------------------------------------------- #
def clean_html(raw: str) -> str:
    if not raw:
        return ""
    text = _STYLE_SCRIPT.sub(" ", raw)
    text = _TAGS.sub(" ", text)
    text = html.unescape(text)
    text = _MULTI_WS.sub(" ", text)
    text = _NEWLINES.sub("\n\n", text)
    return text.strip()


def chunk_texts(text: str, max_words: int = None,
                overlap: int = None) -> list[str]:
    """Split an arbitrary-length document into bounded word-window chunks."""
    max_words = max_words or config.MAX_WORDS
    overlap = config.CHUNK_OVERLAP if overlap is None else overlap
    words = _WORD_SPLIT.split(text)
    if not words:
        return []
    out = []
    step = max(max_words - overlap, 1)
    for i in range(0, len(words), step):
        out.append(" ".join(words[i:i + max_words]))
        if i + max_words >= len(words):
            break
    return out


# --------------------------------------------------------------------------- #
# CPU heuristic filter
# --------------------------------------------------------------------------- #
def _vocab_entropy(text: str) -> float:
    toks = re.findall(r"[A-Za-z0-9']+", text.lower())
    if not toks:
        return 0.0
    return len(set(toks)) / len(toks)


def quality_filter(text: str) -> bool:
    words = _WORD_SPLIT.split(text)
    if not (config.MIN_WORDS <= len(words) <= config.MAX_WORDS):
        return False
    lower = " ".join(words).lower()
    if any(term in lower for term in config.NOISE_TERMS):
        return False
    if _vocab_entropy(text) < config.ENTROPY_RATIO:
        return False
    return True


def collect_from(urls: list[str]) -> list[str]:
    """Fetch, clean, chunk, filter. Returns ready-to-tokenize chunks."""
    raw = fetch_targets(urls)
    kept = []
    for block in raw:
        c = clean_html(block)
        for chunk in chunk_texts(c):
            if quality_filter(chunk):
                kept.append(chunk)
    log.info("collect_from(%d urls): %d usable chunks", len(urls), len(kept))
    return kept


# --------------------------------------------------------------------------- #
# Binary writer / loader
# --------------------------------------------------------------------------- #
def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def next_block_number() -> int:
    nums = [int(p.stem.split("_")[-1])
            for p in config.BUFFER_DIR.glob("knowledge_block_*.pt")]
    return (max(nums) + 1) if nums else 0


def write_block(ids: LongTensor, tok: ByteLevelBPE, source: str = "seed") -> Path:
    if ids.numel() == 0:
        return Path("")
    block_path = config.BUFFER_DIR / f"knowledge_block_{next_block_number():06d}.pt"
    tmp = block_path.with_suffix(".pt.tmp")
    payload = {
        "ids": ids.cpu(),
        "meta": {
            "source": str(source),
            "n_tokens": int(ids.numel()),
            "tok_fp": getattr(tok, "fingerprint", "untrained"),
        },
    }
    try:
        torch.save(payload, tmp)
        tmp.replace(block_path)
    except Exception as exc:  # noqa: BLE001
        log.error("block write failed (%s): %s", block_path.name, exc)
        tmp.unlink(missing_ok=True)
        return Path("")
    log.info("wrote %s (%d tokens)", block_path.name, ids.numel())
    return block_path


def load_all_blocks() -> list[dict]:
    """Load valid blocks; skip corrupted AND vocabulary-stale ones."""
    tok_fp = getattr(current_tokenizer(), "fingerprint", "untrained")
    blocks = []
    stale = []
    for path in sorted(config.BUFFER_DIR.glob("knowledge_block_*.pt")):
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            ids = payload["ids"]
            if not isinstance(ids, LongTensor):
                ids = LongTensor(ids)
            meta = payload.get("meta", {})
            if ids.numel() == 0:
                stale.append(path)
                continue
            if meta.get("tok_fp", "") != tok_fp:
                stale.append(path)
                continue
            blocks.append({"path": path, "ids": ids,
                           "source": meta.get("source", "?")})
        except Exception as exc:  # noqa: BLE001
            log.warning("skipping corrupt block %s: %s", path.name, exc)
            stale.append(path)
    for p in stale:
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass
    if stale:
        log.info("removed %d stale/corrupt block(s)", len(stale))
    return blocks


def _hashes_seen(blocks: list[dict]) -> set[str]:
    seen = set()
    for b in blocks:
        src = str(b.get("source", ""))
        if ":" in src:
            seen.add(src.split(":", 1)[1])
    return seen


# --------------------------------------------------------------------------- #
# Seeding
# --------------------------------------------------------------------------- #
def _builtin_corpus() -> list[str]:
    return [
        "The theory of evolution by natural selection is the fundamental "
        "mechanism by which living organisms adapt and change over time. "
        "Charles Darwin proposed that individuals within a population vary "
        "and that advantageous traits become more common across generations.",
        "Thermodynamics describes how heat and energy move through physical "
        "systems. The second law states that entropy of an isolated system "
        "never decreases, which explains why engines require a temperature "
        "difference to work and why time appears to move in one direction.",
        "Machine learning is a branch of artificial intelligence that enables "
        "computers to learn patterns directly from data instead of following "
        "explicit programming rules hard coded by hand.",
        "Black holes are regions of spacetime where gravity is so intense "
        "that nothing, not even light, can escape once it crosses the event "
        "horizon that marks the boundary of each collapsed star.",
        "The study of ancient civilizations reveals how cities emerged when "
        "agriculture produced surplus food and writing was invented to record "
        "taxes, grain stores, laws, and eventually literature and history.",
        "Photosynthesis converts sunlight into chemical energy, allowing "
        "plants to build sugars from carbon dioxide and water and releasing "
        "oxygen that sustains nearly every animal on the planet.",
    ]


def seed_if_empty() -> list[dict]:
    blocks = load_all_blocks()
    if blocks:
        log.info("buffer has %d valid block(s); skipping seed", len(blocks))
        return blocks

    # no valid blocks -> wipe stale ones and rebuild from scratch
    for p in config.BUFFER_DIR.glob("knowledge_block_*.pt"):
        p.unlink(missing_ok=True)

    texts = []
    sources = []
    web = collect_from(config.TARGET_URLS) or []
    books = collect_from(config.GUTENBERG_URLS) or []
    texts += web
    sources += ["web"] * len(web)
    texts += books
    sources += ["book"] * len(books)
    if not texts:
        log.warning("network unreachable; falling back to builtin corpus")
        texts = _builtin_corpus()
        sources = ["builtin"] * len(texts)

    tok = ensure_tokenizer(texts)
    seen = set()
    for text, src in zip(texts, sources):
        h = _text_hash(text)
        if h in seen:
            continue
        seen.add(h)
        ids = LongTensor(tok.encode(text))
        if ids.numel():
            write_block(ids, tok, source=f"{src}:{h}")
    out = load_all_blocks()
    if not out:
        raise RuntimeError("seed produced no blocks - check buffer_data/")
    log.info("seeded %d block(s), vocab=%d fp=%s",
             len(out), tok.vocabulary_size(), tok.fingerprint)
    return out


# --------------------------------------------------------------------------- #
# Dynamic knowledge discovery (self_learner callback)
# --------------------------------------------------------------------------- #
def _source_pool() -> list[str]:
    return list(config.TARGET_URLS) + list(config.GUTENBERG_URLS)


def _next_sources() -> tuple[list[str], int]:
    pool = _source_pool()
    round_no = 0
    try:
        round_no = int(json.loads(
            _DISCOVERY_STATE.read_text(encoding="utf-8")).get("round", 0))
    except Exception:  # noqa: BLE001
        round_no = 0
    start = (round_no * config.DISCOVERY_ROUND_BATCH) % len(pool)
    batch = (pool * 2)[start:start + config.DISCOVERY_ROUND_BATCH]
    with _DISCOVERY_STATE.open("w", encoding="utf-8") as fh:
        json.dump({"round": round_no + 1}, fh)
    return batch, round_no + 1


def discover_new_knowledge(current_blocks: list[dict]) -> list[dict]:
    """Harvest the next unseen source batch, dedupe, write blocks."""
    log.info("knowledge discovery triggered")
    urls, round_no = _next_sources()
    chunks = collect_from(urls)
    if not chunks:
        log.warning("discovery round %d fetched nothing; keeping buffer",
                    round_no)
        return current_blocks

    tok = current_tokenizer()
    seen = _hashes_seen(current_blocks)
    added = 0
    for text in chunks:
        h = _text_hash(text)
        if h in seen:
            continue
        seen.add(h)
        ids = LongTensor(tok.encode(text))
        if ids.numel() and quality_filter(text):
            p = write_block(ids, tok, source=f"web:{h}")
            if p.exists():
                added += 1

    refreshed = load_all_blocks()
    if not refreshed:
        return current_blocks
    from random import shuffle
    shuffle(refreshed)
    log.info("discovery round %d added %d block(s); buffer now %d",
             round_no, added, len(refreshed))
    return refreshed


def vocabulary_label() -> int:
    """Vocab size to build the model with (tokenizer-driven)."""
    return max(current_tokenizer().vocabulary_size(), 64)


# --------------------------------------------------------------------------- #
# Memory-mapped corpus (pre-tokenization, built ONCE outside the train loop)
# --------------------------------------------------------------------------- #
def corpus_fingerprint(blocks: list[dict]) -> str:
    """Stable id for the assembled corpus (order-invariant).

    Sorted by block name so a reshuffled-but-identical buffer maps to the
    same fingerprint and never triggers a pointless mmap rebuild.
    """
    h = hashlib.sha256()
    h.update(str(len(blocks)).encode())
    for b in sorted(blocks, key=lambda d: str(d["path"].name)):
        h.update(bytes(str(b["path"].name) + ":" + str(b["ids"].numel()) + " ",
                        "utf-8"))
    return h.hexdigest()[:16]


def build_corpus_mmap(blocks: list[dict]) -> Path:
    """Flatten block token-ids into ONE compact mmap file + offset index.

    Pure rearrangement of already-tokenized ids: no tokenizer calls, no
    network, and (on a warm cache) sub-second. Vocab 4096 < 65536, so ids
    pack into uint16 -> the 685-block corpus fits in a few MB.
    """
    if not blocks:
        raise ValueError("cannot build corpus from zero blocks")
    fp = corpus_fingerprint(blocks)
    array_path = config.BUFFER_DIR / f"corpus_{fp}.npy"
    index_path = config.BUFFER_DIR / f"corpus_{fp}.json"
    if array_path.exists() and index_path.exists():
        log.info("corpus mmap already present (%s)", array_path.name)
        return array_path

    import numpy as np

    total = sum(int(b["ids"].numel()) for b in blocks)
    offsets = []
    flat = np.empty(total, dtype=np.uint16)
    cursor = 0
    train_offs, val_offs = [], []
    for i, b in enumerate(blocks):
        arr = b["ids"].numpy().astype(np.uint16, copy=False)
        flat[cursor:cursor + arr.size] = arr
        if (i % 100) / 100.0 < config.VAL_SPLIT:
            val_offs.append([cursor, int(arr.size)])
        else:
            train_offs.append([cursor, int(arr.size)])
        offsets.append([cursor, int(arr.size)])
        cursor += int(arr.size)
    np.save(array_path, flat)
    import json as _json
    index_path.write_text(_json.dumps({
        "n_tokens": int(total),
        "block_count": len(blocks),
        "tok_fp": getattr(current_tokenizer(), "fingerprint", "untrained"),
        "train": train_offs, "val": val_offs, "offsets": offsets,
        "corpus_fp": fp,
    }, indent=0), encoding="utf-8")
    # drop superseded corpus artifacts (locked files are simply kept;
    # the running engine may still have them memory-mapped on Windows)
    for old in config.BUFFER_DIR.glob("corpus_*.json"):
        if "corpus_" + fp not in old.name:
            for victim in (old, config.BUFFER_DIR / (old.name[:-5] + ".npy")):
                try:
                    victim.unlink(missing_ok=True)
                except OSError:
                    pass
    log.info("corpus mmap written: %s (%d tokens, %d train / %d val spans)",
             array_path.name, total, len(train_offs), len(val_offs))
    return array_path


def load_corpus_index() -> dict | None:
    """Return the newest matching corpus index, or None if stale/missing."""
    own_fp = getattr(current_tokenizer(), "fingerprint", "untrained")
    best = None
    for idx in sorted(config.BUFFER_DIR.glob("corpus_*.json"), reverse=True):
        try:
            import json as _json
            meta = _json.loads(idx.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if meta.get("tok_fp") != own_fp:
            continue
        arr = config.BUFFER_DIR / f"corpus_{meta['corpus_fp']}.npy"
        if arr.exists():
            meta["array_path"] = str(arr)
            best = meta
            break
    if best is None:
        log.info("no compatible corpus mmap yet")
    return best