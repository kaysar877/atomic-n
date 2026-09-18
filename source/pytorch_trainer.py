"""Optimized PyTorch AMP engine.

Frozen (do NOT touch in this experiment): architecture, tokenizer, dataset
selection, optimizer (AdamW 3e-4), LR handling, model dimensions.

Execution-layer optimizations only:
  * one-time memory-mapped corpus (uint16), never re-tokenized
  * fixed context length + fixed batch shapes (no per-step Python padders)
  * persistent DataLoader (2-4 workers) + pinned memory + non_blocking copies
  * micro-batch + gradient accumulation to bound VRAM/temperature
  * adaptive throttling (temp + CPU convoy) with a pause ceiling
  * metrics every METRICS_INTERVAL steps (tokens/s, step time, CPU/GPU/VRAM,
    temp, power, train loss)
  * latest.pt at CHECKPOINT_INTERVAL, best.pt only on validation improvement
  * full resumability incl. RNG + tokenizer fingerprint guard
"""
from __future__ import annotations

import logging
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import config

from atomic_block import AtomicFFN
from atomic_v2_cell import AtomicV2FFN

log = logging.getLogger("pytorch_trainer")

LATEST_PATH = config.CHECKPOINTS_DIR / "latest.pt"
BEST_PATH = config.CHECKPOINTS_DIR / "best.pt"


# --------------------------------------------------------------------------- #
# Frozen architecture (identical to baseline)
# --------------------------------------------------------------------------- #
class CausalBlock(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = nn.MultiheadAttention(n_embd, n_head, dropout=dropout,
                                          batch_first=True)
        self.ln2 = nn.LayerNorm(n_embd)
        # Cell switch (SELFLEARN_CELL=atomic|mlp|atomic_v2, default atomic).
        # Param-matched: AtomicFFN(d) == MLP d->4d->d == AtomicV2FFN(d)
        # (2,099,712 @ d=512). Lets the baseline sweep reuse this exact file:
        # the ONLY difference between runs becomes the cell. Running
        # processes are unaffected (they hold their own imported module).
        if os.environ.get("SELFLEARN_CELL", "atomic") == "mlp":
            self.mlp = nn.Sequential(
                nn.Linear(n_embd, 4 * n_embd), nn.GELU(),
                nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout))
            self.cell = "mlp"
        elif os.environ.get("SELFLEARN_CELL") == "atomic_v2":
            self.mlp = AtomicV2FFN(n_embd, dropout=dropout)
            self.cell = "atomic_v2"
        else:
            self.mlp = AtomicFFN(n_embd)
            self.cell = "atomic"

    def forward(self, x, mask):
        a, _ = self.attn(x, x, x, attn_mask=mask, need_weights=False)
        x = self.ln1(x + a)
        return self.ln2(x + self.mlp(x))


class GPT(nn.Module):
    def __init__(self, vocab_size: int, n_embd: int = config.N_EMBD,
                 n_head: int = config.N_HEAD, n_layer: int = config.N_LAYER,
                 dropout: float = config.DROPOUT,
                 block_size: int = config.BLOCK_SIZE):
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.tok = nn.Embedding(vocab_size, n_embd)
        self.pos = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList(
            [CausalBlock(n_embd, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.tok.weight = self.head.weight
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, idx):
        b, t = idx.shape
        assert t <= self.block_size, f"block too long: {t}"
        x = self.tok(idx) + self.pos(torch.arange(t, device=idx.device))
        mask = torch.triu(torch.full((t, t), float("-inf"), device=idx.device),
                          diagonal=1)
        for blk in self.blocks:
            x = blk(x, mask)
        return self.head(self.ln_f(x))

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# --------------------------------------------------------------------------- #
# Memory-mapped corpus sampler (CPU prep, async via DataLoader workers)
# --------------------------------------------------------------------------- #
class MmapCorpusDataset(torch.utils.data.IterableDataset):
    """Infinite stream of fixed-shape (MB, CONTEXT+1) windows from the mmap.

    Two access modes:
      * worker-stream (num_workers > 0): each persistent worker draws an
        independent deterministic stream seeded by LOADER_SEED+worker_id;
        used for the 2-vs-4 benchmark. Draws are reproducible per-seed but
        not bit-continuous across a resume (position lists shuffle).
      * one-shot seeds (num_workers == 0, recommended): ``batch_from_seed``
        derives a concrete batch from a single 63-bit seed. The engine keeps
        ONE persisted sampling RNG in the checkpoint, so a resumed run
        reproduces the exact next batch -> fully deterministic continuation.
    """

    def __init__(self, index: dict, micro_batch: int = config.MICRO_BATCH,
                 context: int = config.CONTEXT, seed: int = config.LOADER_SEED):
        super().__init__()
        self.array_path = index["array_path"]
        self.spans = index["train"]
        self.val_spans = index.get("val", [])
        self.micro_batch = micro_batch
        self.context = context
        self.seed = seed
        self._usable = None
        self._usable_val = None

    def usable(self) -> list[tuple[int, int]]:
        if self._usable is None:
            self._usable = [(lo, ln) for lo, ln in self.spans
                            if ln >= self.context + 1]
        return self._usable

    def usable_val(self) -> list[tuple[int, int]]:
        if self._usable_val is None:
            self._usable_val = [(lo, ln) for lo, ln in self.val_spans
                                if ln >= self.context + 1]
        return self._usable_val

    def batch_from_seed(self, seed: int) -> torch.Tensor:
        """Build one (micro_batch, context+1) window batch from `seed`."""
        usable = self.usable()
        rng = random.Random(seed)
        rows = []
        for _ in range(self.micro_batch):
            lo, ln = usable[rng.randrange(len(usable))]
            start = rng.randint(0, ln - self.context - 1)
            rows.append(self._slice(lo, start))
        return torch.stack(rows)

    def _mem(self):
        if not hasattr(self, "_mm"):
            self._mm = np.memmap(self.array_path, dtype=np.uint16, mode="r")
        return self._mm

    def _slice(self, lo: int, start: int) -> torch.Tensor:
        """Long tensor for one window; materializes a writable copy."""
        seg = self._mem()[lo + start:lo + start + self.context + 1]
        return torch.from_numpy(np.ascontiguousarray(seg, dtype=np.int64)).long()

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        usable = self.usable()
        rng = random.Random(self.seed * 1_000_003 + worker_id)
        while True:
            rows = []
            for _ in range(self.micro_batch):
                lo, ln = usable[rng.randrange(len(usable))]
                start = rng.randint(0, ln - self.context - 1)
                rows.append(self._slice(lo, start))
            yield torch.stack(rows)


def _psutil_cpu_percent() -> float | None:
    try:
        import psutil
        # non-blocking: compares against the previous call's timestamp.
        # First call after import returns 0.0; subsequent calls are real.
        return float(psutil.cpu_percent(interval=None))
    except Exception:  # noqa: BLE001 - optional metric
        return None


# --------------------------------------------------------------------------- #
# Optimized trainer
# --------------------------------------------------------------------------- #
class PyTorchAMPModel:
    def __init__(self, vocab_size: int, corpus_index: dict | None = None,
                 device: str | None = None):
        self.device = device or config.DEFAULT_DEVICE
        if not torch.cuda.is_available():
            self.device = "cpu"
        self.use_cuda = self.device != "cpu"
        torch.set_num_threads(config.CPU_TRAIN_THREADS)

        self.vocab_size = vocab_size
        self.model = GPT(vocab_size).to(self.device)
        if config.USE_TORCH_COMPILE and self.use_cuda:
            self.model = torch.compile(self.model)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.LR,
                                           weight_decay=float(
                                               os.environ.get("SELFLEARN_WD", "0.01")))
        self.scheduler = self._make_scheduler()
        self.bf16 = bool(self.use_cuda and config.USE_BF16_ON_CUDA
                         and torch.cuda.get_device_capability() >= (8, 0))
        self.scaler = None                     # bf16 needs no GradScaler
        self.amp_dtype = torch.bfloat16 if self.bf16 else torch.float16
        self.loss_fn = nn.CrossEntropyLoss()

        self.total_steps = 0
        self.converged = False
        self.converge_reason = ""
        self.best_val = float("inf")
        self.val_lashes = 0
        self._tok_fp = ""
        self.micro_batch = config.MICRO_BATCH
        self.grad_accum = config.GRAD_ACCUM
        self.context = config.CONTEXT
        self.num_workers = config.DATALOADER_WORKERS
        self._pause = float(config.PAUSE_BETWEEN_STEPS)
        self._loader = None

        self._rng = random.Random(config.RANDOM_SEED)
        self._sample_rng = random.Random(config.LOADER_SEED + 17)  # data RNG
        self._step_times: list[float] = []
        self._last_loss = 0.0
        self._metrics_since = 0
        self._metrics_file = config.LOGS_DIR / "metrics.csv"
        self._metrics_wrote_header = False

        self._last_temp = 0.0
        self._last_temp_t = 0.0

        if corpus_index is not None:
            self.reload_corpus(corpus_index)
        self._resume_if_exists()

    # ------------------------------------------------------------------ #
    # Loader lifecycle
    # ------------------------------------------------------------------ #
    def _make_scheduler(self):
        """Optional cosine+warmup LR schedule, gated by env (default: None).

        SELFLEARN_SCHEDULE=cosine  -> cosine decay to SELFLEARN_MIN_LR_FRAC
        SELFLEARN_WARMUP=0         -> linear warmup steps (0 = none)
        SELFLEARN_TOTAL_STEPS=N    -> total optimizer steps for the cosine T_max
        Frozen arena behavior (scheduler=None) when SELFLEARN_SCHEDULE is unset.
        """
        if os.environ.get("SELFLEARN_SCHEDULE", "") != "cosine":
            return None
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
        total = int(os.environ.get("SELFLEARN_TOTAL_STEPS", str(config.MAX_STEPS)))
        warmup = max(0, int(os.environ.get("SELFLEARN_WARMUP", "0")))
        min_frac = float(os.environ.get("SELFLEARN_MIN_LR_FRAC", "0.1"))
        warm = LinearLR(self.optimizer, start_factor=0.01,
                        total_iters=warmup) if warmup else None
        cosine = CosineAnnealingLR(
            self.optimizer, T_max=max(1, total - warmup),
            eta_min=config.LR * min_frac)
        if warm is not None:
            return SequentialLR(self.optimizer, [warm, cosine],
                                milestones=[warmup])
        return cosine

    def reload_corpus(self, index: dict) -> None:
        from data_engine import current_tokenizer
        self._tok_fp = getattr(current_tokenizer(), "fingerprint", "untrained")
        self._dset = MmapCorpusDataset(index)
        self._make_loader()

    def _make_loader(self) -> None:
        if self.num_workers <= 0:
            self._loader = None
            self._loader_iter = None
            return
        self._loader = torch.utils.data.DataLoader(
            self._dset,
            batch_size=None,
            num_workers=self.num_workers,
            persistent_workers=True,
            prefetch_factor=config.PREFETCH_FACTOR,
            pin_memory=bool(config.PIN_MEMORY and self.use_cuda),
            timeout=60,
        )
        self._loader_iter = iter(self._loader)

    def set_workers(self, n: int) -> None:
        n = max(0, int(n))
        if n == self.num_workers:
            return
        log.info("data-loader workers %d -> %d (CPU convoy control)",
                 self.num_workers, n)
        self.num_workers = n
        self._make_loader()

    def _next_batch(self) -> torch.Tensor:
        """Deterministic batch producer (persisted sampling RNG).

        num_workers==0 (default): draws the next window batch from the
        persisted ``_sample_rng`` so a resume continues the exact stream.
        num_workers>0 (benchmark variant): pulls from persistent workers.
        """
        if self._loader is not None:
            try:
                return next(self._loader_iter)
            except (StopIteration, RuntimeError):
                self._make_loader()
                return next(self._loader_iter)
        return self._dset.batch_from_seed(self._sample_rng.getrandbits(63))

    # ------------------------------------------------------------------ #
    # Checkpoints (full resumability)
    # ------------------------------------------------------------------ #
    def save(self, tag: str = "latest") -> None:
        path = BEST_PATH if tag == "best" else LATEST_PATH
        tmp = path.with_suffix(".pt.tmp")
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict()
                         if self.scheduler is not None else None,
            "step": self.total_steps,
            "best_val": self.best_val,
            "val_lashes": self.val_lashes,
            "rng_state": self._rng.getstate(),
            "torch_rng": torch.random.get_rng_state(),
            "sample_rng": self._sample_rng.getstate(),
            "loader_seed": config.LOADER_SEED,
            "tok_fp": self._tok_fp,
            "metrics": {"avg_step_s": self.avg_step_time(),
                        "tokens_per_s": self.tokens_per_s()},
        }, tmp)
        tmp.replace(path)
        if tag == "best":
            log.info("NEW BEST val=%.4f (step %d)", self.best_val,
                     self.total_steps)

    def _resume_if_exists(self) -> None:
        if not LATEST_PATH.exists():
            return
        try:
            ckpt = torch.load(LATEST_PATH, map_location=self.device,
                              weights_only=False)
            saved_fp = ckpt.get("tok_fp", "")
            if saved_fp and self._tok_fp and saved_fp != self._tok_fp:
                log.warning("checkpoint tokenizer %s != corpus %s; refusing "
                            "resume", saved_fp[:8], self._tok_fp[:8])
                self.total_steps = 0
                return
            self.model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            if self.scheduler is not None and ckpt.get("scheduler"):
                self.scheduler.load_state_dict(ckpt["scheduler"])
            self.total_steps = int(ckpt.get("step", 0))
            self.best_val = float(ckpt.get("best_val", float("inf")))
            self.val_lashes = int(ckpt.get("val_lashes", 0))
            if "rng_state" in ckpt:
                self._rng.setstate(ckpt["rng_state"])
            if "sample_rng" in ckpt:
                self._sample_rng.setstate(ckpt["sample_rng"])
            if "torch_rng" in ckpt and not self.use_cuda:
                torch.random.set_rng_state(ckpt["torch_rng"])
            log.info("resumed from %s (step %d, best_val %.4f)",
                     LATEST_PATH.name, self.total_steps, self.best_val)
        except Exception as exc:  # noqa: BLE001
            log.warning("could not resume %s (%s); starting fresh",
                        LATEST_PATH.name, exc)
            self.total_steps = 0

    # ------------------------------------------------------------------ #
    # Thermal + CPU adaptive throttling
    # ------------------------------------------------------------------ #
    def gpu_temp(self) -> float | None:
        if not self.use_cuda:
            return None
        if time.time() - self._last_temp_t < config.GPU_TEMP_POLL_S:
            return self._last_temp
        self._last_temp_t = time.time()
        try:
            out = subprocess.run(  # noqa: S603
                ["nvidia-smi", "--query-gpu=temperature.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5, check=False)
            self._last_temp = float(out.stdout.strip().splitlines()[0])
        except Exception:
            self._last_temp = 0.0
        return self._last_temp

    def throttle(self) -> None:
        """Adaptive pause: raise the yield when temp or CPU is high."""
        if not self.use_cuda and _psutil_cpu_percent() is None:
            return
        target = config.PAUSE_BETWEEN_STEPS
        temp = self.gpu_temp() or 0.0
        if temp >= config.GPU_TEMP_THROTTLE:
            target = max(target, 0.5 * config.THROTTLE_MAX_PAUSE)
        if temp >= config.THERMAL_HARD_LIMIT_C:
            target = config.THROTTLE_MAX_PAUSE
        if _psutil_cpu_percent() is not None:
            cpu = _psutil_cpu_percent()
            if cpu > config.CPU_HIGH_PERCENT:
                target = max(target, 0.3 * config.THROTTLE_MAX_PAUSE)
                if self.num_workers > 2:
                    self.set_workers(2)
        self._pause = min(target, config.THROTTLE_MAX_PAUSE)
        if self._pause > 0:
            time.sleep(self._pause)
        if temp >= config.THERMAL_HARD_LIMIT_C:
            while (self.gpu_temp() or temp) > config.THERMAL_HARD_LIMIT_C:
                time.sleep(0.5)

    # ------------------------------------------------------------------ #
    # Training / validation
    # ------------------------------------------------------------------ #
    def _forward_loss(self, x: torch.Tensor) -> torch.Tensor:
        inp, tgt = x[:, :-1], x[:, 1:]
        with torch.autocast("cuda", dtype=self.amp_dtype,
                            enabled=self.bf16):
            logits = self.model(inp)
            return self.loss_fn(logits.reshape(-1, self.vocab_size),
                                tgt.reshape(-1))
        # NOTE: loader yields CONTEXT+1 tokens; predict positions 1..CONTEXT
        # from 0..CONTEXT-1 — identical to the frozen baseline LM objective.

    def train_step(self) -> float:
        """One optimizer step = GRAD_ACCUM micro-batches of MICRO_BATCH windows."""
        t0 = time.perf_counter()
        self.optimizer.zero_grad(set_to_none=True)
        acc_loss = 0.0
        for _ in range(self.grad_accum):
            batch = self._next_batch()
            if self.use_cuda:
                if batch.device == torch.device("cpu"):
                    batch = batch.pin_memory()
                x = batch.to(self.device, non_blocking=True)
            else:
                x = batch.to(self.device)
            loss = self._forward_loss(x) / self.grad_accum
            loss.backward()
            acc_loss += float(loss.detach())
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        self.total_steps += 1
        self._last_loss = acc_loss
        self._step_times.append(time.perf_counter() - t0)
        if len(self._step_times) > 500:
            self._step_times.pop(0)
        self.throttle()
        return float(self._last_loss)

    @torch.no_grad()
    def evaluate(self) -> float:
        """Fixed-shape validation over the mmap val spans (cheap cadence)."""
        try:
            spans = self._dset.usable_val()
        except AttributeError:
            return float("inf")
        if not spans:
            return float("inf")
        rng = random.Random(777)
        total = count = 0.0
        self.model.eval()
        batch = []

        def flush():
            nonlocal total, count
            if not batch:
                return
            x = torch.stack(batch)
            if self.use_cuda:
                x = x.to(self.device, non_blocking=True)
            total += float(self._forward_loss(x).detach())
            count += 1.0
            batch.clear()

        for _ in range(max(4, min(8, len(spans)) * 4)):
            lo, ln = spans[rng.randrange(len(spans))]
            start = rng.randint(0, ln - self.context - 1)
            batch.append(self._dset._slice(lo, start))
            if len(batch) == self.micro_batch:
                flush()
        flush()
        self.model.train()
        return total / count if count else float("inf")

    def diagnose_product_path(self) -> None:
        """Phase-12: one cheap probe pass over a single val batch.

        Logs per-layer mean/std of A', B', I and gradient norms. Runs only
        when SELFLEARN_DIAG=1 (one extra micro-batch per validation).
        """
        try:
            spans = self._dset.usable_val()
            if not spans:
                return
            import random as _r
            rng = _r.Random(4242 + self.total_steps)
            lo, ln = spans[rng.randrange(len(spans))]
            start = rng.randint(0, ln - self.context - 1)
            x = self._dset._slice(lo, start).unsqueeze(0).to(self.device)
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            loss = self._forward_loss(x)
            loss.backward()
            parts = []
            for i, blk in enumerate(self.model.blocks):
                mlp = blk.mlp
                if getattr(mlp, "cell", "") != "atomic":
                    continue
                # recompute branch stats cheaply from a no-grad forward
                with torch.no_grad():
                    xxf, xxc = x.chunk(2, dim=-1)
                    a = mlp.act(mlp.w_f(xxf)).float()
                    b = mlp.act(mlp.w_c(xxc)).float()
                    ii = (a * b).float()
                gf = mlp.w_f.weight.grad
                go = mlp.w_o.weight.grad
                parts.append(
                    f"L{i}:stdA={a.std():.3f}/stdB={b.std():.3f}/"
                    f"stdI={ii.std():.3f}/maxI={ii.abs().max():.2f}/"
                    f"gF={gf.norm() if gf is not None else -1:.3f}/"
                    f"gO={go.norm() if go is not None else -1:.3f}")
            self.optimizer.zero_grad(set_to_none=True)
            log.info("DIAG step=%d %s", self.total_steps, " | ".join(parts))
        except Exception as exc:  # noqa: BLE001 - diagnostics never break training
            log.warning("diagnostics skipped (%s)", type(exc).__name__)

    def validate(self) -> None:
        val_loss = self.evaluate()
        if os.environ.get("SELFLEARN_DIAG") == "1":
            self.diagnose_product_path()
        if val_loss >= float("inf"):
            return
        self.val_lashes += 1
        improved = val_loss < self.best_val - config.VAL_MIN_IMPROVEMENT
        if improved:
            self.best_val = val_loss
            self.val_lashes = 0
            self.save("best")
        log.info("VAL loss=%.4f best=%.4f staleness=%d/%d",
                 val_loss, self.best_val, self.val_lashes,
                 config.VAL_FLOOR_FIRE)
        if self.val_lashes >= config.VAL_FLOOR_FIRE:
            self.converged = True
            self.converge_reason = (f"validation loss stalled at {self.best_val:.4f} "
                                    f"for {config.VAL_FLOOR_FIRE} eval(s)")

    # ------------------------------------------------------------------ #
    # Metrics
    # ------------------------------------------------------------------ #
    def avg_step_time(self) -> float:
        return (sum(self._step_times) / len(self._step_times)
                if self._step_times else 0.0)

    def tokens_per_s(self) -> float:
        dt = self.avg_step_time()
        toks = config.EFFECTIVE_BATCH * self.context
        return toks / dt if dt > 0 else 0.0

    def metrics(self) -> dict:
        m = {
            "step": self.total_steps,
            "train_loss": round(float(self._last_loss), 5),
            "step_s": round(self.avg_step_time(), 4),
            "tokens_s": round(self.tokens_per_s(), 1),
        }
        if self.use_cuda:
            try:
                out = subprocess.run(  # noqa: S603
                    ["nvidia-smi",
                     "--query-gpu=temperature.gpu,utilization.gpu,memory.used,"
                     "power.draw", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5, check=False)
                parts = (out.stdout.strip().splitlines() or [""])[0].split(",")
                vals = [p.strip() for p in parts]
                def _f(s: str) -> float:
                    try:
                        return float(s)
                    except ValueError:
                        return -1.0
                if len(vals) >= 4:
                    m.update(temp_c=_f(vals[0]), gpu_pct=_f(vals[1]),
                             vram_mb=_f(vals[2]), power_w=_f(vals[3]))
            except Exception:
                pass
        cpu = _psutil_cpu_percent()
        m["cpu_pct"] = round(cpu, 1) if cpu is not None else -1.0
        return m

    def record_metrics(self) -> None:
        m = self.metrics()
        line = ",".join(str(m.get(k, "-")) for k in
                        ("step", "train_loss", "step_s", "tokens_s", "cpu_pct",
                         "temp_c", "gpu_pct", "vram_mb", "power_w"))
        if not self._metrics_wrote_header:
            if not self._metrics_file.exists():
                self._metrics_file.write_text(
                    "step,train_loss,step_s,tokens_s,cpu_pct,temp_c,gpu_pct,"
                    "vram_mb,power_w\n", encoding="utf-8")
            self._metrics_wrote_header = True
        with self._metrics_file.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        log.info("M step=%-6d train=%.4f  %6.1f tok/s  step=%.3fs  "
                 "cpu=%s%% gpu=%s%% vram=%sMB temp=%sC pw=%sW",
                 m["step"], m["train_loss"], m["tokens_s"], m["step_s"],
                 m.get("cpu_pct", "-"), m.get("gpu_pct", "-"),
                 m.get("vram_mb", "-"), m.get("temp_c", "-"),
                 m.get("power_w", "-"))

    def refresh_dataset(self, blocks: list[dict]) -> None:
        from data_engine import build_corpus_mmap, load_corpus_index
        build_corpus_mmap(blocks)
        index = load_corpus_index()
        if index:
            self.reload_corpus(index)
        log.info("dataset refreshed (mmap rebuilt); %d blocks", len(blocks))

    def close(self) -> None:
        self.save("latest")
        if self.use_cuda:
            torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# torch.compile correctness + benchmark helpers (run via `main --benchmark`)
# --------------------------------------------------------------------------- #
def compile_passes_stability(steps: int = config.NUMERICAL_STABILITY_STEPS) -> bool:
    """Compare compiled vs eager logits on identical fixed input (bf16).

    Any failure (no C++ toolchain, inductor unsupported, numeric drift)
    means "do not enable torch.compile" -> returns False, never raises.
    """
    try:
        torch.set_num_threads(config.CPU_TRAIN_THREADS)
        seed_data = torch.randint(0, 512, (2, 96), device="cpu")
        eager = GPT(512).to("cpu").eval()
        comp = GPT(512).to("cpu")
        comp.load_state_dict(eager.state_dict())
        comp = torch.compile(comp)
        with torch.no_grad():
            a = eager(seed_data)
            b = comp(seed_data)
        diff = (a - b).abs()
        rel = diff.max().item() / a.abs().max().item()
        log.info("compile stability: max_rel_diff=%.2e %>", rel)
        return rel < 5e-2
    except Exception as exc:  # noqa: BLE001 - missing toolchain counts as FAIL
        log.warning("torch.compile unavailable/unstable (%s); staying eager",
                    type(exc).__name__)
        return False


def bench_throughput(engine, steps: int = 30) -> float:
    """Warm engine then measure steady-state tokens/sec (step-time based)."""
    for _ in range(5):                       # warmup (compiles / caches)
        engine.train_step()
    engine._step_times.clear()
    for _ in range(steps):
        engine.train_step()
    return engine.tokens_per_s()