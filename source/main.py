"""Main orchestration entrypoint for the self-learning pipeline."""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
from data_engine import (  # noqa: E402
    seed_if_empty,
    discover_new_knowledge,
    vocabulary_label,
    build_corpus_mmap,
    load_corpus_index,
)
from self_learner import LossPlateauMonitor  # noqa: E402

log = logging.getLogger("main")


def _setup_logging() -> None:
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)
    fh = logging.FileHandler(config.LOG_FILE, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)


def _pick_engine(force_pytorch: bool, force_airllm: bool):
    if force_airllm and not force_pytorch:
        from airllm_trainer import available
        ok, reason = available()
        if not ok:
            log.critical("AirLLM forced but unavailable: %s", reason)
            sys.exit(1)
        return "airllm"
    if force_pytorch:
        log.info("engine: Standard PyTorch AMP (forced)")
        return "pytorch"
    if config.USE_AIRLLM:
        from airllm_trainer import available
        ok, reason = available()
        if ok:
            log.info("engine: AirLLM layer-wise offloading (%s)", reason)
            return "airllm"
        log.warning("AirLLM unavailable (%s) - falling back to PyTorch AMP",
                    reason)
    return "pytorch"


def run_benchmark() -> None:
    """A/B benchmark: baseline trainer vs optimized, workers 2v4, torch.compile.

    Frozen model/optimizer: throughput measured in useful tokens/sec under
    identical effective batch. Thermal pause disabled during measurement.
    """
    import time as _t

    if not torch.cuda.is_available():
        log.warning("benchmark needs CUDA; skipping")
        return

    torch.set_num_threads(config.CPU_TRAIN_THREADS)
    from pytorch_trainer import (PyTorchAMPModel, bench_throughput,
                                 compile_passes_stability)
    from pytorch_trainer_baseline import PyTorchAMPModel as Baseline

    blocks = seed_if_empty()
    if len(blocks) < 8:
        raise RuntimeError("benchmark needs a seeded buffer")
    vocab = vocabulary_label()
    report = ["engine,tokens_per_s,notes"]
    notes = []

    stable = compile_passes_stability()
    log.info("torch.compile numerical stability: %s", "PASS" if stable else "FAIL")

    # 1) baseline (block-window sampler, single sharp fp32 loss)
    from torch import LongTensor
    saved_pause = config.PAUSE_BETWEEN_STEPS
    config.PAUSE_BETWEEN_STEPS = 0.0
    bl = Baseline(vocab_size=vocab)
    bl.thermal_pause = lambda: None
    ids = blocks[0]["ids"]
    if not isinstance(ids, LongTensor):
        ids = LongTensor(ids)
    probe = bl._sample_windows(ids)
    base_toks = int(probe.numel())                   # actual window tokens
    t0 = _t.perf_counter()
    for _ in range(6):
        bl.train_step(ids)
    base_s = (_t.perf_counter() - t0) / 6.0
    base_tps = base_toks / base_s
    report.append(f"baseline,{base_tps:.1f},6 warmup+meas; {base_toks} toks/step")
    del bl
    torch.cuda.empty_cache()
    config.PAUSE_BETWEEN_STEPS = saved_pause

    # 2) optimized (mmap loader, grad-accum) workers 2
    config.DATALOADER_WORKERS = 2
    opt2 = PyTorchAMPModel(vocab_size=vocab, corpus_index=load_corpus_index())
    opt2.throttle = lambda: None
    opt2.record_metrics = lambda: None
    tps2 = bench_throughput(opt2, steps=30)
    report.append(f"optimized_w2,{tps2:.1f},grad_accum={config.GRAD_ACCUM} x mb={config.MICRO_BATCH} ctx={config.CONTEXT}")
    del opt2
    torch.cuda.empty_cache()

    # 3) optimized workers 4
    config.DATALOADER_WORKERS = 4
    opt4 = PyTorchAMPModel(vocab_size=vocab, corpus_index=load_corpus_index())
    opt4.throttle = lambda: None
    opt4.record_metrics = lambda: None
    tps4 = bench_throughput(opt4, steps=30)
    report.append(f"optimized_w4,{tps4:.1f},workers=4")
    del opt4
    torch.cuda.empty_cache()

    # 4) optimized + torch.compile (only if numerically stable)
    tpsC = None
    if stable:
        config.USE_TORCH_COMPILE = True
        optc = PyTorchAMPModel(vocab_size=vocab, corpus_index=load_corpus_index())
        optc.throttle = lambda: None
        optc.record_metrics = lambda: None
        tpsC = bench_throughput(optc, steps=30)
        report.append(f"optimized_compile,{tpsC:.1f},torch.compile=on")
        del optc
        torch.cuda.empty_cache()
    config.USE_TORCH_COMPILE = False

    log.info("=" * 60)
    log.info("BENCHMARK (tokens/sec, effective batch=%d, ctx=%d)",
             config.EFFECTIVE_BATCH, config.CONTEXT)
    if stable:
        log.info("  torch.compile stability: PASS")
    else:
        log.info("  torch.compile stability: FAIL -> stays disabled")
    for line in report:
        log.info("  %s", line)
    log.info("=" * 60)
    config.LOGS_DIR.mkdir(exist_ok=True)
    (config.LOGS_DIR / "benchmark_report.txt").write_text(
        "\n".join(report) + f"\ncompile_stable={stable}\n", encoding="utf-8")


def main() -> int:
    _setup_logging()
    ap = argparse.ArgumentParser(description="Continuous auto-self-learning pipeline")
    ap.add_argument("--steps", type=int, default=config.MAX_STEPS)
    ap.add_argument("--use-pytorch", action="store_true")
    ap.add_argument("--use-airllm", action="store_true")
    ap.add_argument("--no-thermal", action="store_true", help="disable thermal pause")
    ap.add_argument("--no-pause", action="store_true", help="disable micro-yield sleep")
    ap.add_argument("--benchmark", action="store_true",
                    help="run throughput A/B benchmark and exit")
    args = ap.parse_args()

    if args.benchmark:
        run_benchmark()
        return 0

    if args.no_pause:
        config.PAUSE_BETWEEN_STEPS = 0.0

    log.info("=" * 64)
    log.info("self-learning pipeline (steps=%d, model=%dL/%dE, vocab=%d)",
             args.steps, config.N_LAYER, config.N_EMBD, config.VOCAB_SIZE)
    log.info("=" * 64)

    blocks = seed_if_empty()
    engine_mode = _pick_engine(args.use_pytorch, args.use_airllm)

    if engine_mode == "airllm":
        from airllm_trainer import AirLLMTrainingEngine
        engine = AirLLMTrainingEngine(device=config.DEFAULT_DEVICE)
    else:
        from pytorch_trainer import PyTorchAMPModel
        build_corpus_mmap(blocks)                  # one-time pre-tokenized mmap
        index = load_corpus_index()
        if index is None:
            log.critical("could not assemble corpus mmap; aborting")
            return 1
        engine = PyTorchAMPModel(vocab_size=vocabulary_label(),
                                 corpus_index=index)

    def discover(_old):
        refreshed = discover_new_knowledge(blocks)
        if refreshed and refreshed is not blocks:
            nonlocal_holder["blocks"] = refreshed
        return refreshed

    nonlocal_holder = {"blocks": blocks}
    monitor = LossPlateauMonitor(discover)

    global_step = 0
    last_report = time.time()
    smoothed = None

    try:
        while global_step < args.steps and not engine.converged:
            cur_blocks = nonlocal_holder["blocks"]
            if not cur_blocks:
                log.warning("buffer empty; rediscovering")
                refreshed = discover_new_knowledge([])
                nonlocal_holder["blocks"] = refreshed or cur_blocks
                cur_blocks = nonlocal_holder["blocks"]
                if not cur_blocks:
                    break

            try:
                if engine_mode == "airllm":
                    tsr = cur_blocks[global_step % len(cur_blocks)]["ids"]
                    loss = engine.train_airllm_step([tsr])
                else:
                    loss = engine.train_step()
                if not (loss == loss):                 # NaN guard (loss != loss)
                    log.error("NaN loss detected at step %d", global_step + 1)
                    continue
            except Exception as exc:  # noqa: BLE001
                log.error("step failed (%s); continuing", exc)
                continue

            global_step += 1
            smoothed = loss if smoothed is None else 0.98 * smoothed + 0.02 * loss

            new_blocks = monitor.update(loss)
            if new_blocks and new_blocks is not cur_blocks:
                nonlocal_holder["blocks"] = new_blocks
                # rebuild the mmap ONLY if the block set actually changed;
                # a reshuffled-but-identical buffer reuses the live mmap
                # (avoids touching a memory-mapped file mid-training)
                old_names = {b["path"].name for b in cur_blocks}
                fresh_names = {b["path"].name for b in new_blocks}
                if fresh_names != old_names:
                    build_corpus_mmap(new_blocks)
                    index = load_corpus_index()
                    if index is not None and hasattr(engine, "reload_corpus"):
                        engine.reload_corpus(index)

            if not args.no_thermal and hasattr(engine, "validate") \
                    and global_step % config.VAL_INTERVAL_STEPS == 0:
                engine.validate()

            if engine_mode == "pytorch" and hasattr(engine, "save") \
                    and global_step % config.CHECKPOINT_INTERVAL == 0:
                engine.save("latest")

            if global_step % config.METRICS_INTERVAL == 0:
                if hasattr(engine, "record_metrics"):
                    engine.record_metrics()
            elif time.time() - last_report >= 10:
                last_report = time.time()
                log.info("step %-7d engine=%-7s train=%.4f blocks=%d val_best=%.4f",
                         global_step, engine_mode, smoothed, len(cur_blocks),
                         getattr(engine, "best_val", float("nan")))
    except KeyboardInterrupt:
        log.warning("interrupted; saving before exit")
        if hasattr(engine, "close"):
            engine.close()
        return 130

    if hasattr(engine, "close"):
        engine.close()

    if engine.converged:
        log.info("TRAINING COMPLETE - %s", engine.converge_reason)
        log.info("final best_val=%.4f at step %d", engine.best_val, global_step)
        return 0
    log.info("pipeline finished cleanly at step %d (step budget reached)",
             global_step)
    return 0


if __name__ == "__main__":
    sys.exit(main())