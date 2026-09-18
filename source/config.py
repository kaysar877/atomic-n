"""Central runtime configuration for the self-learning pipeline."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BUFFER_DIR = Path(os.environ.get("SELFLEARN_BUFFER_DIR") or ROOT / "buffer_data")
CHECKPOINTS_DIR = Path(os.environ.get("SELFLEARN_CHECKPOINTS_DIR") or ROOT / "checkpoints")
LOGS_DIR = Path(os.environ.get("SELFLEARN_LOGS_DIR") or ROOT / "logs")
DATA_DIR = Path(os.environ.get("SELFLEARN_DATA_DIR") or ROOT / "data")
os.makedirs(BUFFER_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# --------------------------------------------------------------------------- #
# Engine selection
# --------------------------------------------------------------------------- #
USE_AIRLLM = False                   # PyTorch AMP engine (real gradients)
AIRLLM_MODEL_NAME = "garage-bAInd/Orca-Mistral-7B"
AIRLLM_LOCAL_PATH = None
LR = float(os.environ.get("SELFLEARN_LR") or 3e-4)   # sweep override

# --------------------------------------------------------------------------- #
# Hardware / thermal parameters
# --------------------------------------------------------------------------- #
PAUSE_BETWEEN_STEPS = 0.250          # 250ms micro-yield sleep after each step
THERMAL_THRESHOLD_C = 65.0           # target keep-below temperature
THERMAL_HARD_LIMIT_C = 75.0          # forced cooling pause until below
GPU_TEMP_POLL_S = 2.0
CPU_TRAIN_THREADS = 2
BATCH_SIZE = 16
USE_BF16_ON_CUDA = True              # bf16 on Ampere+, fp16 otherwise via scaler
DEFAULT_DEVICE = "cuda:0"            # falls back to "cpu" automatically

# --------------------------------------------------------------------------- #
# Model architecture (fits RTX 3050 6GB easily, bf16)
# --------------------------------------------------------------------------- #
N_LAYER = 6
N_EMBD = 512
N_HEAD = 8
BLOCK_SIZE = 512
DROPOUT = 0.1
VOCAB_SIZE = 4096                    # byte-level BPE vocab cap

# --------------------------------------------------------------------------- #
# Training schedule
# --------------------------------------------------------------------------- #
MAX_STEPS = 50000
CHECKPOINT_INTERVAL = 100            # save checkpoints/latest.pt every 100 steps
RANDOM_SEED = int(os.environ.get("SELFLEARN_RANDOM_SEED") or 1337)

# Validation / "well-trained" completion criterion ------------------------- #
VAL_SPLIT = 0.10                     # fraction of blocks held out
VAL_INTERVAL_STEPS = 2000            # evaluate every N train steps (cheap cadence)
VAL_MIN_IMPROVEMENT = 0.0005         # must beat best val by >= this to count
VAL_FLOOR_FIRE = 8                   # N evals without improvement => converged
TRAIN_WINDOWS_PER_STEP = 16          # random windows sampled per step

# --------------------------------------------------------------------------- #
# Execution pipeline (optimization experiment). Architecture/tokenizer/
# optimizer/LR/dataset are frozen; these knobs tune throughput only.
# --------------------------------------------------------------------------- #
CONTEXT = 256                        # fixed sequence length per window
MICRO_BATCH = 4                      # windows per GPU fwd/bwd
GRAD_ACCUM = 4                       # micro-steps before one optimizer step
EFFECTIVE_BATCH = MICRO_BATCH * GRAD_ACCUM   # 16 (matches prior batch)
DATALOADER_WORKERS = 2               # benchmark 2 vs 4 (start=2, conservative)
PREFETCH_FACTOR = 4
PIN_MEMORY = True
LOADER_SEED = int(os.environ.get("SELFLEARN_LOADER_SEED") or 4242)   # base seed for the persistent sampler

CHECKPOINT_INTERVAL = 250            # latest.pt recovery cadence
METRICS_INTERVAL = 100               # log throughput/system metrics every N steps
THROTTLE_MAX_PAUSE = 0.30            # ceiling for adaptive pause (s)
CPU_HIGH_PERCENT = 85.0              # throttle trigger for CPU convoy
GPU_TEMP_THROTTLE = 68.0             # adaptive pause kicks in at >= this temp

USE_TORCH_COMPILE = False            # set True only after benchmark confirms win
NUMERICAL_STABILITY_STEPS = 20       # compile-correctness pre-check window

# --------------------------------------------------------------------------- #
# Data acquisition / filtering (data_engine.py)
# --------------------------------------------------------------------------- #
TARGET_URLS = [
    # Wikipedia REST summaries (JSON, text "extract" field)
    "https://en.wikipedia.org/api/rest_v1/page/summary/Artificial_intelligence",
    "https://en.wikipedia.org/api/rest_v1/page/summary/Climate_change",
    "https://en.wikipedia.org/api/rest_v1/page/summary/Thermodynamics",
    "https://en.wikipedia.org/api/rest_v1/page/summary/Black_hole",
    "https://en.wikipedia.org/api/rest_v1/page/summary/Quantum_mechanics",
    "https://en.wikipedia.org/api/rest_v1/page/summary/Photosynthesis",
    "https://en.wikipedia.org/api/rest_v1/page/summary/Roman_Empire",
    "https://en.wikipedia.org/api/rest_v1/page/summary/DNA",
    "https://en.wikipedia.org/api/rest_v1/page/summary/Machine_learning",
    "https://en.wikipedia.org/api/rest_v1/page/summary/Ancient_Greece",
    "https://en.wikipedia.org/api/rest_v1/page/summary/Evolution",
    "https://en.wikipedia.org/api/rest_v1/page/summary/Renaissance",
    "https://en.wikipedia.org/api/rest_v1/page/summary/Plate_tectonics",
    "https://en.wikipedia.org/api/rest_v1/page/summary/Language",
    "https://en.wikipedia.org/api/rest_v1/page/summary/Monarchy",
    "https://en.wikipedia.org/api/rest_v1/page/summary/Electromagnetism",
]

# Public-domain books (Gutenberg plain text, UTF-8) -> real prose scale
GUTENBERG_URLS = [
    "https://www.gutenberg.org/cache/epub/84/pg84.txt",    # Frankenstein
    "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",  # Pride & Prejudice
    "https://www.gutenberg.org/cache/epub/98/pg98.txt",    # A Tale of Two Cities
    "https://www.gutenberg.org/cache/epub/11/pg11.txt",    # Alice in Wonderland
    "https://www.gutenberg.org/cache/epub/2701/pg2701.txt",  # Moby Dick
    "https://www.gutenberg.org/cache/epub/1661/pg1661.txt",  # Sherlock Holmes
    "https://www.gutenberg.org/cache/epub/1952/pg1952.txt",  # The Jungle Book
]

MAX_CONCURRENCY = 4
REQUEST_TIMEOUT_S = 20.0
REQUEST_RETRIES = 2
MAX_FETCH_BYTES = 8 * 1024 * 1024    # hard safety cap per source
USER_AGENT = ("MiniSelfLearner/2.0 (educational pipeline; "
              "contact: admin@localhost)")

MIN_WORDS = 20
MAX_WORDS = 1000                     # chunks are split at this word bound
CHUNK_OVERLAP = 50                   # words of overlap between long-text chunks
ENTROPY_RATIO = 0.35
NOISE_TERMS = (
    "cookie policy", "privacy policy", "404 not found", "you are using an",
    "accept all cookies", "sign in to continue", "terms of service",
    "subscribe to our", "javascript is disabled",
    "project gutenberg", "this ebook", "end of the project gutenberg",
    "copyright", "gutenbergtm",
)
SCRAPE_MIN_INTERVAL_S = 300
DISCOVERY_ROUND_BATCH = 4            # sources harvested per plateau event

# --------------------------------------------------------------------------- #
# Plateau detector (self_learner.py)
# --------------------------------------------------------------------------- #
PLATEAU_ROLLING_WINDOW = 50
PLATEAU_IMPROVEMENT = 0.01
PLATEAU_STEPS = 100

LOG_FILE = LOGS_DIR / "runtime.log"