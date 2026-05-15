from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "fma_metadata"
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "model"
REPORT_DIR = ROOT / "reports"
DB_PATH = DATA_DIR / "musics.db"

RANDOM_STATE = 42

# Synthetic data config
DEFAULT_USER_COUNT = 1200
DEFAULT_INTERACTIONS_PER_USER = 35
DEFAULT_SONG_LIMIT = 12000

# Model config
TARGET = "listened"
