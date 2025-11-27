import os
from pathlib import Path

import dotenv

_env_file_path = Path().home() / ".config" / "becominglit" / ".env"
dotenv.load_dotenv(_env_file_path, override=True)


BECOMINGLIT_DATASET_PATH = os.getenv("BECOMINGLIT_DATASET_PATH")
assert BECOMINGLIT_DATASET_PATH is not None, (
    f"Environment variable BECOMINGLIT_DATASET_PATH not set. Please set it in {_env_file_path}"
)

BECOMINGLIT_EXPERIMENT_PATH = os.getenv("BECOMINGLIT_EXPERIMENT_PATH")
assert BECOMINGLIT_EXPERIMENT_PATH is not None, (
    f"Environment variable BECOMINGLIT_EXPERIMENT_PATH not set. Please set it in {_env_file_path}"
)

BECOMINGLIT_FLAME_TRACKING_PATH = os.getenv("BECOMINGLIT_FLAME_TRACKING_PATH")
assert BECOMINGLIT_FLAME_TRACKING_PATH is not None, (
    f"Environment variable BECOMINGLIT_FLAME_TRACKING_PATH not set. Please set it in {_env_file_path}"
)

IMAGE_FILE_FORMAT = os.getenv("IMAGE_FILE_FORMAT", "avif")
