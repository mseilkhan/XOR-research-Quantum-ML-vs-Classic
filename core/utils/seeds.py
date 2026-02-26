from __future__ import annotations

import os
import random
import numpy as np


def set_global_seed(seed: int) -> None:
    """Set seeds for reproducibility (Python + NumPy)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
