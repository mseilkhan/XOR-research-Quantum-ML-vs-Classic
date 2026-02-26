from __future__ import annotations

import os
import random
from typing import Dict

import numpy as np


def set_global_determinism(seed: int) -> Dict[str, int]:
    """
    Enforce deterministic behavior across Python, NumPy and supported libraries.

    Returns a dict of applied seeds for logging/metadata.
    """
    # Python
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # Hash-based ops (best effort)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # PennyLane (best effort)
    try:
        import pennylane as qml

        # Some devices read numpy RNG implicitly; seeding numpy above is essential.
        # PennyLane also supports explicit seeding via qml.numpy.random
        try:
            qml.numpy.random.seed(seed)
        except Exception:
            pass
    except Exception:
        pass

    return {
        "python": seed,
        "numpy": seed,
        "python_hash": seed,
        "pennylane_numpy": seed,
    }