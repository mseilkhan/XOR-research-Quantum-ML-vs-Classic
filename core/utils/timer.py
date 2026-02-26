from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class Timer:
    """Simple wall-clock timer context manager."""
    start: Optional[float] = None
    end: Optional[float] = None

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        self.end = None
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.end = time.perf_counter()

    @property
    def seconds(self) -> float:
        if self.start is None:
            return 0.0
        if self.end is None:
            return time.perf_counter() - self.start
        return self.end - self.start
