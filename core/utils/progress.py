from __future__ import annotations

import time


class SimpleProgress:
    """
    Minimal progress helper (no external deps).
    Prints one-line progress with elapsed + ETA.
    """
    def __init__(self, total: int, prefix: str = ""):
        self.total = max(1, int(total))
        self.prefix = prefix
        self.t0 = time.perf_counter()

    def update(self, i: int, msg: str = "") -> None:
        i = min(self.total, max(0, int(i)))
        elapsed = time.perf_counter() - self.t0
        rate = elapsed / max(1, i)
        eta = rate * (self.total - i)
        print(f"{self.prefix}{i}/{self.total} | elapsed {elapsed:6.1f}s | ETA {eta:6.1f}s | {msg}", end="\r")

    def done(self, msg: str = "") -> None:
        elapsed = time.perf_counter() - self.t0
        print(f"{self.prefix}{self.total}/{self.total} | elapsed {elapsed:6.1f}s | {msg}".ljust(120))
