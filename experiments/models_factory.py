from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from core.models import LogisticRegression2D, MLP1Hidden, VQC2Q, VQCConfig


@dataclass(frozen=True)
class ModelSpec:
    """Lightweight descriptor for tables/filenames."""
    name: str


def make_linear() -> tuple[ModelSpec, Callable[[int], LogisticRegression2D]]:
    return ModelSpec("Linear"), (lambda seed: LogisticRegression2D(model_seed=seed))


def make_mlp(h: int) -> tuple[ModelSpec, Callable[[int], MLP1Hidden]]:
    return ModelSpec(f"MLP(h={h})"), (lambda seed: MLP1Hidden(model_seed=seed, h=h))


def make_vqc(L: int, shots: Optional[int]) -> tuple[ModelSpec, Callable[[int], VQC2Q]]:
    label = f"VQC(L={L},{'analytic' if shots is None else f'shots={shots}'})"
    cfg = VQCConfig(L=L, shots=shots)
    return ModelSpec(label), (lambda seed: VQC2Q(model_seed=seed, cfg=cfg))
