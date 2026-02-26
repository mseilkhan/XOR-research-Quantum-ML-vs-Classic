from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Any, List, Optional, Sequence, Tuple
import numpy as np

from tqdm import tqdm

from core.data.xor_dataset import DatasetSplit, make_split
from core.train.trainer import Trainer, TrainConfig
from core.models.base import BaseBinaryClassifier


MODEL_SEEDS_DEFAULT = [0, 1, 2, 3, 4]


@dataclass(frozen=True)
class RunRecord:
    """Single run record (one model seed)."""
    model_name: str
    model_seed: int
    dataset_name: str
    sigma: Optional[float]
    n_per_cluster: Optional[int]
    shots: Optional[int]
    L: Optional[int]
    h: Optional[int]
    train_acc: float
    test_acc: float
    train_loss: float
    test_loss: float
    n_params: int
    train_seconds: float


def _mean_std(xs: Sequence[float]) -> Tuple[float, float]:
    arr = np.array(xs, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0))


def run_repeated(
    *,
    model_factory: Callable[[int], BaseBinaryClassifier],
    split: DatasetSplit,
    dataset_name: str,
    model_name: str,
    train_cfg: TrainConfig,
    model_seeds: Sequence[int] = MODEL_SEEDS_DEFAULT,
    meta: Dict[str, Any],
    progress_desc: Optional[str] = None,
    logger=None,
) -> Tuple[List[RunRecord], Dict[str, Any]]:

    """
    Run 5 seeds (0..4) for a fixed dataset configuration.
    Returns (raw_records, summary_dict(mean±std)).
    """
    trainer = Trainer(train_cfg)
    records: List[RunRecord] = []

    seed_iter = tqdm(list(model_seeds), desc=progress_desc, leave=False) if progress_desc else model_seeds
    for s in seed_iter:
        if logger:
            logger.info(f"[run_repeated] model={model_name} seed={int(s)} meta={meta}")

        model = model_factory(int(s))
        fit_res = trainer.fit(model, split.X_train, split.y_train, split.X_test, split.y_test)

        final = Trainer.final_metrics(model, split.X_train, split.y_train, split.X_test, split.y_test)

        records.append(RunRecord(
            model_name=model_name,
            model_seed=int(s),
            dataset_name=dataset_name,
            sigma=meta.get("sigma"),
            n_per_cluster=meta.get("n_per_cluster"),
            shots=meta.get("shots"),
            L=meta.get("L"),
            h=meta.get("h"),
            train_acc=final["train_acc"],
            test_acc=final["test_acc"],
            train_loss=final["train_loss"],
            test_loss=final["test_loss"],
            n_params=int(fit_res.n_params),
            train_seconds=float(fit_res.train_seconds),
        ))

    summary = {
        **meta,
        "model_name": model_name,
        "dataset_name": dataset_name,
        "train_acc_mean": _mean_std([r.train_acc for r in records])[0],
        "train_acc_std": _mean_std([r.train_acc for r in records])[1],
        "test_acc_mean": _mean_std([r.test_acc for r in records])[0],
        "test_acc_std": _mean_std([r.test_acc for r in records])[1],
        "train_loss_mean": _mean_std([r.train_loss for r in records])[0],
        "train_loss_std": _mean_std([r.train_loss for r in records])[1],
        "test_loss_mean": _mean_std([r.test_loss for r in records])[0],
        "test_loss_std": _mean_std([r.test_loss for r in records])[1],
        "n_params_mean": _mean_std([r.n_params for r in records])[0],
        "n_params_std": _mean_std([r.n_params for r in records])[1],
        "train_seconds_mean": _mean_std([r.train_seconds for r in records])[0],
        "train_seconds_std": _mean_std([r.train_seconds for r in records])[1],
    }
    return records, summary


# ---------- Sweep helpers required by methodology ----------

def sweep_noise_datasetB(
    *,
    sigmas: Sequence[float],
    n_per_cluster: int,
    model_factory: Callable[[int], BaseBinaryClassifier],
    model_name: str,
    train_cfg: TrainConfig,
    data_seed: int = 42,
    split_seed: int = 42,
    model_seeds: Sequence[int] = MODEL_SEEDS_DEFAULT,
    logger=None,
) -> Tuple[List[RunRecord], List[Dict[str, Any]]]:

    """Accuracy vs sigma on Dataset B (mean±std across model seeds)."""
    all_records: List[RunRecord] = []
    summaries: List[Dict[str, Any]] = []

    for sigma in tqdm(list(sigmas), desc=f"{model_name}: noise sweep (sigmas)"):
        if logger:
            logger.info(f"[sweep_noise_datasetB] model={model_name} sigma={float(sigma)} n={int(n_per_cluster)}")
        split = make_split("B", sigma=float(sigma), n_per_cluster=int(n_per_cluster),
                           data_seed=data_seed, split_seed=split_seed)
        meta = {"sigma": float(sigma), "n_per_cluster": int(n_per_cluster)}
        rec, summ = run_repeated(
            model_factory=model_factory,
            split=split,
            dataset_name="B",
            model_name=model_name,
            train_cfg=train_cfg,
            model_seeds=model_seeds,
            meta=meta,
            progress_desc=f"{model_name}: seeds @ sigma={float(sigma):.2f}",
            logger=logger,
        )
        all_records.extend(rec)
        summaries.append(summ)

    return all_records, summaries


def sweep_size_datasetB(
    *,
    sizes: Sequence[int],
    sigma: float,
    model_factory: Callable[[int], BaseBinaryClassifier],
    model_name: str,
    train_cfg: TrainConfig,
    data_seed: int = 42,
    split_seed: int = 42,
    model_seeds: Sequence[int] = MODEL_SEEDS_DEFAULT,
    logger=None,
) -> Tuple[List[RunRecord], List[Dict[str, Any]]]:

    """Accuracy vs n_per_cluster on Dataset B (mean±std across model seeds)."""
    all_records: List[RunRecord] = []
    summaries: List[Dict[str, Any]] = []

    for n_per_cluster in tqdm(list(sizes), desc=f"{model_name}: size sweep (n_per_cluster)"):
        if logger:
            logger.info(f"[sweep_size_datasetB] model={model_name} sigma={float(sigma)} n={int(n_per_cluster)}")
        split = make_split("B", sigma=float(sigma), n_per_cluster=int(n_per_cluster),
                           data_seed=data_seed, split_seed=split_seed)
        meta = {"sigma": float(sigma), "n_per_cluster": int(n_per_cluster)}
        rec, summ = run_repeated(
            model_factory=model_factory,
            split=split,
            dataset_name="B",
            model_name=model_name,
            train_cfg=train_cfg,
            model_seeds=model_seeds,
            meta=meta,
            progress_desc=f"{model_name}: seeds @ n={int(n_per_cluster)}",
            logger=logger,
        )
        all_records.extend(rec)
        summaries.append(summ)

    return all_records, summaries
