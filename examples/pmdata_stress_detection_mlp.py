"""Daily stress detection on PMData with an MLP, with ablations.

Contributors:
    Brian Zhao (NetID: bnzhao2, bnzhao2@illinois.edu)
    Wanting Mao (NetID: wmao8, wmao8@illinois.edu)
    Aditya Radhakrishnan (NetID: aditya63, aditya63@illinois.edu)

Paper (reproduced):
    Zeyu Yang, Han Yu, Akane Sano. "Contrastive Pretraining for Stress
    Detection with Multimodal Wearable Sensor Data and Surveys." Conference
    on Health, Inference, and Learning (CHIL), PMLR 287, 2025.
    https://proceedings.mlr.press/v287/yang25a.html

Pipeline:
    PMDataDataset -> PMDataStressDetection -> MLP (pyhealth.models) ->
    Trainer, evaluated with ROC-AUC as in the paper. The paper's supervised
    multimodal baseline reaches 80.81 test AUC on PMData with a random
    day-level split; this script re-creates that setting and then varies the
    task.

Ablations (none of them are in the paper):
    1. Feature groups of the ``survey`` input (the hourly ``wearable`` input
       is always used): all groups (the paper's setting); all but the
       same-day ``wellness`` answers (fatigue, readiness, sleep, soreness),
       which come from the same questionnaire as the stress label; only
       passively sensed Fitbit data plus demographics; demographics only.
    2. Label definition: stressed means a score above 3 (paper) or above 2.
    3. Split: random days (participants shared between train and test, the
       paper's main setting) or unseen participants (PatientSplit).

Setup:
    60/20/20 train/validation/test split; the model with the best
    validation AUC is evaluated on the test part. Processors (z-scoring and
    imputation) are fitted on the training part only. Each configuration is
    repeated with ``--seeds`` seeds; the table shows mean and standard
    deviation of test AUC.

Results on the real PMData (1,688 labeled days; 5 seeds, 20 epochs,
hidden_dim 64; test ROC-AUC mean ± std):

    ========================== =============== ===================
    task                       random days     unseen participants
    ========================== =============== ===================
    all features (paper)       0.821 ± 0.019   0.557 ± 0.084
    no same-day wellness       0.796 ± 0.012   0.511 ± 0.073
    passive sensing only       0.793 ± 0.010   0.496 ± 0.086
    demographics only          0.723 ± 0.022   0.323 ± 0.080
    all features, stress > 2   0.702 ± 0.027   0.599 ± 0.065
    ========================== =============== ===================

Findings:
    - The paper's setting reproduces: 0.821 with random day splits, against
      the paper's 80.81 (supervised) and 81.14 (contrastive pretraining).
    - The self-reported answers from the stress questionnaire add little
      (0.821 vs 0.796), and passive Fitbit data plus demographics reach
      0.793 without any self-report.
    - Demographics with the hourly data already reach 0.723 on random days
      but 0.323 on unseen participants. Demographics act as a participant
      ID, so on random days much of the score comes from recognising the
      participant and their usual stress level, which does not carry over
      to new people.
    - For unseen participants every setting is near chance (0.50-0.60),
      far from the 89.36 the paper reports for its participant-level
      split. A 20% test part holds only three of the 16 participants, so
      these numbers are noisy, but none comes close.
    - Labeling scores above 2 as stressed (a much more common class) makes
      random-day detection harder (0.702).

Usage:
    # Smoke test on synthetic data (results are not meaningful):
    python examples/pmdata_stress_detection_mlp.py
    # Real data (folder holding p01 ... p16):
    python examples/pmdata_stress_detection_mlp.py --root /path/to/pmdata
"""

import argparse
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from pyhealth.datasets import PatientSplit, PMDataDataset, Split, get_dataloader
from pyhealth.models import MLP
from pyhealth.tasks import PMDataStressDetection
from pyhealth.trainer import Trainer

ALL_GROUPS = [
    "demographics",
    "activity",
    "sleep",
    "training",
    "nutrition",
    "injury",
    "wellness",
]

#: Task configurations of the ablation, by name.
CONFIGS: Dict[str, Dict] = {
    "all features (paper)": {"feature_groups": ALL_GROUPS},
    "no same-day wellness": {
        "feature_groups": [g for g in ALL_GROUPS if g != "wellness"]
    },
    "passive sensing only": {"feature_groups": ["demographics", "activity", "sleep"]},
    "demographics only": {"feature_groups": ["demographics"]},
    "all features, stress > 2": {
        "feature_groups": ALL_GROUPS,
        "stress_threshold": 2,
    },
}


class DaySplit(Split):
    """Random split of samples (days), ignoring which participant they are.

    This is the paper's "random splitting": a participant's days can be in
    both the training and the test part.

    Args:
        ratios: Fractions of the parts, training part first.
        seed: Seed for the shuffle.
    """

    def __init__(self, ratios: Sequence[float], seed: int = 0) -> None:
        self.ratios = tuple(float(r) for r in ratios)
        self.seed = seed

    def split_indices(
        self, patient_to_index: Mapping[str, Sequence[int]]
    ) -> List[np.ndarray]:
        """Shuffles all sample indices and cuts them by ``ratios``.

        Args:
            patient_to_index: Patient id to the indices of its samples.

        Returns:
            One sorted index array per part.
        """
        indices = np.array(
            sorted(i for group in patient_to_index.values() for i in group),
            dtype=np.int64,
        )
        shuffled = np.random.default_rng(self.seed).permutation(indices)
        cuts = np.cumsum([int(len(shuffled) * r) for r in self.ratios[:-1]])
        return [np.sort(part) for part in np.split(shuffled, cuts)]

    def to_dict(self) -> dict:
        """Describes the split for PyHealth's cache key.

        Returns:
            A JSON-serialisable description.
        """
        return {"kind": "day", "ratios": list(self.ratios), "seed": self.seed}


def write_synthetic_pmdata(root: Path, n_participants: int, n_days: int) -> None:
    """Writes random PMData-style raw files for a smoke test.

    Stress is made to depend weakly on steps and fatigue so the pipeline has
    something to learn, but the numbers mean nothing.

    Args:
        root: Folder to write ``p01``, ``p02``, ... into.
        n_participants: Number of synthetic participants.
        n_days: Days of data per participant.
    """
    rng = np.random.default_rng(0)
    for p in range(1, n_participants + 1):
        folder = root / f"p{p:02d}"
        (folder / "fitbit").mkdir(parents=True)
        (folder / "pmsys").mkdir()
        days = pd.date_range("2019-11-01", periods=n_days, freq="D")
        hours = pd.date_range(days[0], periods=24 * n_days, freq="h")
        steps = rng.poisson(400, len(hours))
        for name, values in [
            ("steps", steps),
            ("calories", rng.normal(100, 10, len(hours)).round(1)),
            ("distance", (steps * 70).astype(float)),
        ]:
            records = [
                {"dateTime": str(t), "value": str(v)} for t, v in zip(hours, values)
            ]
            (folder / "fitbit" / f"{name}.json").write_text(json.dumps(records))
        heart = [
            {"dateTime": str(t), "value": {"bpm": int(v), "confidence": 3}}
            for t, v in zip(hours, rng.normal(70, 8, len(hours)))
        ]
        (folder / "fitbit" / "heart_rate.json").write_text(json.dumps(heart))
        sedentary = [
            {"dateTime": str(d), "value": str(int(v))}
            for d, v in zip(days, rng.normal(600, 60, n_days))
        ]
        (folder / "fitbit" / "sedentary_minutes.json").write_text(json.dumps(sedentary))
        daily_steps = steps.reshape(n_days, 24).sum(axis=1)
        fatigue = rng.integers(1, 6, n_days)
        score = 3 + (fatigue - 3) * 0.5 + (daily_steps < daily_steps.mean())
        stress = np.clip(np.round(score + rng.normal(0, 0.7, n_days)), 1, 5)
        pd.DataFrame(
            {
                "effective_time_frame": [f"{d.date()}T08:00:00.000Z" for d in days],
                "fatigue": fatigue,
                "mood": rng.integers(1, 6, n_days),
                "readiness": rng.integers(1, 11, n_days),
                "sleep_duration_h": rng.integers(5, 10, n_days),
                "sleep_quality": rng.integers(1, 6, n_days),
                "soreness": rng.integers(1, 6, n_days),
                "soreness_area": "[]",
                "stress": stress.astype(int),
            }
        ).to_csv(folder / "pmsys" / "wellness.csv", index=False)


def run_once(
    dataset: PMDataDataset,
    task: PMDataStressDetection,
    split: Split,
    seed: int,
    epochs: int,
    hidden_dim: int,
) -> Optional[float]:
    """Trains an MLP on one split and returns its test ROC-AUC.

    Args:
        dataset: The PMData dataset.
        task: The task configuration.
        split: How to split samples into train/validation/test.
        seed: Seed for model initialisation.
        epochs: Training epochs.
        hidden_dim: MLP hidden size.

    Returns:
        Test ROC-AUC, or ``None`` if a part holds only one class.
    """
    train, val, test = dataset.set_task(task, split=split, num_workers=1)
    for part in (train, val, test):
        if len({int(s["stress"].item()) for s in part}) < 2:
            return None
    torch.manual_seed(seed)
    model = MLP(dataset=train, hidden_dim=hidden_dim)
    trainer = Trainer(model=model, metrics=["roc_auc"], enable_logging=False)
    trainer.train(
        train_dataloader=get_dataloader(train, batch_size=64, shuffle=True),
        val_dataloader=get_dataloader(val, batch_size=256),
        epochs=epochs,
        monitor="roc_auc",
        optimizer_params={"lr": 1e-3},
    )
    return trainer.evaluate(get_dataloader(test, batch_size=256))["roc_auc"]


def main() -> None:
    """Runs every configuration and split, then prints a results table."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", help="PMData folder; synthetic data if unset")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=64)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        root = args.root
        min_hours = 24
        if root is None:
            print("SMOKE TEST on synthetic data; results are not meaningful.")
            root = str(Path(tmp) / "pmdata")
            write_synthetic_pmdata(Path(root), n_participants=6, n_days=20)
            min_hours = 0
        dataset = PMDataDataset(root=root, cache_dir=str(Path(tmp) / "cache"))

        splits = {
            "random days": lambda seed: DaySplit((0.6, 0.2, 0.2), seed),
            "unseen participants": lambda seed: PatientSplit((0.6, 0.2, 0.2), seed),
        }
        rows = []
        for config_name, config in CONFIGS.items():
            task = PMDataStressDetection(min_hours=min_hours, **config)
            for split_name, make_split in splits.items():
                aucs = [
                    run_once(
                        dataset,
                        task,
                        make_split(seed),
                        seed,
                        args.epochs,
                        args.hidden_dim,
                    )
                    for seed in range(args.seeds)
                ]
                aucs = [a for a in aucs if a is not None]
                rows.append(
                    {
                        "task": config_name,
                        "split": split_name,
                        "auc_mean": np.mean(aucs) if aucs else np.nan,
                        "auc_std": np.std(aucs) if aucs else np.nan,
                        "runs": len(aucs),
                    }
                )
                print(rows[-1])

    print("\nTest ROC-AUC (mean ± std over seeds)")
    table = pd.DataFrame(rows)
    print(
        table.pivot(index="task", columns="split", values="auc_mean")
        .round(3)
        .to_string()
    )
    print(
        table.pivot(index="task", columns="split", values="auc_std")
        .round(3)
        .to_string()
    )


if __name__ == "__main__":
    main()
