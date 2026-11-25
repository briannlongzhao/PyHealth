"""Full pipeline: PMData stress detection with CLSP pretraining, with ablations.

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
    PMDataDataset -> PMDataStressDetection(include_text=True) -> CLSP
    (contrastive pretraining against BERT embeddings of each day's text
    description) -> linear probe or fine-tuning -> test ROC-AUC.

Experiments:
    Reproduction (paper Table 1, PMData): the same CLSP network trained
    with labels only ("supervised", the paper's supervised multimodal
    baseline) against contrastive pretraining followed by fine-tuning or a
    linear probe. Run with random day splits (the paper's main setting) and
    with unseen participants.

    Ablations (not in the paper), random day splits:
      - soft contrastive targets (the authors' code) instead of the
        one-to-one targets of the paper's Eq. 1;
      - text descriptions without the hourly statistics lines;
      - zero-shot prediction: no labels at all, each day is compared with
        "label: relaxed" and "label: stressed" in the shared space;
      - label scarcity: only 10% of the training days keep their label for
        fine-tuning, while pretraining uses all training days.

Setup:
    60/20/20 split; pretraining 50 epochs (batch 128), then 30 epochs of
    fine-tuning or linear probing (batch 64) keeping the best validation
    AUC; Adam, lr 1e-3; hidden and projection size 64; temperature 0.07.
    Every number is the mean ± std of test ROC-AUC over ``--seeds`` seeds.

Results on the real PMData (1,688 labeled days, 5 seeds, test ROC-AUC):

    ============================================ ============= =============
    experiment                                   random days   unseen part.
    ============================================ ============= =============
    supervised (no pretraining)                  0.804 ± 0.029 0.547 ± 0.127
    CLSP + linear probe                          0.743 ± 0.020 0.568 ± 0.117
    CLSP + fine-tuning                           0.824 ± 0.033 0.522 ± 0.104
    CLSP (soft targets) + fine-tuning            0.803 ± 0.017
    CLSP (text without statistics) + fine-tuning 0.813 ± 0.015
    zero-shot (no labels)                        0.463 ± 0.051
    10% labels: supervised                       0.684 ± 0.063
    10% labels: CLSP + fine-tuning               0.694 ± 0.051
    ============================================ ============= =============

Findings:
    - The paper's PMData result reproduces: pretraining then fine-tuning
      beats the same network trained with labels only, 0.824 vs 0.804
      (+0.020, better in 4 of 5 seeds), close to the paper's 81.14 vs
      80.81. The gain is smaller than the seed-to-seed spread.
    - A linear probe on the frozen pretrained encoder (0.743) is well
      below fine-tuning, so the contrastive features alone are weaker than
      supervised ones here.
    - For unseen participants neither model is above chance in a stable
      way (0.52-0.57, std > 0.1), and pretraining does not help (better in
      1 of 5 seeds); the paper reports 89.36 for its participant-level
      split.
    - The paper's one-to-one targets beat the code's soft targets (0.824 vs
      0.803, better in 4 of 5 seeds); dropping the hourly statistics from
      the text costs 0.011.
    - Zero-shot matching to "label: stressed" is at chance (0.463): the
      descriptions never mention stress, so the text space carries no label
      information.
    - With 10% of the labels, pretraining on all training days adds 0.010
      (3 of 5 seeds): PMData has too few unlabeled days for contrastive
      pretraining to help much, as the paper notes.

Usage:
    # Smoke test on synthetic data (results are not meaningful):
    python examples/pmdata_stress_detection_clsp.py
    # Real data (folder holding p01 ... p16):
    python examples/pmdata_stress_detection_clsp.py --root /path/to/pmdata
"""

import argparse
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from pmdata_stress_detection_mlp import DaySplit, write_synthetic_pmdata

from pyhealth.datasets import (
    PatientSplit,
    PMDataDataset,
    SampleDataset,
    Split,
    get_dataloader,
)
from pyhealth.metrics import binary_metrics_fn
from pyhealth.models import CLSP
from pyhealth.models.clsp import FrozenTextEncoder
from pyhealth.tasks import PMDataStressDetection
from pyhealth.trainer import Trainer


class LabeledSubsetSplit(Split):
    """Adds a labeled subset of the training part to another split.

    Returns ``[train, labeled_train, val, test]``: the base split's parts
    with a random ``fraction`` of the training samples inserted second.
    Processors are fitted on the whole training part.

    Args:
        base: The split to extend; its parts are ``[train, val, test]``.
        fraction: Share of training samples kept for supervised training.
        seed: Seed for choosing the subset.
    """

    def __init__(self, base: Split, fraction: float, seed: int = 0) -> None:
        self.base = base
        self.fraction = fraction
        self.seed = seed

    def split_indices(
        self, patient_to_index: Mapping[str, Sequence[int]]
    ) -> List[np.ndarray]:
        """Splits indices and draws the labeled training subset.

        Args:
            patient_to_index: Patient id to the indices of its samples.

        Returns:
            ``[train, labeled_train, val, test]`` index arrays.
        """
        train, val, test = self.base.split_indices(patient_to_index)
        size = max(2, int(round(len(train) * self.fraction)))
        rng = np.random.default_rng(self.seed)
        labeled = np.sort(rng.choice(train, size=size, replace=False))
        return [train, labeled, val, test]

    def to_dict(self) -> dict:
        """Describes the split for PyHealth's cache key.

        Returns:
            A JSON-serialisable description.
        """
        return {
            "kind": "labeled_subset",
            "base": self.base.to_dict(),
            "fraction": self.fraction,
            "seed": self.seed,
        }


class Experiment:
    """Builds, pretrains and evaluates CLSP models with shared settings.

    Args:
        text_encoder: Shared frozen text encoder.
        pretrain_epochs: Contrastive pretraining epochs.
        epochs: Fine-tuning / linear probing epochs.
    """

    def __init__(
        self, text_encoder: Callable, pretrain_epochs: int, epochs: int
    ) -> None:
        self.text_encoder = text_encoder
        self.pretrain_epochs = pretrain_epochs
        self.epochs = epochs

    def new_model(self, train: SampleDataset, seed: int, **kwargs) -> CLSP:
        """Builds a CLSP model with the shared text encoder.

        Args:
            train: Training samples.
            seed: Seed for the initial weights.
            **kwargs: Extra CLSP arguments (e.g. ``soft_targets``).

        Returns:
            The model.
        """
        torch.manual_seed(seed)
        return CLSP(train, text_encoder=self.text_encoder, **kwargs)

    def pretrain(self, train: SampleDataset, seed: int, **kwargs) -> CLSP:
        """Pretrains a new model contrastively on all training samples.

        Args:
            train: Training samples (labels are not used).
            seed: Seed for weights and batch order.
            **kwargs: Extra CLSP arguments.

        Returns:
            The pretrained model.
        """
        model = self.new_model(train, seed, stage="pretrain", **kwargs)
        Trainer(model=model, enable_logging=False).train(
            train_dataloader=get_dataloader(train, batch_size=128, shuffle=True),
            epochs=self.pretrain_epochs,
            optimizer_params={"lr": 1e-3},
        )
        return model

    def copy(self, model: CLSP, train: SampleDataset, **kwargs) -> CLSP:
        """Copies a model's weights into a new model (sharing the encoder).

        Args:
            model: The model to copy.
            train: Training samples.
            **kwargs: The CLSP arguments the model was built with.

        Returns:
            The copy.
        """
        clone = self.new_model(train, 0, **kwargs)
        clone.load_state_dict(model.state_dict())
        return clone

    def train_labels(
        self,
        model: CLSP,
        stage: str,
        labeled: SampleDataset,
        val: SampleDataset,
        test: SampleDataset,
    ) -> float:
        """Trains with labels in ``stage`` and returns the test ROC-AUC.

        Args:
            model: The model, pretrained or not.
            stage: ``"finetune"`` or ``"linear_probe"``.
            labeled: Labeled training samples.
            val: Validation samples for choosing the best epoch.
            test: Test samples.

        Returns:
            Test ROC-AUC of the best validation epoch.
        """
        model.set_stage(stage)
        trainer = Trainer(model=model, metrics=["roc_auc"], enable_logging=False)
        trainer.train(
            train_dataloader=get_dataloader(labeled, batch_size=64, shuffle=True),
            val_dataloader=get_dataloader(val, batch_size=256),
            epochs=self.epochs,
            monitor="roc_auc",
            optimizer_params={"lr": 1e-3},
        )
        return trainer.evaluate(get_dataloader(test, batch_size=256))["roc_auc"]

    @staticmethod
    def zero_shot(model: CLSP, test: SampleDataset) -> float:
        """Scores the test samples by similarity to the two class texts.

        Args:
            model: A pretrained model.
            test: Test samples.

        Returns:
            Test ROC-AUC of the "label: stressed" probability.
        """
        model.eval()
        y_true, y_prob = [], []
        for batch in get_dataloader(test, batch_size=256):
            y_prob.append(model.zero_shot(**batch)[:, 1].cpu())
            y_true.append(batch["stress"].view(-1).cpu())
        return binary_metrics_fn(
            torch.cat(y_true).numpy(), torch.cat(y_prob).numpy(), ["roc_auc"]
        )["roc_auc"]


def has_both_classes(*parts: SampleDataset) -> bool:
    """Returns whether every part holds both labels (AUC is defined).

    Args:
        *parts: Sample datasets.

    Returns:
        True if each part has stressed and relaxed samples.
    """
    return all(len({int(s["stress"].item()) for s in p}) == 2 for p in parts)


def run_seed(
    dataset: PMDataDataset,
    experiment: Experiment,
    seed: int,
    min_hours: int,
) -> List[Dict]:
    """Runs every experiment for one seed.

    Args:
        dataset: The PMData dataset.
        experiment: Shared training settings.
        seed: Seed for splits and weights.
        min_hours: Least hours of wearable data per day.

    Returns:
        One row per experiment with its test ROC-AUC.
    """
    rows: List[Dict] = []

    def add(name: str, split: str, auc: Optional[float]) -> None:
        rows.append({"experiment": name, "split": split, "auc": auc})
        print(f"seed {seed} | {split:19s} | {name:40s} | {auc}")

    task = PMDataStressDetection(include_text=True, min_hours=min_hours)
    splits = {
        "random days": DaySplit((0.6, 0.2, 0.2), seed),
        "unseen participants": PatientSplit((0.6, 0.2, 0.2), seed),
    }
    for split_name, split in splits.items():
        train, val, test = dataset.set_task(task, split=split, num_workers=1)
        if not has_both_classes(train, val, test):
            continue
        supervised = experiment.new_model(train, seed)
        add(
            "supervised (no pretraining)",
            split_name,
            experiment.train_labels(supervised, "finetune", train, val, test),
        )
        pretrained = experiment.pretrain(train, seed)
        if split_name == "random days":
            add(
                "zero-shot (no labels)",
                split_name,
                experiment.zero_shot(pretrained, test),
            )
        probe = experiment.copy(pretrained, train)
        add(
            "CLSP + linear probe",
            split_name,
            experiment.train_labels(probe, "linear_probe", train, val, test),
        )
        add(
            "CLSP + fine-tuning",
            split_name,
            experiment.train_labels(pretrained, "finetune", train, val, test),
        )

    day_split = splits["random days"]
    train, val, test = dataset.set_task(task, split=day_split, num_workers=1)
    soft = experiment.pretrain(train, seed, soft_targets=True)
    add(
        "CLSP (soft targets) + fine-tuning",
        "random days",
        experiment.train_labels(soft, "finetune", train, val, test),
    )

    no_stats = PMDataStressDetection(
        include_text=True, text_statistics=False, min_hours=min_hours
    )
    train, val, test = dataset.set_task(no_stats, split=day_split, num_workers=1)
    model = experiment.pretrain(train, seed)
    add(
        "CLSP (text without statistics) + fine-tuning",
        "random days",
        experiment.train_labels(model, "finetune", train, val, test),
    )

    subset = LabeledSubsetSplit(day_split, fraction=0.1, seed=seed)
    train, labeled, val, test = dataset.set_task(task, split=subset, num_workers=1)
    if has_both_classes(labeled):
        model = experiment.new_model(train, seed)
        add(
            "10% labels: supervised",
            "random days",
            experiment.train_labels(model, "finetune", labeled, val, test),
        )
        model = experiment.pretrain(train, seed)
        add(
            "10% labels: CLSP + fine-tuning",
            "random days",
            experiment.train_labels(model, "finetune", labeled, val, test),
        )
    return rows


def main() -> None:
    """Runs all seeds and prints the results table."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", help="PMData folder; synthetic data if unset")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--pretrain-epochs", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--text-encoder", default="bert-base-uncased")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        root, min_hours = args.root, 24
        if root is None:
            print("SMOKE TEST on synthetic data; results are not meaningful.")
            root = str(Path(tmp) / "pmdata")
            write_synthetic_pmdata(Path(root), n_participants=6, n_days=20)
            min_hours = 0
        dataset = PMDataDataset(root=root, cache_dir=str(Path(tmp) / "cache"))
        experiment = Experiment(
            FrozenTextEncoder(args.text_encoder), args.pretrain_epochs, args.epochs
        )
        rows = []
        for seed in range(args.seeds):
            rows += run_seed(dataset, experiment, seed, min_hours)

    table = pd.DataFrame(rows).dropna(subset=["auc"])
    summary = table.groupby(["experiment", "split"], sort=False)["auc"].agg(
        ["mean", "std", "count"]
    )
    print("\nTest ROC-AUC over seeds")
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
