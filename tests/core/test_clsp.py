"""Tests for the CLSP model.

Contributors:
    Brian Zhao (NetID: bnzhao2, bnzhao2@illinois.edu)
    Wanting Mao (NetID: wmao8, wmao8@illinois.edu)
    Aditya Radhakrishnan (NetID: aditya63, aditya63@illinois.edu)

Paper (reproduced):
    Zeyu Yang, Han Yu, Akane Sano. "Contrastive Pretraining for Stress
    Detection with Multimodal Wearable Sensor Data and Surveys." CHIL 2025.
    https://proceedings.mlr.press/v287/yang25a.html

The tests use four synthetic samples with a 6-step, 2-channel wearable
series, a 3-feature survey vector and short texts, and a fake text encoder
that returns fixed random vectors, so no language model is downloaded.
"""

import unittest
import zlib
from typing import Sequence

import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import CLSP


class FakeTextEncoder:
    """Maps each text to a fixed random vector (seeded by the text)."""

    embedding_dim = 8

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, texts: Sequence[str], device: torch.device) -> torch.Tensor:
        self.calls += 1
        vectors = []
        for text in texts:
            generator = torch.Generator().manual_seed(zlib.crc32(text.encode()))
            vectors.append(torch.randn(self.embedding_dim, generator=generator))
        return torch.stack(vectors).to(device)


def _samples(n: int = 4):
    generator = torch.Generator().manual_seed(0)
    return [
        {
            "patient_id": f"p{i % 2}",
            "wearable": torch.randn(6, 2, generator=generator).tolist(),
            "survey": torch.randn(3, generator=generator).tolist(),
            "text": f"steps: mean is {i}\nfatigue: {i % 3}",
            "stress": i % 2,
        }
        for i in range(n)
    ]


SCHEMA = {"wearable": "tensor", "survey": "tensor", "text": "raw"}


class TestCLSP(unittest.TestCase):
    """Instantiation, stages, losses, gradients and zero-shot prediction."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = create_sample_dataset(
            samples=_samples(),
            input_schema=SCHEMA,
            output_schema={"stress": "binary"},
            dataset_name="synthetic",
        )
        cls.batch = next(iter(get_dataloader(cls.dataset, batch_size=4)))

    def _model(self, **kwargs) -> CLSP:
        torch.manual_seed(0)
        options = dict(
            hidden_dim=8,
            projection_dim=4,
            num_layers=1,
            num_heads=2,
            dropout=0.0,
            text_encoder=FakeTextEncoder(),
        )
        options.update(kwargs)
        return CLSP(self.dataset, **options)

    def test_forward_shapes(self) -> None:
        model = self._model()
        output = model(**self.batch)
        self.assertEqual(model.stage, "finetune")
        self.assertEqual(tuple(output["y_prob"].shape), (4, 1))
        self.assertEqual(tuple(output["logit"].shape), (4, 1))
        self.assertEqual(tuple(output["y_true"].shape), (4, 1))
        self.assertEqual(output["loss"].dim(), 0)
        self.assertTrue(((output["y_prob"] >= 0) & (output["y_prob"] <= 1)).all())

    def test_embed(self) -> None:
        output = self._model()(**self.batch, embed=True)
        self.assertEqual(tuple(output["embed"].shape), (4, 4))

    def test_pretrain_loss_and_gradients(self) -> None:
        model = self._model(stage="pretrain")
        loss = model(**self.batch)["loss"]
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.wearable_encoder.input.weight.grad)
        self.assertIsNotNone(model.survey_encoder.input.weight.grad)
        self.assertIsNotNone(model.text_projection.projection.weight.grad)
        self.assertIsNone(model.classifier.weight.grad)  # not in the loss

    def test_pretrain_differs_from_label_loss(self) -> None:
        model = self._model(stage="pretrain")
        pretrain_loss = model(**self.batch)["loss"]
        model.set_stage("finetune")
        self.assertNotAlmostEqual(
            pretrain_loss.item(), model(**self.batch)["loss"].item(), places=4
        )

    def test_hard_contrastive_loss(self) -> None:
        model = self._model()
        aligned = torch.eye(4)
        low = model.contrastive_loss(aligned * 10, aligned * 10)
        high = model.contrastive_loss(aligned, aligned.flip(0))
        self.assertLess(low.item(), 1e-3)  # matched pairs: near-zero loss
        self.assertGreater(high.item(), 1.0)

    def test_soft_targets(self) -> None:
        model = self._model(soft_targets=True, stage="pretrain")
        loss = model(**self.batch)["loss"]
        self.assertTrue(torch.isfinite(loss))
        hard = self._model(stage="pretrain")(**self.batch)["loss"]
        self.assertNotAlmostEqual(loss.item(), hard.item(), places=4)

    def test_linear_probe_freezes_encoder(self) -> None:
        model = self._model()
        model.set_stage("linear_probe")
        model(**self.batch)["loss"].backward()
        self.assertIsNone(model.wearable_encoder.input.weight.grad)
        self.assertIsNone(model.sample_projection.projection.weight.grad)
        self.assertIsNotNone(model.classifier.weight.grad)
        model.set_stage("finetune")
        self.assertTrue(model.wearable_encoder.input.weight.requires_grad)

    def test_zero_shot(self) -> None:
        model = self._model().eval()
        probs = model.zero_shot(**self.batch)
        self.assertEqual(tuple(probs.shape), (4, 2))
        self.assertTrue(torch.allclose(probs.sum(dim=1), torch.ones(4)))

    def test_text_encoder_is_frozen_and_not_saved(self) -> None:
        encoder = FakeTextEncoder()
        model = self._model(text_encoder=encoder, stage="pretrain")
        model(**self.batch)
        self.assertEqual(encoder.calls, 1)
        names = list(model.state_dict())
        self.assertTrue(any(n.startswith("text_projection") for n in names))
        self.assertFalse(any("text_encoder" in n for n in names))

    def test_wearable_only(self) -> None:
        model = self._model(survey_key=None, text_key=None)
        self.assertIsNone(model.survey_encoder)
        self.assertEqual(tuple(model(**self.batch)["y_prob"].shape), (4, 1))
        with self.assertRaises(ValueError):
            model.set_stage("pretrain")

    def test_trainer_step(self) -> None:
        from pyhealth.trainer import Trainer

        model = self._model(stage="pretrain")
        before = model.wearable_encoder.input.weight.detach().clone()
        Trainer(model=model, device="cpu", enable_logging=False).train(
            get_dataloader(self.dataset, batch_size=4), epochs=1
        )
        after = model.wearable_encoder.input.weight.detach().cpu()
        self.assertFalse(torch.equal(before, after))

    def test_invalid_arguments(self) -> None:
        with self.assertRaises(ValueError):
            self._model(wearable_key="missing")
        with self.assertRaises(ValueError):
            self._model(num_heads=3)
        with self.assertRaises(ValueError):
            self._model(stage="train")


if __name__ == "__main__":
    unittest.main()
