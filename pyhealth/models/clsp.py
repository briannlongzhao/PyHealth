"""CLSP: contrastive language-sequence pretraining for wearable data.

The name follows the paper, which calls its method "contrastive
language-sequence pretraining", and the authors' ``run_clsp_*`` scripts.

Contributors:
    Brian Zhao (NetID: bnzhao2, bnzhao2@illinois.edu)
    Wanting Mao (NetID: wmao8, wmao8@illinois.edu)
    Aditya Radhakrishnan (NetID: aditya63, aditya63@illinois.edu)

Paper (reproduced):
    Zeyu Yang, Han Yu, Akane Sano. "Contrastive Pretraining for Stress
    Detection with Multimodal Wearable Sensor Data and Surveys." Conference
    on Health, Inference, and Learning (CHIL), PMLR 287, 2025.
    https://proceedings.mlr.press/v287/yang25a.html
    Authors' code: https://github.com/comp-well-org/clip-stress-detection

Description:
    A CLIP-style model for multimodal wearable data. A multimodal encoder
    (a Transformer over hourly wearable data and a residual MLP over tabular
    survey features, fused late) is pretrained without labels to match each
    day with a generated text description of the same day, embedded by a
    frozen pretrained language model. The encoder is then used for a
    downstream task with linear probing or end-to-end fine-tuning.
"""

import logging
from typing import Callable, Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from pyhealth.datasets import SampleDataset

from .base_model import BaseModel

logger = logging.getLogger(__name__)

#: Training stages of :class:`CLSP`.
STAGES = ("pretrain", "linear_probe", "finetune")


class FrozenTextEncoder:
    """A frozen Hugging Face text encoder with mean pooling and a cache.

    Texts are tokenized, encoded without gradients, and their token
    embeddings averaged over the attention mask, as in the paper (which
    averages BERT token embeddings). Embeddings are cached by text, so each
    distinct text is encoded once.

    The encoder is not an ``nn.Module`` attribute of the model that uses it,
    so its weights are not trained, not counted in ``model.parameters()``
    and not saved in checkpoints.

    Args:
        model_name: A Hugging Face model id or local path. Defaults to
            ``"bert-base-uncased"``, as in the paper.
        max_length: Longest tokenized text; longer texts are truncated.
        batch_size: Texts encoded per forward pass.

    Attributes:
        embedding_dim: Size of the returned embeddings.

    Examples:
        >>> encoder = FrozenTextEncoder("bert-base-uncased")  # doctest: +SKIP
        >>> encoder(["steps: mean is 10"], torch.device("cpu")).shape  # doctest: +SKIP
        torch.Size([1, 768])
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        max_length: int = 512,
        batch_size: int = 64,
    ) -> None:
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.embedding_dim: int = self.model.config.hidden_size
        self.max_length = max_length
        self.batch_size = batch_size
        self._cache: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def __call__(self, texts: Sequence[str], device: torch.device) -> torch.Tensor:
        """Embeds texts.

        Args:
            texts: The texts.
            device: Device to run the encoder on and return embeddings on.

        Returns:
            A ``(len(texts), embedding_dim)`` float tensor.
        """
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            self.model.to(device)
            for start in range(0, len(missing), self.batch_size):
                chunk = missing[start : start + self.batch_size]
                tokens = self.tokenizer(
                    chunk,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(device)
                hidden = self.model(**tokens).last_hidden_state
                mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                for text, emb in zip(chunk, pooled.float().cpu()):
                    self._cache[text] = emb
        return torch.stack([self._cache[t] for t in texts]).to(device)


class ProjectionHead(nn.Module):
    """Projection head of the authors' code: linear, GELU, linear, residual.

    Args:
        input_dim: Input size.
        projection_dim: Output size.
        dropout: Dropout before the residual connection.

    Examples:
        >>> ProjectionHead(8, 4, dropout=0.0)(torch.randn(2, 8)).shape
        torch.Size([2, 4])
    """

    def __init__(self, input_dim: int, projection_dim: int, dropout: float) -> None:
        super().__init__()
        self.projection = nn.Linear(input_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Projects ``x``.

        Args:
            x: A ``(batch, input_dim)`` tensor.

        Returns:
            A ``(batch, projection_dim)`` tensor.
        """
        projected = self.projection(x)
        x = self.dropout(self.fc(self.gelu(projected)))
        return self.layer_norm(x + projected)


class WearableEncoder(nn.Module):
    """Transformer encoder over a ``(time, channels)`` wearable series.

    Each time step's channels are projected to ``hidden_dim`` and given a
    learned position embedding; a Transformer encoder follows and its
    outputs are averaged over time.

    Args:
        num_steps: Length of the series (24 hourly steps for PMData).
        num_channels: Channels per step.
        hidden_dim: Model width.
        num_layers: Transformer layers.
        num_heads: Attention heads; must divide ``hidden_dim``.
        dropout: Dropout rate.

    Examples:
        >>> WearableEncoder(24, 4, 8, 1, 2, 0.0)(torch.randn(3, 24, 4)).shape
        torch.Size([3, 8])
    """

    def __init__(
        self,
        num_steps: int,
        num_channels: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input = nn.Linear(num_channels, hidden_dim)
        self.position = nn.Parameter(torch.zeros(1, num_steps, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=2 * hidden_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes a batch of series.

        Args:
            x: A ``(batch, num_steps, num_channels)`` tensor.

        Returns:
            A ``(batch, hidden_dim)`` tensor.
        """
        h = self.input(x) + self.position[:, : x.shape[1]]
        return self.encoder(h).mean(dim=1)


class TabularResNet(nn.Module):
    """Residual MLP for tabular features (a "ResNet" for tables).

    Args:
        num_features: Input features.
        hidden_dim: Width of the residual blocks and the output.
        num_blocks: Residual blocks.
        dropout: Dropout rate.

    Examples:
        >>> TabularResNet(5, 8, 2, 0.0)(torch.randn(3, 5)).shape
        torch.Size([3, 8])
    """

    def __init__(
        self, num_features: int, hidden_dim: int, num_blocks: int, dropout: float
    ) -> None:
        super().__init__()
        self.input = nn.Linear(num_features, hidden_dim)
        self.blocks = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(num_blocks)
        )
        self.output = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes a batch of feature vectors.

        Args:
            x: A ``(batch, num_features)`` tensor.

        Returns:
            A ``(batch, hidden_dim)`` tensor.
        """
        h = self.input(x)
        for block in self.blocks:
            h = h + block(h)
        return self.output(h)


class CLSP(BaseModel):
    """Contrastive language-sequence pretraining (CLIP for wearable data).

    Implements the method of Yang et al., "Contrastive Pretraining for
    Stress Detection with Multimodal Wearable Sensor Data and Surveys"
    (CHIL 2025):

    1. A **multimodal encoder** embeds each sample: a Transformer encoder
       over the wearable time series and a residual MLP over the tabular
       features, concatenated and fused by a linear layer (late fusion).
    2. **Contrastive pretraining** (stage ``"pretrain"``) needs no labels.
       Each sample's text description (e.g. from
       :class:`~pyhealth.tasks.PMDataStressDetection` with
       ``include_text=True``) is embedded by a frozen language model (BERT
       by default, averaged over tokens). Projection heads map both sides
       to a shared space, and a symmetric CLIP loss pulls each sample
       towards its own description and away from the others in the batch
       (cosine similarities divided by ``temperature``).
    3. **Downstream training**: a linear classifier on the (ReLU of the)
       projected sample embedding, trained with the dataset's label loss,
       either with the encoder frozen (``"linear_probe"``) or end to end
       (``"finetune"``).

    The stage is set with :meth:`set_stage`, so the same model goes through
    PyHealth's :class:`~pyhealth.trainer.Trainer` for every stage. In every
    stage :meth:`forward` returns the classifier's ``y_prob`` and ``y_true``,
    so the Trainer can evaluate it; only ``loss`` changes.

    Differences from the authors' code, all documented choices:

    - Embeddings are L2-normalised before the similarity, as the paper
      states (cosine similarity); the code used raw dot products.
    - ``soft_targets=False`` (default) uses the one-to-one targets of the
      paper's Eq. 1. ``soft_targets=True`` uses the code's soft targets, the
      softmax of the mean of the sample-sample and text-text similarities
      (divided by ``temperature``, where the code multiplied by it).
    - Token embeddings are averaged over real tokens only (the code also
      averaged padding).
    - The wearable Transformer attends over time steps (the code attended
      over channels), and the tabular encoder is a residual MLP (the code
      used a 1D convolutional ResNet over the feature vector).

    Args:
        dataset: The task's :class:`~pyhealth.datasets.SampleDataset`, with
            one binary, multiclass or multilabel label.
        wearable_key: Input key of the ``(time, channels)`` wearable tensor.
        survey_key: Input key of the tabular feature vector, or ``None`` to
            use the wearable data only.
        text_key: Input key of the text descriptions (a ``"raw"`` string
            field). Needed for pretraining and :meth:`zero_shot`.
        hidden_dim: Width of both encoders and the fused embedding.
        projection_dim: Size of the shared sample-text space.
        num_layers: Transformer layers and residual blocks.
        num_heads: Attention heads; must divide ``hidden_dim``.
        dropout: Dropout of the encoders and projection heads.
        temperature: Softmax temperature of the contrastive loss.
        soft_targets: Use the code's soft contrastive targets instead of the
            paper's one-to-one targets.
        text_encoder: A Hugging Face model id for :class:`FrozenTextEncoder`,
            or any callable ``(texts, device) -> (len(texts), dim) tensor``
            with an ``embedding_dim`` attribute (e.g. for tests).
        stage: Initial stage, one of ``"pretrain"``, ``"linear_probe"`` or
            ``"finetune"``.

    Raises:
        ValueError: If a key is missing from the dataset, the stage is
            unknown, or ``hidden_dim`` is not divisible by ``num_heads``.

    Examples:
        >>> from pyhealth.datasets import PMDataDataset, PatientSplit
        >>> from pyhealth.datasets import get_dataloader
        >>> from pyhealth.models import CLSP
        >>> from pyhealth.tasks import PMDataStressDetection
        >>> from pyhealth.trainer import Trainer
        >>> dataset = PMDataDataset(root="/path/to/pmdata")  # doctest: +SKIP
        >>> train, val, test = dataset.set_task(  # doctest: +SKIP
        ...     PMDataStressDetection(include_text=True),
        ...     split=PatientSplit((0.6, 0.2, 0.2)),
        ... )
        >>> model = CLSP(train, stage="pretrain")  # doctest: +SKIP
        >>> trainer = Trainer(model=model)  # doctest: +SKIP
        >>> trainer.train(get_dataloader(train, 64, shuffle=True), epochs=50)
        ... # doctest: +SKIP
        >>> model.set_stage("finetune")  # doctest: +SKIP
        >>> trainer.train(  # doctest: +SKIP
        ...     get_dataloader(train, 64, shuffle=True),
        ...     get_dataloader(val, 256),
        ...     epochs=20,
        ...     monitor="roc_auc",
        ... )
    """

    def __init__(
        self,
        dataset: SampleDataset,
        wearable_key: str = "wearable",
        survey_key: Optional[str] = "survey",
        text_key: Optional[str] = "text",
        hidden_dim: int = 64,
        projection_dim: int = 64,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        temperature: float = 0.07,
        soft_targets: bool = False,
        text_encoder: Union[str, Callable] = "bert-base-uncased",
        stage: str = "finetune",
    ) -> None:
        super().__init__(dataset)
        for key in [wearable_key, survey_key, text_key]:
            if key is not None and key not in self.feature_keys:
                raise ValueError(
                    f"Input key {key!r} is not in the dataset's input schema "
                    f"{self.feature_keys}"
                )
        if hidden_dim % num_heads:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads "
                f"({num_heads})"
            )
        if len(self.label_keys) != 1:
            raise ValueError("CLSP supports exactly one label key")

        self.wearable_key = wearable_key
        self.survey_key = survey_key
        self.text_key = text_key
        self.label_key = self.label_keys[0]
        self.temperature = temperature
        self.soft_targets = soft_targets

        sample = dataset[0]
        num_steps, num_channels = tuple(sample[wearable_key].shape)
        self.wearable_encoder = WearableEncoder(
            num_steps, num_channels, hidden_dim, num_layers, num_heads, dropout
        )
        fused_inputs = hidden_dim
        self.survey_encoder: Optional[TabularResNet] = None
        if survey_key is not None:
            num_features = int(sample[survey_key].numel())
            self.survey_encoder = TabularResNet(
                num_features, hidden_dim, num_layers, dropout
            )
            fused_inputs += hidden_dim
        self.fusion = nn.Linear(fused_inputs, hidden_dim)
        self.sample_projection = ProjectionHead(hidden_dim, projection_dim, dropout)
        self.classifier = nn.Linear(projection_dim, self.label_processor.output_size())

        # Kept out of the module tree: frozen, not saved in checkpoints.
        self._text_encoder_spec = text_encoder
        self._text_encoder: Optional[Callable] = None
        self.text_projection: Optional[ProjectionHead] = None
        if text_key is not None:
            text_dim = self._get_text_encoder().embedding_dim
            self.text_projection = ProjectionHead(text_dim, projection_dim, dropout)

        self.stage = "finetune"
        self.set_stage(stage)

    def _get_text_encoder(self) -> Callable:
        """Returns the text encoder, loading it on first use.

        Returns:
            The callable text encoder.
        """
        if self._text_encoder is None:
            spec = self._text_encoder_spec
            self._text_encoder = (
                FrozenTextEncoder(spec) if isinstance(spec, str) else spec
            )
        return self._text_encoder

    def _encoder_modules(self) -> List[nn.Module]:
        """Returns the modules frozen by linear probing.

        Returns:
            The encoders, fusion layer and sample projection head.
        """
        modules: List[nn.Module] = [
            self.wearable_encoder,
            self.fusion,
            self.sample_projection,
        ]
        if self.survey_encoder is not None:
            modules.append(self.survey_encoder)
        return modules

    def set_stage(self, stage: str) -> None:
        """Switches between pretraining, linear probing and fine-tuning.

        ``"linear_probe"`` freezes the encoders, fusion layer and sample
        projection head so that only the classifier trains; the other stages
        unfreeze them. Build a new optimizer (e.g. a new ``Trainer`` call)
        after switching.

        Args:
            stage: ``"pretrain"``, ``"linear_probe"`` or ``"finetune"``.

        Raises:
            ValueError: If ``stage`` is unknown, or is ``"pretrain"`` for a
                model without ``text_key``.
        """
        if stage not in STAGES:
            raise ValueError(f"Unknown stage {stage!r}; choose from {STAGES}")
        if stage == "pretrain" and self.text_key is None:
            raise ValueError("Pretraining needs text_key")
        trainable = stage != "linear_probe"
        for module in self._encoder_modules():
            for param in module.parameters():
                param.requires_grad = trainable
        self.stage = stage

    def _tensor(self, kwargs: Dict, key: str) -> torch.Tensor:
        """Reads an input tensor from a batch and moves it to the device.

        Args:
            kwargs: The batch.
            key: The input key.

        Returns:
            The float tensor on the model's device.
        """
        return torch.as_tensor(kwargs[key], dtype=torch.float32).to(self.device)

    def encode(self, **kwargs) -> torch.Tensor:
        """Embeds samples with the multimodal encoder and projection head.

        Args:
            **kwargs: A batch holding ``wearable_key`` (and ``survey_key``).

        Returns:
            The ``(batch, projection_dim)`` projected sample embeddings.
        """
        parts = [self.wearable_encoder(self._tensor(kwargs, self.wearable_key))]
        if self.survey_encoder is not None:
            survey = self._tensor(kwargs, self.survey_key)
            parts.append(self.survey_encoder(survey.flatten(start_dim=1)))
        fused = self.fusion(torch.cat(parts, dim=1))
        return self.sample_projection(fused)

    def encode_text(self, texts: Sequence[str]) -> torch.Tensor:
        """Embeds texts with the frozen encoder and the text projection head.

        Args:
            texts: The texts.

        Returns:
            The ``(len(texts), projection_dim)`` projected text embeddings.

        Raises:
            ValueError: If the model has no ``text_key``.
        """
        if self.text_projection is None:
            raise ValueError("The model was built without text_key")
        embeddings = self._get_text_encoder()(list(texts), self.device)
        return self.text_projection(embeddings.to(self.device))

    def contrastive_loss(
        self, sample_embeddings: torch.Tensor, text_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Symmetric CLIP loss between samples and their descriptions.

        Args:
            sample_embeddings: ``(batch, dim)`` projected sample embeddings.
            text_embeddings: ``(batch, dim)`` projected text embeddings of
                the same samples, in the same order.

        Returns:
            The scalar loss, averaged over both directions.
        """
        s = F.normalize(sample_embeddings, dim=-1)
        t = F.normalize(text_embeddings, dim=-1)
        logits = t @ s.T / self.temperature
        if self.soft_targets:
            similarity = (s @ s.T + t @ t.T) / 2
            targets = F.softmax(similarity / self.temperature, dim=-1)
            text_loss = -(targets * F.log_softmax(logits, dim=-1)).sum(1)
            sample_loss = -(targets.T * F.log_softmax(logits.T, dim=-1)).sum(1)
            return ((text_loss + sample_loss) / 2).mean()
        targets = torch.arange(len(s), device=logits.device)
        return (
            F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)
        ) / 2

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Runs the model in its current stage.

        Args:
            **kwargs: A batch from :func:`~pyhealth.datasets.get_dataloader`
                with the input keys and the label key. ``embed=True`` also
                returns the projected sample embeddings.

        Returns:
            A dict with ``loss`` (the contrastive loss when pretraining, the
            label loss otherwise), ``y_prob``, ``y_true`` and ``logit`` of
            the classifier, and ``embed`` if requested.
        """
        embeddings = self.encode(**kwargs)
        logits = self.classifier(F.relu(embeddings))
        y_true = kwargs[self.label_key].to(self.device)
        if self.stage == "pretrain":
            texts = kwargs[self.text_key]
            loss = self.contrastive_loss(embeddings, self.encode_text(texts))
        else:
            loss = self.label_processor.loss(logits, y_true)
        results = {
            "loss": loss,
            "y_prob": self.label_processor.activation(logits),
            "y_true": y_true,
            "logit": logits,
        }
        if kwargs.get("embed", False):
            results["embed"] = embeddings
        return results

    @torch.no_grad()
    def zero_shot(
        self,
        class_texts: Sequence[str] = ("label: relaxed", "label: stressed"),
        **kwargs,
    ) -> torch.Tensor:
        """Predicts without a classifier by matching samples to class texts.

        Each sample is compared with the text of each class in the shared
        space, as the authors' code did for validation; the softmax of the
        cosine similarities divided by ``temperature`` gives class
        probabilities.

        Args:
            class_texts: One text per class, in class-index order.
            **kwargs: A batch holding the input keys.

        Returns:
            A ``(batch, len(class_texts))`` tensor of class probabilities.
        """
        s = F.normalize(self.encode(**kwargs), dim=-1)
        t = F.normalize(self.encode_text(class_texts), dim=-1)
        return F.softmax(s @ t.T / self.temperature, dim=-1)
