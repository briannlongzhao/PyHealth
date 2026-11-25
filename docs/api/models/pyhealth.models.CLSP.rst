pyhealth.models.CLSP
====================

Overview
--------

CLSP (contrastive language-sequence pretraining) is the method of Yang et al.,
`Contrastive Pretraining for Stress Detection with Multimodal Wearable Sensor
Data and Surveys <https://proceedings.mlr.press/v287/yang25a.html>`_ (CHIL
2025): a CLIP-style model for wearable time series and tabular survey
features.

A multimodal encoder (a Transformer over the wearable series and a residual
MLP over the tabular features, fused late) is pretrained without labels to
match each sample with a generated text description of the same sample,
embedded by a frozen language model (BERT by default). The pretrained encoder
is then trained for the downstream label with a linear probe or end to end.
The stage is switched with :meth:`~pyhealth.models.CLSP.set_stage`, so every
stage runs through :class:`~pyhealth.trainer.Trainer`.

The descriptions come from the task, e.g.
:class:`~pyhealth.tasks.PMDataStressDetection` with ``include_text=True`` on
:class:`~pyhealth.datasets.PMDataDataset`. See
``examples/pmdata_stress_detection_clsp.py`` for the full pipeline.

API Reference
-------------

.. autoclass:: pyhealth.models.CLSP
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.clsp.FrozenTextEncoder
    :members:
    :show-inheritance:
