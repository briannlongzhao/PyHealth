pyhealth.tasks.PMDataStressDetection
====================================

Binary daily stress detection on :class:`~pyhealth.datasets.PMDataDataset`,
following the data preparation of Yang et al., `Contrastive Pretraining for
Stress Detection with Multimodal Wearable Sensor Data and Surveys
<https://proceedings.mlr.press/v287/yang25a.html>`_ (CHIL 2025). Each
participant-day with a wellness report becomes one sample with the day's 24
hourly Fitbit values, a vector of daily tabular features, and a label that is
1 when the self-reported 1-5 stress score is above 3.

.. autoclass:: pyhealth.tasks.PMDataStressDetection
    :members:
    :undoc-members:
    :show-inheritance:
