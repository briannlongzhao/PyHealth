pyhealth.datasets.PMDataDataset
===============================

Overview
--------

PMData is a sports logging dataset of 16 recreational athletes followed for
about five months (November 2019 to March 2020). It combines a Fitbit Versa 2
smartwatch (minute-level steps, calories and distance, heart rate every few
seconds, sleep, exercise), the PMSYS app (a daily wellness report with a 1-5
stress score, training sessions and injuries), daily Google Forms reports
(meals, weight, fluids, alcohol), a participant overview spreadsheet and, for
three participants, photos of their meals.

The data is open (CC BY-NC 4.0). Download the archive from the source below,
unzip it and pass the folder that holds ``p01`` ... ``p16`` and
``participant-overview.xlsx`` as ``root``. On first use the raw files are
converted into ``pmdata-<table>.csv`` files in ``root`` (heart rate becomes
hourly means, other per-minute signals hourly sums); food photos are indexed
by path and EXIF capture time, not loaded.

The dataset is used for daily stress detection by
:class:`~pyhealth.tasks.PMDataStressDetection`, following Yang et al.,
"Contrastive Pretraining for Stress Detection with Multimodal Wearable Sensor
Data and Surveys" (CHIL 2025).

Source: `PMData (Simula) <https://datasets.simula.no/pmdata/>`_; paper:
Thambawita et al., `PMData: a sports logging dataset
<https://doi.org/10.1145/3339825.3394926>`_ (MMSys 2020).

API Reference
-------------

.. autoclass:: pyhealth.datasets.PMDataDataset
    :members:
    :undoc-members:
    :show-inheritance:
