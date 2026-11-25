"""Daily stress detection task for the PMData dataset.

Contributors:
    Brian Zhao (NetID: bnzhao2, bnzhao2@illinois.edu)
    Wanting Mao (NetID: wmao8, wmao8@illinois.edu)
    Aditya Radhakrishnan (NetID: aditya63, aditya63@illinois.edu)

Paper (reproduced):
    Zeyu Yang, Han Yu, Akane Sano. "Contrastive Pretraining for Stress
    Detection with Multimodal Wearable Sensor Data and Surveys." Conference
    on Health, Inference, and Learning (CHIL), PMLR 287, 2025.
    https://proceedings.mlr.press/v287/yang25a.html

Description:
    Turns each PMData participant-day that has a wellness report into one
    sample: the day's 24 hourly Fitbit values (steps, heart rate, calories,
    distance), a vector of daily tabular features (demographics, activity,
    sleep, training, nutrition, injury and same-day wellness answers), and
    a binary label that is 1 when the self-reported 1-5 stress score is
    above a threshold (3 in the paper). Optionally, each sample also gets a
    text description of the day, generated as in the paper, for contrastive
    language-sequence pretraining (:class:`~pyhealth.models.CLSP`).
"""

import math
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from scipy import stats

from .base_task import BaseTask

#: Hourly Fitbit channels, in the order of the wearable tensor's last axis.
WEARABLE_CHANNELS: List[str] = ["steps", "heart_rate", "calories", "distance"]

#: Daily features per group, in the order they appear in ``survey``.
FEATURE_GROUPS: Dict[str, List[str]] = {
    "demographics": ["age", "height", "gender_female"],
    "activity": [
        "sedentary_minutes",
        "lightly_active_minutes",
        "moderately_active_minutes",
        "very_active_minutes",
        "resting_heart_rate",
        "below_default_zone_1",
        "in_default_zone_1",
        "in_default_zone_2",
        "in_default_zone_3",
    ],
    "sleep": [
        "minutes_asleep",
        "minutes_awake",
        "time_in_bed",
        "efficiency",
        "deep_minutes",
        "overall_score",
        "composition_score",
        "revitalization_score",
        "restlessness",
    ],
    "training": ["training_minutes", "max_perceived_exertion"],
    "nutrition": [
        "had_breakfast",
        "had_lunch",
        "had_dinner",
        "had_evening",
        "glasses_of_fluid",
        "weight",
        "alcohol_consumed",
    ],
    "injury": ["had_injury"],
    "wellness": [
        "fatigue",
        "readiness",
        "sleep_duration_h",
        "sleep_quality",
        "soreness",
    ],
}

_MEALS = ["breakfast", "lunch", "dinner", "evening"]

#: Features written as yes/no in the text description.
_YES_NO_FEATURES = {f"had_{m}" for m in _MEALS} | {"alcohol_consumed", "had_injury"}


def _to_float(value: Any) -> float:
    """Converts an event attribute to a float.

    Event attributes are loaded as strings; empty, missing and non-numeric
    values become NaN.

    Args:
        value: The raw attribute value.

    Returns:
        The value as a float, or NaN.
    """
    if value is None:
        return math.nan
    if isinstance(value, bool):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _format_number(value: float) -> str:
    """Formats a number for the text description.

    Args:
        value: The number.

    Returns:
        The number without decimals if it is whole, else with one decimal.
    """
    rounded = round(value, 1)
    if float(rounded).is_integer():
        return str(int(rounded))
    return f"{rounded:.1f}"


def channel_statistics(values: Sequence[float]) -> Dict[str, float]:
    """Summarises one day of an hourly channel as in the paper's prompts.

    The paper's code describes each channel with these nine statistics,
    rounded to one decimal. Missing hours are ignored.

    Args:
        values: The channel's hourly values; NaN for missing hours.

    Returns:
        The statistics by name, in prompt order. Empty if no hour has a
        value. Kurtosis and skewness are omitted if they are undefined (fewer
        than two values or a constant channel).

    Examples:
        >>> channel_statistics([1.0, 2.0, 3.0, float("nan")])["mean"]
        2.0
    """
    x = np.asarray([v for v in values if not math.isnan(v)], dtype=float)
    if x.size == 0:
        return {}
    result = {
        "mean": float(np.mean(x)),
        "standard deviation": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "median": float(np.median(x)),
        "root mean squared": float(np.sqrt(np.mean(x**2))),
    }
    if x.size > 1 and np.ptp(x) > 0:
        result["kurtosis"] = float(stats.kurtosis(x))
        result["skewness"] = float(stats.skew(x))
    result["interquartile range"] = float(stats.iqr(x))
    return {k: round(v, 1) for k, v in result.items()}


def describe_day(
    wearable: Sequence[Sequence[float]],
    channels: Sequence[str],
    features: Dict[str, float],
    feature_names: Sequence[str],
    include_statistics: bool = True,
) -> str:
    """Writes the text description of one day, as in the paper's prompts.

    One line per hourly channel with its statistics (see
    :func:`channel_statistics`), then one ``name: value`` line per known
    daily feature. Missing values are left out. The stress score, mood and
    label are never included.

    Args:
        wearable: The day's ``24 x len(channels)`` hourly values.
        channels: Names of the wearable channels.
        features: Daily feature values by name.
        feature_names: Features to describe, in order.
        include_statistics: Whether to write the channel statistics lines.

    Returns:
        The description, one line per item.

    Examples:
        >>> text = describe_day([[10.0]] * 24, ["steps"], {"age": 30.0}, ["age"])
        >>> text.splitlines()[0][:45]
        'steps: mean is 10, standard deviation is 0, m'
        >>> text.splitlines()[1]
        'age: 30'
    """
    lines = []
    if include_statistics:
        for i, channel in enumerate(channels):
            summary = channel_statistics([hour[i] for hour in wearable])
            if summary:
                parts = [f"{k} is {_format_number(v)}" for k, v in summary.items()]
                lines.append(f"{channel.replace('_', ' ')}: {', '.join(parts)}")
    for name in feature_names:
        value = features.get(name, math.nan)
        if value is None or math.isnan(value):
            continue
        if name == "gender_female":
            lines.append(f"gender: {'female' if value else 'male'}")
        elif name in _YES_NO_FEATURES:
            lines.append(f"{name.replace('_', ' ')}: {'yes' if value else 'no'}")
        else:
            lines.append(f"{name.replace('_', ' ')}: {_format_number(value)}")
    return "\n".join(lines)


def _is_true(value: Any) -> bool:
    """Returns whether a boolean-like attribute (e.g. ``"True"``) is true.

    Args:
        value: The raw attribute value.

    Returns:
        True for ``True``, ``"true"``, ``"True"`` or ``"1"``.
    """
    return str(value).strip().lower() in {"true", "1", "1.0"}


class PMDataStressDetection(BaseTask):
    """Binary daily stress detection on PMData, as in Yang et al. (CHIL 2025).

    One sample is made for each participant-day with a wellness report whose
    stress score is between 1 and 5 (a few reports hold 0, which is outside
    the scale, and are skipped). If a participant sent several reports on
    one day, the last one is used. Days are calendar days of the local clock
    times in the data.

    Inputs:

    - ``wearable``: a ``24 x len(wearable_channels)`` list of hourly values
      for that day (hour 0 first). Hours without data are NaN.
    - ``survey``: a vector of the features in ``feature_groups`` (see
      ``FEATURE_GROUPS`` for the names and order). Missing values are NaN.

    Both inputs use a tensor processor that z-scores each feature with
    statistics fitted on the training samples and then fills NaN with 0,
    the training mean. The paper's code also filled missing values with a
    constant, but used quantile normalization.

    - ``text`` (only with ``include_text=True``): a description of the day
      written as in the paper's prompts (see :func:`describe_day`), with
      statistics of each hourly channel and one line per known feature of
      ``feature_groups``. The stress score, mood and label are left out.
      It is kept as a string (``"raw"`` processor) for a model with a text
      encoder, such as :class:`~pyhealth.models.CLSP`.

    Output: ``stress`` is 1 if the stress score is above
    ``stress_threshold`` and 0 otherwise.

    Feature groups (all are used by default, as in the paper):

    - ``demographics``: age, height and gender from ``participants``.
    - ``activity``: that day's ``fitbit_daily`` values.
    - ``sleep``: the main sleep log that ends that morning
      (``date_of_sleep``) and that day's sleep score.
    - ``training``: total minutes and highest perceived exertion of that
      day's ``srpe`` sessions (0 if none).
    - ``nutrition``: that day's Google Forms report (meals eaten, glasses
      of fluid, weight, alcohol).
    - ``injury``: 1 if that day's injury report lists an injury, 0 if it
      lists none, NaN without a report.
    - ``wellness``: the other answers in the same wellness report as the
      stress score (fatigue, readiness, sleep duration and quality,
      soreness). ``mood`` is left out, as in the paper. Because these come
      from the same questionnaire as the label, leaving this group out
      shows how much of the performance rests on them.

    Args:
        stress_threshold: Scores above this are labeled stressed. Defaults
            to 3, as in the paper.
        feature_groups: Groups of ``FEATURE_GROUPS`` to put in ``survey``.
            Defaults to all groups.
        wearable_channels: Hourly channels of ``WEARABLE_CHANNELS`` to put
            in ``wearable``. Defaults to all four.
        min_hours: Least number of hours with any wearable value for a day
            to be kept. Defaults to 24 (complete days only, as in the
            paper's code); 0 keeps every day.
        include_text: Whether to add the ``text`` description. Defaults to
            False.
        text_statistics: Whether ``text`` includes the channel statistics
            lines (as in the paper). Defaults to True; False keeps only the
            daily feature lines.

    Raises:
        ValueError: If a feature group or channel is unknown, if no feature
            group is given, or if ``min_hours`` is outside 0-24.

    Examples:
        >>> from pyhealth.datasets import PMDataDataset
        >>> from pyhealth.tasks import PMDataStressDetection
        >>> dataset = PMDataDataset(root="/path/to/pmdata")  # doctest: +SKIP
        >>> task = PMDataStressDetection(
        ...     feature_groups=["demographics", "activity", "sleep"],
        ... )
        >>> samples = dataset.set_task(task)  # doctest: +SKIP
        >>> samples[0]["wearable"].shape  # doctest: +SKIP
        torch.Size([24, 4])
    """

    task_name: str = "PMDataStressDetection"
    input_schema: Dict[str, Any] = {
        "wearable": (
            "tensor",
            {"normalization": "zscore", "impute_strategy": "zero"},
        ),
        "survey": (
            "tensor",
            {"normalization": "zscore", "impute_strategy": "zero"},
        ),
    }
    output_schema: Dict[str, str] = {"stress": "binary"}

    def __init__(
        self,
        stress_threshold: int = 3,
        feature_groups: Optional[Sequence[str]] = None,
        wearable_channels: Optional[Sequence[str]] = None,
        min_hours: int = 24,
        include_text: bool = False,
        text_statistics: bool = True,
    ) -> None:
        super().__init__()
        groups = (
            list(FEATURE_GROUPS) if feature_groups is None else list(feature_groups)
        )
        unknown = [g for g in groups if g not in FEATURE_GROUPS]
        if unknown:
            raise ValueError(
                f"Unknown feature groups {unknown}; choose from {list(FEATURE_GROUPS)}"
            )
        if not groups:
            raise ValueError("feature_groups must name at least one group")
        channels = (
            list(WEARABLE_CHANNELS)
            if wearable_channels is None
            else list(wearable_channels)
        )
        unknown = [c for c in channels if c not in WEARABLE_CHANNELS]
        if unknown or not channels:
            raise ValueError(
                f"wearable_channels must be a non-empty subset of "
                f"{WEARABLE_CHANNELS}, got {channels}"
            )
        if not 0 <= min_hours <= 24:
            raise ValueError(f"min_hours must be between 0 and 24, got {min_hours}")

        self.stress_threshold = stress_threshold
        self.feature_groups = groups
        self.wearable_channels = channels
        self.min_hours = min_hours
        self.include_text = include_text
        self.text_statistics = text_statistics
        if include_text:
            self.input_schema = {**type(self).input_schema, "text": "raw"}

    @property
    def feature_names(self) -> List[str]:
        """Names of the ``survey`` features, in order.

        Returns:
            The feature names of the selected groups.
        """
        return [name for g in self.feature_groups for name in FEATURE_GROUPS[g]]

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Makes one sample per labeled day of a participant.

        Args:
            patient: A :class:`~pyhealth.data.Patient` from
                :class:`~pyhealth.datasets.PMDataDataset`.

        Returns:
            A list of samples, each with ``patient_id``, ``date``
            (``YYYY-MM-DD``), ``stress_score``, ``wearable``, ``survey``,
            ``stress`` and, with ``include_text``, ``text``. Empty if the
            participant has no usable day.
        """
        reports: Dict[date, Any] = {}
        for event in patient.get_events(event_type="wellness"):
            if event.timestamp is None:
                continue
            reports[event.timestamp.date()] = event  # last report of the day
        if not reports:
            return []

        hourly = self._hourly_by_day(patient)
        static = self._static_features(patient)
        daily = self._daily_features(patient)

        samples = []
        for day in sorted(reports):
            score = _to_float(reports[day]["stress"])
            if math.isnan(score) or not 1 <= score <= 5:
                continue
            hours = hourly.get(day, {})
            if len(hours) < self.min_hours:
                continue
            wearable = [
                [hours.get(h, {}).get(c, math.nan) for c in self.wearable_channels]
                for h in range(24)
            ]
            features = {
                "training_minutes": 0.0,
                "max_perceived_exertion": 0.0,
                **static,
                **daily.get(day, {}),
            }
            features.update(self._wellness_features(reports[day]))
            survey = [features.get(name, math.nan) for name in self.feature_names]
            sample = {
                "patient_id": patient.patient_id,
                "date": day.isoformat(),
                "stress_score": int(score),
                "wearable": wearable,
                "survey": survey,
                "stress": int(score > self.stress_threshold),
            }
            if self.include_text:
                sample["text"] = describe_day(
                    wearable,
                    self.wearable_channels,
                    features,
                    self.feature_names,
                    include_statistics=self.text_statistics,
                )
            samples.append(sample)
        return samples

    def _hourly_by_day(self, patient: Any) -> Dict[date, Dict[int, Dict[str, float]]]:
        """Groups hourly Fitbit values by day and hour.

        Hours whose selected channels are all missing are left out, so the
        number of entries of a day is its number of observed hours.

        Args:
            patient: The participant.

        Returns:
            ``{day: {hour: {channel: value}}}``.
        """
        by_day: Dict[date, Dict[int, Dict[str, float]]] = {}
        for event in patient.get_events(event_type="fitbit_hourly"):
            if event.timestamp is None:
                continue
            values = {
                c: _to_float(event.attr_dict.get(c)) for c in self.wearable_channels
            }
            if all(math.isnan(v) for v in values.values()):
                continue
            by_day.setdefault(event.timestamp.date(), {})[event.timestamp.hour] = values
        return by_day

    @staticmethod
    def _static_features(patient: Any) -> Dict[str, float]:
        """Reads the participant's demographics.

        Args:
            patient: The participant.

        Returns:
            ``age``, ``height`` and ``gender_female`` (NaN if unknown).
        """
        events = patient.get_events(event_type="participants")
        if not events:
            return {}
        attrs = events[0].attr_dict
        gender = str(attrs.get("gender") or "").strip().lower()
        return {
            "age": _to_float(attrs.get("age")),
            "height": _to_float(attrs.get("height")),
            "gender_female": {"female": 1.0, "male": 0.0}.get(gender, math.nan),
        }

    @staticmethod
    def _daily_features(patient: Any) -> Dict[date, Dict[str, float]]:
        """Collects the per-day features of every table.

        Args:
            patient: The participant.

        Returns:
            ``{day: {feature: value}}`` for the activity, sleep, training,
            nutrition and injury groups.
        """
        days: Dict[date, Dict[str, float]] = {}

        def put(day: date, values: Dict[str, float]) -> None:
            days.setdefault(day, {}).update(values)

        for event in patient.get_events(event_type="fitbit_daily"):
            if event.timestamp is not None:
                put(
                    event.timestamp.date(),
                    {
                        k: _to_float(event.attr_dict.get(k))
                        for k in FEATURE_GROUPS["activity"]
                    },
                )

        for event in patient.get_events(event_type="sleep"):
            attrs = event.attr_dict
            if not _is_true(attrs.get("main_sleep")):
                continue
            try:
                day = datetime.fromisoformat(str(attrs.get("date_of_sleep"))).date()
            except ValueError:
                continue
            put(day, {k: _to_float(attrs.get(k)) for k in FEATURE_GROUPS["sleep"][:5]})

        for event in patient.get_events(event_type="sleep_score"):
            if event.timestamp is not None:
                put(
                    event.timestamp.date(),
                    {
                        k: _to_float(event.attr_dict.get(k))
                        for k in FEATURE_GROUPS["sleep"][5:]
                    },
                )

        training: Dict[date, List[float]] = {}
        for event in patient.get_events(event_type="srpe"):
            if event.timestamp is None:
                continue
            minutes = _to_float(event.attr_dict.get("duration_min"))
            exertion = _to_float(event.attr_dict.get("perceived_exertion"))
            total, highest = training.get(event.timestamp.date(), [0.0, 0.0])
            training[event.timestamp.date()] = [
                total + (0.0 if math.isnan(minutes) else minutes),
                max(highest, 0.0 if math.isnan(exertion) else exertion),
            ]
        # Days without a session get 0 in __call__ (no training, not missing).
        for day, (total, highest) in training.items():
            put(day, {"training_minutes": total, "max_perceived_exertion": highest})

        for event in patient.get_events(event_type="reporting"):
            if event.timestamp is None:
                continue
            attrs = event.attr_dict
            meals = str(attrs.get("meals") or "").lower()
            alcohol = str(attrs.get("alcohol_consumed") or "").strip().lower()
            values = {f"had_{m}": float(m in meals) for m in _MEALS}
            values.update(
                {
                    "glasses_of_fluid": _to_float(attrs.get("glasses_of_fluid")),
                    "weight": _to_float(attrs.get("weight")),
                    "alcohol_consumed": {"yes": 1.0, "no": 0.0}.get(alcohol, math.nan),
                }
            )
            put(event.timestamp.date(), values)

        for event in patient.get_events(event_type="injury"):
            if event.timestamp is None:
                continue
            injuries = str(event.attr_dict.get("injuries") or "").strip()
            if injuries in {"", "nan", "None"}:
                continue
            had_injury = float(injuries not in {"{}", "[]"})
            current = days.get(event.timestamp.date(), {}).get("had_injury", 0.0)
            put(event.timestamp.date(), {"had_injury": max(current, had_injury)})

        return days

    @staticmethod
    def _wellness_features(report: Any) -> Dict[str, float]:
        """Reads the same-day wellness answers other than stress and mood.

        Args:
            report: The day's ``wellness`` event.

        Returns:
            The ``wellness`` group features.
        """
        return {
            k: _to_float(report.attr_dict.get(k)) for k in FEATURE_GROUPS["wellness"]
        }
