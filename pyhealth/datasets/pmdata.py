"""PMData sports-logging dataset for PyHealth.

Contributors:
    Brian Zhao (NetID: bnzhao2, bnzhao2@illinois.edu)
    Wanting Mao (NetID: wmao8, wmao8@illinois.edu)
    Aditya Radhakrishnan (NetID: aditya63, aditya63@illinois.edu)

Paper (reproduced):
    Zeyu Yang, Han Yu, Akane Sano. "Contrastive Pretraining for Stress
    Detection with Multimodal Wearable Sensor Data and Surveys." Conference
    on Health, Inference, and Learning (CHIL), PMLR 287, 2025.
    https://proceedings.mlr.press/v287/yang25a.html

Dataset:
    Vajira Thambawita et al. "PMData: a sports logging dataset." ACM
    Multimedia Systems Conference (MMSys), 2020.
    https://datasets.simula.no/pmdata/

Description:
    Loads the PMData lifelogging dataset (16 participants, about five months
    each) into PyHealth. Raw Fitbit JSON exports, PMSYS questionnaires,
    Google Forms reports, the participant overview spreadsheet and the food
    image folders are converted once into tidy ``pmdata-<table>.csv`` files,
    which :class:`~pyhealth.datasets.BaseDataset` then loads as event tables.
"""

import logging
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Callable, Dict, List, Optional

import orjson
import pandas as pd

from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)

#: Every table the dataset provides, in the order they are prepared.
PMDATA_TABLES: List[str] = [
    "participants",
    "fitbit_hourly",
    "fitbit_daily",
    "sleep",
    "sleep_score",
    "exercise",
    "wellness",
    "srpe",
    "injury",
    "reporting",
    "food_images",
]

#: Timestamp format used in every prepared CSV.
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
_HEART_RATE_PATTERN = re.compile(
    rb'"dateTime":\s*"([^"]+)",\s*"value":\s*\{\s*"bpm":\s*([0-9.]+)'
)
_XLSX_NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


class PMDataDataset(BaseDataset):
    """PMData: a sports logging dataset with Fitbit data and daily surveys.

    PMData records 16 recreational athletes for about five months with a
    Fitbit Versa 2 smartwatch, the PMSYS sports logging app (daily wellness,
    training load and injuries) and Google Forms (meals, weight, fluids and
    alcohol). Three participants also uploaded photos of their meals. The
    daily wellness questionnaire includes a 1-5 stress score, which
    :class:`~pyhealth.tasks.PMDataStressDetection` turns into a binary
    stress label.

    Source: https://datasets.simula.no/pmdata/ (CC BY-NC 4.0). Paper:
    Thambawita et al., "PMData: a sports logging dataset", MMSys 2020,
    https://doi.org/10.1145/3339825.3394926.

    Getting the data:
        - Access: open, no account needed.
        - Download: the archive linked from the source page (also mirrored
          on OSF), about 1.4 GB zipped, most of it food photos.
        - ``root``: the unzipped ``pmdata`` folder, which holds ``p01`` ...
          ``p16`` and ``participant-overview.xlsx``.

    On first use the raw files are converted into ``pmdata-<table>.csv``
    files in ``root``; later runs reuse them. The large heart-rate exports
    (over 100 MB per participant) are reduced to hourly values, which takes
    about half a minute once.

    Tables (``participant_id`` is the patient id in every table):

    - ``participants``: age, height, gender, max heart rate, stride lengths
      (no timestamp).
    - ``fitbit_hourly``: hourly steps, calories and distance (sums of the
      per-minute values) and mean heart rate (bpm).
    - ``fitbit_daily``: sedentary / lightly / moderately / very active
      minutes, resting heart rate and minutes in each heart-rate zone.
    - ``sleep``: one row per Fitbit sleep log (start time, minutes asleep,
      awake, in bed, efficiency, minutes in each sleep stage).
    - ``sleep_score``: Fitbit sleep scores, timestamped at wake-up.
    - ``exercise``: Fitbit exercise sessions.
    - ``wellness``: daily PMSYS wellness report (fatigue, mood, readiness,
      sleep duration and quality, soreness, stress).
    - ``srpe``: PMSYS training sessions (activity, perceived exertion,
      duration).
    - ``injury``: PMSYS injury reports.
    - ``reporting``: daily Google Forms report (meals, weight, glasses of
      fluid, alcohol).
    - ``food_images``: path of each meal photo, timestamped with its EXIF
      capture time when present. Pixels are not loaded.

    Times are kept as the local clock times written in the raw files; the
    trailing ``Z`` of PMSYS and sleep-score times is dropped, as in the
    paper's code, so that every table can be aligned by calendar day.

    Args:
        root: Folder holding the participant folders ``p01`` ... ``p16``.
        tables: Tables to load. Defaults to all tables in ``PMDATA_TABLES``.
        dataset_name: Name of the dataset. Defaults to ``"pmdata"``.
        config_path: Path to the YAML config. Defaults to the bundled
            ``configs/pmdata.yaml``.
        **kwargs: Passed on to :class:`~pyhealth.datasets.BaseDataset`
            (for example ``cache_dir`` or ``num_workers``).

    Raises:
        FileNotFoundError: If ``root`` does not exist, or if it holds neither
            prepared CSV files nor any participant folder.
        ValueError: If ``tables`` names a table that does not exist.

    Examples:
        >>> from pyhealth.datasets import PMDataDataset
        >>> dataset = PMDataDataset(root="/path/to/pmdata")
        >>> dataset.stats()
        >>> patient = dataset.get_patient("p01")
        >>> wellness = patient.get_events(event_type="wellness")
        >>> wellness[0].stress
        >>> samples = dataset.set_task()  # PMDataStressDetection
    """

    def __init__(
        self,
        root: str,
        tables: Optional[List[str]] = None,
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        root_path = Path(root)
        if not root_path.is_dir():
            raise FileNotFoundError(f"PMData root folder not found: {root}")
        if tables is None:
            tables = list(PMDATA_TABLES)
        unknown = sorted(set(tables) - set(PMDATA_TABLES))
        if unknown:
            raise ValueError(
                f"Unknown PMData tables {unknown}; choose from {PMDATA_TABLES}"
            )
        if config_path is None:
            config_path = str(Path(__file__).parent / "configs" / "pmdata.yaml")

        missing = [t for t in PMDATA_TABLES if not (root_path / _csv_name(t)).exists()]
        if missing:
            logger.info("Preparing PMData tables %s in %s", missing, root)
            self.prepare_metadata(str(root_path), missing)

        super().__init__(
            root=str(root_path),
            tables=tables,
            dataset_name=dataset_name or "pmdata",
            config_path=config_path,
            **kwargs,
        )

    @staticmethod
    def prepare_metadata(root: str, tables: Optional[List[str]] = None) -> None:
        """Converts the raw PMData files into ``pmdata-<table>.csv`` files.

        Called by ``__init__`` for every table whose CSV is missing. A
        participant missing a raw file (for example ``p08`` has no
        ``injury.csv``) is skipped for that table with a log message.

        Args:
            root: Folder holding the participant folders.
            tables: Tables to prepare. Defaults to all tables.

        Raises:
            FileNotFoundError: If ``root`` has no participant folder.
        """
        root_path = Path(root)
        participants = sorted(
            p.name
            for p in root_path.iterdir()
            if p.is_dir() and re.fullmatch(r"p\d+", p.name)
        )
        if not participants:
            raise FileNotFoundError(
                f"No participant folders (p01, p02, ...) found in {root}. "
                "Download PMData from https://datasets.simula.no/pmdata/ and "
                "pass the unzipped folder as root."
            )

        builders: Dict[str, Callable[[Path, List[str]], pd.DataFrame]] = {
            "participants": _build_participants,
            "fitbit_hourly": _build_fitbit_hourly,
            "fitbit_daily": _build_fitbit_daily,
            "sleep": _build_sleep,
            "sleep_score": _build_sleep_score,
            "exercise": _build_exercise,
            "wellness": _build_wellness,
            "srpe": _build_srpe,
            "injury": _build_injury,
            "reporting": _build_reporting,
            "food_images": _build_food_images,
        }
        for table in tables or PMDATA_TABLES:
            df = builders[table](root_path, participants)
            df.to_csv(root_path / _csv_name(table), index=False)
            logger.info("Wrote %d rows to %s", len(df), _csv_name(table))

    @property
    def default_task(self):
        """Returns the default task, binary daily stress detection.

        Returns:
            PMDataStressDetection: The stress detection task with default
            settings.
        """
        from pyhealth.tasks import PMDataStressDetection

        return PMDataStressDetection()


def _csv_name(table: str) -> str:
    """Returns the prepared CSV file name of a table.

    Args:
        table: Table name, e.g. ``"wellness"``.

    Returns:
        The file name, e.g. ``"pmdata-wellness.csv"``.
    """
    return f"pmdata-{table}.csv"


def _format_times(values: pd.Series, **kwargs) -> pd.Series:
    """Parses times and formats them as ``TIMESTAMP_FORMAT`` strings.

    A trailing ``Z`` is dropped so that times stay in the local clock time
    written in the file. Unparseable values become missing.

    Args:
        values: Raw time strings.
        **kwargs: Passed on to ``pd.to_datetime`` (e.g. ``format``).

    Returns:
        Formatted time strings, with missing values for unparseable times.
    """
    text = values.astype("string").str.replace(r"Z$", "", regex=True)
    times = pd.to_datetime(text, errors="coerce", **kwargs)
    return times.dt.strftime(TIMESTAMP_FORMAT)


def _concat(frames: List[pd.DataFrame], columns: List[str]) -> pd.DataFrame:
    """Concatenates per-participant frames into one table.

    Args:
        frames: Frames that each hold a ``participant_id`` column.
        columns: Columns of the output table, used when ``frames`` is empty.

    Returns:
        The concatenated table with exactly ``columns``.
    """
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True).reindex(columns=columns)


def _read_json(path: Path) -> list:
    """Reads a Fitbit JSON export.

    Args:
        path: Path to the JSON file.

    Returns:
        The decoded list of records.
    """
    return orjson.loads(path.read_bytes())


def _raw_file(root: Path, participant: str, relative: str) -> Optional[Path]:
    """Returns a participant's raw file, or ``None`` if it is missing.

    Args:
        root: Dataset root.
        participant: Participant id, e.g. ``"p01"``.
        relative: Path inside the participant folder, e.g.
            ``"pmsys/wellness.csv"``.

    Returns:
        The file path, or ``None`` (with a log message) if it does not exist.
    """
    path = root / participant / relative
    if not path.exists():
        logger.info("%s has no %s; skipping it", participant, relative)
        return None
    return path


def _read_xlsx_rows(path: Path) -> List[List[str]]:
    """Reads the first sheet of an ``.xlsx`` file with the standard library.

    Avoids a dependency on ``openpyxl`` for the one small spreadsheet in
    PMData. Cells are returned as strings, with ``""`` for empty cells.

    Args:
        path: Path to the ``.xlsx`` file.

    Returns:
        The rows of the first sheet, each a list of cell strings.
    """
    with zipfile.ZipFile(path) as archive:
        shared: List[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            tree = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in tree.findall("s:si", _XLSX_NS):
                shared.append(
                    "".join(t.text or "" for t in item.iter(f"{{{_XLSX_NS['s']}}}t"))
                )
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))

    rows: List[List[str]] = []
    for row in sheet.iter(f"{{{_XLSX_NS['s']}}}row"):
        cells: Dict[int, str] = {}
        for cell in row.findall("s:c", _XLSX_NS):
            letters = re.match(r"[A-Z]+", cell.get("r", "A")).group(0)
            column = 0
            for letter in letters:
                column = column * 26 + ord(letter) - ord("A") + 1
            value = cell.find("s:v", _XLSX_NS)
            text = value.text if value is not None and value.text else ""
            if cell.get("t") == "s" and text:
                text = shared[int(text)]
            cells[column - 1] = text.strip()
        if cells:
            rows.append([cells.get(i, "") for i in range(max(cells) + 1)])
    return rows


def _build_participants(root: Path, participants: List[str]) -> pd.DataFrame:
    """Builds the static ``participants`` table from the overview sheet.

    Args:
        root: Dataset root.
        participants: Participant ids found in ``root``.

    Returns:
        One row per participant. Participants without a spreadsheet row
        still get a row, with missing attributes.
    """
    columns = [
        "participant_id",
        "age",
        "height",
        "gender",
        "a_or_b_person",
        "max_heart_rate",
        "stride_walk",
        "stride_run",
    ]
    records: Dict[str, Dict[str, str]] = {}
    sheet_path = root / "participant-overview.xlsx"
    if sheet_path.exists():
        rows = _read_xlsx_rows(sheet_path)
        header_index = next(
            (i for i, r in enumerate(rows) if r and r[0] == "Participant ID"), None
        )
        if header_index is not None:
            header = [h.lower() for h in rows[header_index]]
            wanted = {
                "age": "age",
                "height": "height",
                "gender": "gender",
                "a or b person": "a_or_b_person",
                "max heart rate": "max_heart_rate",
                "stride walk": "stride_walk",
                "stride run": "stride_run",
            }
            positions = {wanted[h]: i for i, h in enumerate(header) if h in wanted}
            for row in rows[header_index + 1 :]:
                if not row or not re.fullmatch(r"p\d+", row[0]):
                    continue
                records[row[0]] = {
                    name: row[i] if i < len(row) else ""
                    for name, i in positions.items()
                }
    else:
        logger.warning("participant-overview.xlsx not found in %s", root)

    table = pd.DataFrame(
        [{"participant_id": p, **records.get(p, {})} for p in participants]
    ).reindex(columns=columns)
    # astype("string") also covers an all-missing column (no spreadsheet).
    table["gender"] = table["gender"].astype("string").str.strip().str.lower()
    return table


def _hourly_series(path: Path, name: str) -> pd.Series:
    """Reduces a per-minute Fitbit export to hourly sums.

    Args:
        path: Path to ``steps.json``, ``calories.json`` or ``distance.json``.
        name: Name of the returned series.

    Returns:
        Hourly sums indexed by the start of each hour.
    """
    records = _read_json(path)
    if not records:
        return pd.Series(name=name, dtype=float)
    frame = pd.DataFrame(records)
    times = pd.to_datetime(frame["dateTime"], errors="coerce")
    values = pd.to_numeric(frame["value"], errors="coerce")
    series = pd.Series(values.to_numpy(), index=times, name=name)
    return series[series.index.notna()].resample("h").sum(min_count=1)


def _hourly_heart_rate(path: Path) -> pd.Series:
    """Reduces the Fitbit heart-rate export to hourly mean bpm.

    The export holds a reading every few seconds (over 100 MB per
    participant), so it is scanned with a regular expression rather than
    decoded as a whole.

    Args:
        path: Path to ``heart_rate.json``.

    Returns:
        Hourly mean heart rate indexed by the start of each hour.
    """
    matches = _HEART_RATE_PATTERN.findall(path.read_bytes())
    if not matches:
        return pd.Series(name="heart_rate", dtype=float)
    times = pd.to_datetime([m[0].decode() for m in matches], errors="coerce")
    values = pd.to_numeric([m[1].decode() for m in matches], errors="coerce")
    series = pd.Series(values, index=times, name="heart_rate")
    return series[series.index.notna()].resample("h").mean()


def _build_fitbit_hourly(root: Path, participants: List[str]) -> pd.DataFrame:
    """Builds the ``fitbit_hourly`` table.

    Args:
        root: Dataset root.
        participants: Participant ids found in ``root``.

    Returns:
        One row per participant-hour with steps, heart rate, calories and
        distance.
    """
    columns = [
        "participant_id",
        "timestamp",
        "steps",
        "heart_rate",
        "calories",
        "distance",
    ]
    frames = []
    for participant in participants:
        series = []
        for name in ["steps", "heart_rate", "calories", "distance"]:
            path = _raw_file(root, participant, f"fitbit/{name}.json")
            if path is None:
                continue
            if name == "heart_rate":
                series.append(_hourly_heart_rate(path))
            else:
                series.append(_hourly_series(path, name))
        if not series:
            continue
        frame = pd.concat(series, axis=1).sort_index()
        frame.index.name = "timestamp"
        frame = frame.reset_index()
        frame["timestamp"] = frame["timestamp"].dt.strftime(TIMESTAMP_FORMAT)
        frame.insert(0, "participant_id", participant)
        frames.append(frame)
    return _concat(frames, columns)


def _build_fitbit_daily(root: Path, participants: List[str]) -> pd.DataFrame:
    """Builds the ``fitbit_daily`` table.

    Args:
        root: Dataset root.
        participants: Participant ids found in ``root``.

    Returns:
        One row per participant-day with activity minutes, resting heart
        rate and minutes in each heart-rate zone.
    """
    minute_files = [
        "sedentary_minutes",
        "lightly_active_minutes",
        "moderately_active_minutes",
        "very_active_minutes",
    ]
    zones = [
        "BELOW_DEFAULT_ZONE_1",
        "IN_DEFAULT_ZONE_1",
        "IN_DEFAULT_ZONE_2",
        "IN_DEFAULT_ZONE_3",
    ]
    columns = (
        ["participant_id", "date"]
        + minute_files
        + ["resting_heart_rate"]
        + [z.lower() for z in zones]
    )
    frames = []
    for participant in participants:
        series = []
        for name in minute_files:
            path = _raw_file(root, participant, f"fitbit/{name}.json")
            if path is None:
                continue
            records = _read_json(path)
            series.append(
                pd.Series({r["dateTime"]: r["value"] for r in records}, name=name)
            )
        path = _raw_file(root, participant, "fitbit/resting_heart_rate.json")
        if path is not None:
            values = {}
            for r in _read_json(path):
                value = r.get("value")
                rate = value.get("value") if isinstance(value, dict) else None
                values[r["dateTime"]] = rate if rate else None
            series.append(pd.Series(values, name="resting_heart_rate"))
        path = _raw_file(root, participant, "fitbit/time_in_heart_rate_zones.json")
        if path is not None:
            for zone in zones:
                values = {}
                for r in _read_json(path):
                    value = r.get("value")
                    in_zones = (
                        value.get("valuesInZones", {})
                        if isinstance(value, dict)
                        else {}
                    )
                    values[r["dateTime"]] = in_zones.get(zone)
                series.append(pd.Series(values, name=zone.lower()))
        if not series:
            continue
        frame = pd.concat(series, axis=1)
        frame.index.name = "date"
        frame = frame.reset_index()
        frame["date"] = _format_times(frame["date"], format="ISO8601")
        frame.insert(0, "participant_id", participant)
        frames.append(frame)
    return _concat(frames, columns)


def _build_sleep(root: Path, participants: List[str]) -> pd.DataFrame:
    """Builds the ``sleep`` table, one row per Fitbit sleep log.

    Args:
        root: Dataset root.
        participants: Participant ids found in ``root``.

    Returns:
        Sleep logs with their stage summaries.
    """
    columns = [
        "participant_id",
        "start_time",
        "end_time",
        "date_of_sleep",
        "main_sleep",
        "minutes_to_fall_asleep",
        "minutes_asleep",
        "minutes_awake",
        "time_in_bed",
        "efficiency",
        "deep_minutes",
        "light_minutes",
        "rem_minutes",
        "wake_minutes",
    ]
    rows = []
    for participant in participants:
        path = _raw_file(root, participant, "fitbit/sleep.json")
        if path is None:
            continue
        for r in _read_json(path):
            summary = (r.get("levels") or {}).get("summary") or {}
            row = {
                "participant_id": participant,
                "start_time": r.get("startTime"),
                "end_time": r.get("endTime"),
                "date_of_sleep": r.get("dateOfSleep"),
                "main_sleep": r.get("mainSleep"),
                "minutes_to_fall_asleep": r.get("minutesToFallAsleep"),
                "minutes_asleep": r.get("minutesAsleep"),
                "minutes_awake": r.get("minutesAwake"),
                "time_in_bed": r.get("timeInBed"),
                "efficiency": r.get("efficiency"),
            }
            for stage in ["deep", "light", "rem", "wake"]:
                row[f"{stage}_minutes"] = (summary.get(stage) or {}).get("minutes")
            rows.append(row)
    table = pd.DataFrame(rows).reindex(columns=columns)
    for column in ["start_time", "end_time", "date_of_sleep"]:
        table[column] = _format_times(table[column], format="ISO8601")
    return table


def _build_sleep_score(root: Path, participants: List[str]) -> pd.DataFrame:
    """Builds the ``sleep_score`` table.

    Args:
        root: Dataset root.
        participants: Participant ids found in ``root``.

    Returns:
        Fitbit sleep scores, timestamped at wake-up.
    """
    columns = [
        "participant_id",
        "timestamp",
        "overall_score",
        "composition_score",
        "revitalization_score",
        "duration_score",
        "deep_sleep_in_minutes",
        "resting_heart_rate",
        "restlessness",
    ]
    frames = []
    for participant in participants:
        path = _raw_file(root, participant, "fitbit/sleep_score.csv")
        if path is None:
            continue
        frame = pd.read_csv(path)
        frame["timestamp"] = _format_times(frame["timestamp"], format="ISO8601")
        frame.insert(0, "participant_id", participant)
        frames.append(frame)
    return _concat(frames, columns)


def _build_exercise(root: Path, participants: List[str]) -> pd.DataFrame:
    """Builds the ``exercise`` table from Fitbit exercise logs.

    Args:
        root: Dataset root.
        participants: Participant ids found in ``root``.

    Returns:
        One row per exercise session.
    """
    columns = [
        "participant_id",
        "start_time",
        "activity_name",
        "log_type",
        "duration_min",
        "active_duration_min",
        "calories",
        "steps",
        "distance",
        "average_heart_rate",
    ]
    rows = []
    for participant in participants:
        path = _raw_file(root, participant, "fitbit/exercise.json")
        if path is None:
            continue
        for r in _read_json(path):
            duration = r.get("duration")
            active = r.get("activeDuration")
            rows.append(
                {
                    "participant_id": participant,
                    "start_time": r.get("startTime"),
                    "activity_name": r.get("activityName"),
                    "log_type": r.get("logType"),
                    "duration_min": duration / 60000 if duration else None,
                    "active_duration_min": active / 60000 if active else None,
                    "calories": r.get("calories"),
                    "steps": r.get("steps"),
                    "distance": r.get("distance"),
                    "average_heart_rate": r.get("averageHeartRate"),
                }
            )
    table = pd.DataFrame(rows).reindex(columns=columns)
    table["start_time"] = _format_times(table["start_time"], format="ISO8601")
    return table


def _build_pmsys(
    root: Path,
    participants: List[str],
    file_name: str,
    time_column: str,
    columns: List[str],
) -> pd.DataFrame:
    """Builds a table from one of the PMSYS questionnaire CSVs.

    Args:
        root: Dataset root.
        participants: Participant ids found in ``root``.
        file_name: CSV name inside ``pmsys/``, e.g. ``"wellness.csv"``.
        time_column: Column holding the report time.
        columns: Output columns; the report time is written as
            ``timestamp``.

    Returns:
        The concatenated table.
    """
    frames = []
    for participant in participants:
        path = _raw_file(root, participant, f"pmsys/{file_name}")
        if path is None:
            continue
        frame = pd.read_csv(path)
        frame["timestamp"] = _format_times(frame[time_column], format="ISO8601")
        frame.insert(0, "participant_id", participant)
        frames.append(frame)
    return _concat(frames, columns)


def _build_wellness(root: Path, participants: List[str]) -> pd.DataFrame:
    """Builds the ``wellness`` table (daily PMSYS wellness report).

    Args:
        root: Dataset root.
        participants: Participant ids found in ``root``.

    Returns:
        One row per report, including the 1-5 ``stress`` score.
    """
    columns = [
        "participant_id",
        "timestamp",
        "fatigue",
        "mood",
        "readiness",
        "sleep_duration_h",
        "sleep_quality",
        "soreness",
        "soreness_area",
        "stress",
    ]
    return _build_pmsys(
        root, participants, "wellness.csv", "effective_time_frame", columns
    )


def _build_srpe(root: Path, participants: List[str]) -> pd.DataFrame:
    """Builds the ``srpe`` table (PMSYS training sessions).

    Args:
        root: Dataset root.
        participants: Participant ids found in ``root``.

    Returns:
        One row per session, timestamped at its end.
    """
    columns = [
        "participant_id",
        "timestamp",
        "activity_names",
        "perceived_exertion",
        "duration_min",
    ]
    return _build_pmsys(root, participants, "srpe.csv", "end_date_time", columns)


def _build_injury(root: Path, participants: List[str]) -> pd.DataFrame:
    """Builds the ``injury`` table (PMSYS injury reports).

    Args:
        root: Dataset root.
        participants: Participant ids found in ``root``.

    Returns:
        One row per report; ``injuries`` is ``"{}"`` when no injury was
        reported.
    """
    columns = ["participant_id", "timestamp", "injuries"]
    return _build_pmsys(
        root, participants, "injury.csv", "effective_time_frame", columns
    )


def _build_reporting(root: Path, participants: List[str]) -> pd.DataFrame:
    """Builds the ``reporting`` table (daily Google Forms report).

    Args:
        root: Dataset root.
        participants: Participant ids found in ``root``.

    Returns:
        One row per report with meals, weight, fluids and alcohol, dated by
        the day the report is about (``DD/MM/YYYY`` in the raw file).
    """
    columns = [
        "participant_id",
        "date",
        "meals",
        "weight",
        "glasses_of_fluid",
        "alcohol_consumed",
    ]
    frames = []
    for participant in participants:
        path = _raw_file(root, participant, "googledocs/reporting.csv")
        if path is None:
            continue
        frame = pd.read_csv(path)
        frame["date"] = _format_times(frame["date"], format="%d/%m/%Y")
        frame.insert(0, "participant_id", participant)
        frames.append(frame)
    return _concat(frames, columns)


def _image_capture_time(path: Path) -> Optional[str]:
    """Reads a photo's EXIF capture time.

    Args:
        path: Path to the image.

    Returns:
        The capture time as a ``TIMESTAMP_FORMAT`` string, or ``None`` if
        the image has no readable EXIF time.
    """
    try:
        from PIL import Image

        with Image.open(path) as image:
            exif = image.getexif()
            raw = exif.get_ifd(0x8769).get(0x9003) or exif.get(0x0132)
        if not raw:
            return None
        return pd.to_datetime(raw, format="%Y:%m:%d %H:%M:%S").strftime(
            TIMESTAMP_FORMAT
        )
    except Exception:  # unreadable image or malformed EXIF
        return None


def _build_food_images(root: Path, participants: List[str]) -> pd.DataFrame:
    """Builds the ``food_images`` table of meal photo paths.

    Args:
        root: Dataset root.
        participants: Participant ids found in ``root``.

    Returns:
        One row per photo with its absolute path and EXIF capture time.
    """
    columns = ["participant_id", "timestamp", "image_path"]
    rows = []
    for participant in participants:
        folder = root / participant / "food-images"
        if not folder.is_dir():
            continue
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() not in _IMAGE_SUFFIXES:
                continue
            rows.append(
                {
                    "participant_id": participant,
                    "timestamp": _image_capture_time(path),
                    "image_path": str(path.resolve()),
                }
            )
    return pd.DataFrame(rows).reindex(columns=columns)
