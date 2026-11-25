"""Tests for PMDataDataset and PMDataStressDetection.

Contributors:
    Brian Zhao (NetID: bnzhao2, bnzhao2@illinois.edu)
    Wanting Mao (NetID: wmao8, wmao8@illinois.edu)
    Aditya Radhakrishnan (NetID: aditya63, aditya63@illinois.edu)

Paper (reproduced):
    Zeyu Yang, Han Yu, Akane Sano. "Contrastive Pretraining for Stress
    Detection with Multimodal Wearable Sensor Data and Surveys." CHIL 2025.
    https://proceedings.mlr.press/v287/yang25a.html

All tests use tiny synthetic data: raw PMData-style files for two
participants written to a temporary folder, or ``Patient`` objects built in
memory. No real PMData file is read.
"""

import json
import math
import shutil
import tempfile
import unittest
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import polars as pl

from pyhealth.data import Patient
from pyhealth.datasets import PMDataDataset
from pyhealth.datasets.pmdata import PMDATA_TABLES
from pyhealth.tasks import PMDataStressDetection
from pyhealth.tasks.pmdata_stress_detection import FEATURE_GROUPS

DAY = "2019-11-01"


def _write_xlsx(path: Path, rows: List[List[str]]) -> None:
    """Writes a minimal ``.xlsx`` whose cells are all shared strings."""
    strings: List[str] = []
    cells = []
    for r, row in enumerate(rows, start=1):
        row_cells = []
        for c, value in enumerate(row):
            if value == "":
                continue
            strings.append(value)
            ref = f"{chr(ord('A') + c)}{r}"
            row_cells.append(f'<c r="{ref}" t="s"><v>{len(strings) - 1}</v></c>')
        cells.append(f'<row r="{r}">{"".join(row_cells)}</row>')
    ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    shared = "".join(f"<si><t>{s}</t></si>" for s in strings)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("xl/sharedStrings.xml", f'<sst xmlns="{ns}">{shared}</sst>')
        archive.writestr(
            "xl/worksheets/sheet1.xml",
            f'<worksheet xmlns="{ns}"><sheetData>{"".join(cells)}</sheetData>'
            "</worksheet>",
        )


def _write_json(path: Path, records: List[Dict[str, Any]]) -> None:
    """Writes a Fitbit-style JSON export."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records))


def _write_raw_pmdata(root: Path) -> None:
    """Writes raw PMData files for two synthetic participants.

    ``p01`` has every file, one value per hour for all 24 hours of
    2019-11-01 (two minute rows in hour 0) and wellness reports on two days
    (stress 4 and 2) plus an out-of-scale report (stress 0). ``p02`` has a
    single hour of wearable data, one report (stress 5) and no
    ``injury.csv``, ``reporting.csv`` or food images.
    """
    _write_xlsx(
        root / "participant-overview.xlsx",
        [
            ["", "", "", "", "Stride length from fitbit"],
            ["Participant ID", "Age", "Height", "Gender", "Max heart rate"],
            ["p01", "48", "195", "male", "182"],
            ["p02", "26", "163", "female ", ""],
        ],
    )

    p01 = root / "p01"
    minutes = [f"{DAY} {h:02d}:00:00" for h in range(24)] + [f"{DAY} 00:01:00"]
    for name, value in [("steps", "10"), ("calories", "1.5"), ("distance", "7")]:
        _write_json(
            p01 / "fitbit" / f"{name}.json",
            [{"dateTime": t, "value": value} for t in minutes],
        )
    _write_json(
        p01 / "fitbit" / "heart_rate.json",
        [
            {"dateTime": f"{DAY} 00:00:05", "value": {"bpm": 60, "confidence": 3}},
            {"dateTime": f"{DAY} 00:00:10", "value": {"bpm": 70, "confidence": 3}},
        ]
        + [
            {"dateTime": f"{DAY} {h:02d}:30:00", "value": {"bpm": 80, "confidence": 2}}
            for h in range(1, 24)
        ],
    )
    for name, value in [
        ("sedentary_minutes", "600"),
        ("lightly_active_minutes", "200"),
        ("moderately_active_minutes", "30"),
        ("very_active_minutes", "20"),
    ]:
        _write_json(
            p01 / "fitbit" / f"{name}.json",
            [{"dateTime": f"{DAY} 00:00:00", "value": value}],
        )
    _write_json(
        p01 / "fitbit" / "resting_heart_rate.json",
        [
            {
                "dateTime": f"{DAY} 00:00:00",
                "value": {"date": "11/01/19", "value": 55.5, "error": 6.0},
            }
        ],
    )
    _write_json(
        p01 / "fitbit" / "time_in_heart_rate_zones.json",
        [
            {
                "dateTime": f"{DAY} 00:00:00",
                "value": {
                    "valuesInZones": {
                        "BELOW_DEFAULT_ZONE_1": 1200.0,
                        "IN_DEFAULT_ZONE_1": 100.0,
                        "IN_DEFAULT_ZONE_2": 10.0,
                        "IN_DEFAULT_ZONE_3": 1.0,
                    }
                },
            }
        ],
    )
    _write_json(
        p01 / "fitbit" / "sleep.json",
        [
            {
                "dateOfSleep": DAY,
                "startTime": "2019-10-31 23:30:00",
                "endTime": f"{DAY}T07:00:00.000",
                "mainSleep": True,
                "minutesToFallAsleep": 0,
                "minutesAsleep": 400,
                "minutesAwake": 50,
                "timeInBed": 450,
                "efficiency": 95,
                "levels": {
                    "summary": {
                        "deep": {"minutes": 60},
                        "light": {"minutes": 250},
                        "rem": {"minutes": 90},
                        "wake": {"minutes": 50},
                    }
                },
            }
        ],
    )
    (p01 / "fitbit" / "sleep_score.csv").write_text(
        "timestamp,sleep_log_entry_id,overall_score,composition_score,"
        "revitalization_score,duration_score,deep_sleep_in_minutes,"
        "resting_heart_rate,restlessness\n"
        f"{DAY}T07:00:00Z,1,80,20,21,39,60,55,0.05\n"
    )
    _write_json(
        p01 / "fitbit" / "exercise.json",
        [
            {
                "startTime": f"{DAY} 17:00:00",
                "activityName": "Run",
                "logType": "auto_detected",
                "duration": 1800000,
                "activeDuration": 1800000,
                "calories": 300,
                "steps": 4000,
                "averageHeartRate": 150,
            }
        ],
    )
    (p01 / "pmsys").mkdir(parents=True)
    (p01 / "pmsys" / "wellness.csv").write_text(
        "effective_time_frame,fatigue,mood,readiness,sleep_duration_h,"
        "sleep_quality,soreness,soreness_area,stress\n"
        f"{DAY}T08:00:00.000Z,2,3,6,7,3,2,[],2\n"
        f"{DAY}T20:00:00.000Z,2,3,6,7,3,2,[],4\n"
        "2019-11-02T08:00:00.000Z,3,3,5,6,2,3,[],2\n"
        "2019-11-03T08:00:00.000Z,3,3,5,6,2,3,[],0\n"
    )
    (p01 / "pmsys" / "srpe.csv").write_text(
        "end_date_time,activity_names,perceived_exertion,duration_min\n"
        f"{DAY}T17:30:00.000Z,\"['running']\",7,30\n"
        f"{DAY}T19:00:00.000Z,\"['strength']\",5,20\n"
    )
    (p01 / "pmsys" / "injury.csv").write_text(
        "effective_time_frame,injuries\n"
        f"{DAY}T08:00:00.000Z,\"{{'left_knee': 'minor'}}\"\n"
        "2019-11-02T08:00:00.000Z,{}\n"
    )
    (p01 / "googledocs").mkdir(parents=True)
    (p01 / "googledocs" / "reporting.csv").write_text(
        "date,timestamp,meals,weight,glasses_of_fluid,alcohol_consumed\n"
        '01/11/2019,01/12/2019 21:00:00,"Breakfast, Dinner",80,6,Yes\n'
    )
    (p01 / "food-images").mkdir(parents=True)
    (p01 / "food-images" / "IMG_0001.jpg").write_bytes(b"not a real jpeg")
    (p01 / "food-images" / "notes.txt").write_text("not an image")

    p02 = root / "p02"
    for name in ["steps", "calories", "distance"]:
        _write_json(
            p02 / "fitbit" / f"{name}.json",
            [{"dateTime": f"{DAY} 10:00:00", "value": "5"}],
        )
    _write_json(
        p02 / "fitbit" / "heart_rate.json",
        [{"dateTime": f"{DAY} 10:00:05", "value": {"bpm": 90, "confidence": 3}}],
    )
    (p02 / "pmsys").mkdir(parents=True)
    (p02 / "pmsys" / "wellness.csv").write_text(
        "effective_time_frame,fatigue,mood,readiness,sleep_duration_h,"
        "sleep_quality,soreness,soreness_area,stress\n"
        f"{DAY}T09:00:00.000Z,4,2,3,5,2,4,[],5\n"
    )


class TestPMDataPreparation(unittest.TestCase):
    """Raw files are converted into the expected tidy tables."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(tempfile.mkdtemp())
        _write_raw_pmdata(cls.root)
        PMDataDataset.prepare_metadata(str(cls.root))
        cls.tables = {
            t: pd.read_csv(cls.root / f"pmdata-{t}.csv") for t in PMDATA_TABLES
        }

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.root)

    def test_every_table_is_written(self) -> None:
        for table in PMDATA_TABLES:
            self.assertTrue((self.root / f"pmdata-{table}.csv").exists(), table)
            self.assertIn("participant_id", self.tables[table].columns)

    def test_participants_from_spreadsheet(self) -> None:
        df = self.tables["participants"].set_index("participant_id")
        self.assertEqual(sorted(df.index), ["p01", "p02"])
        self.assertEqual(df.loc["p01", "age"], 48)
        self.assertEqual(df.loc["p02", "gender"], "female")
        self.assertEqual(df.loc["p01", "max_heart_rate"], 182)
        self.assertTrue(math.isnan(df.loc["p02", "max_heart_rate"]))

    def test_hourly_aggregation(self) -> None:
        df = self.tables["fitbit_hourly"]
        p01 = df[df.participant_id == "p01"].set_index("timestamp")
        self.assertEqual(len(p01), 24)
        first = p01.loc[f"{DAY} 00:00:00"]
        self.assertEqual(first["steps"], 20)  # two minute rows summed
        self.assertAlmostEqual(first["calories"], 3.0)
        self.assertEqual(first["heart_rate"], 65)  # mean of 60 and 70
        self.assertEqual(p01.loc[f"{DAY} 05:00:00", "heart_rate"], 80)

    def test_daily_and_sleep_tables(self) -> None:
        daily = self.tables["fitbit_daily"].iloc[0]
        self.assertEqual(daily["sedentary_minutes"], 600)
        self.assertEqual(daily["resting_heart_rate"], 55.5)
        self.assertEqual(daily["below_default_zone_1"], 1200)
        sleep = self.tables["sleep"].iloc[0]
        self.assertEqual(sleep["start_time"], "2019-10-31 23:30:00")
        self.assertEqual(sleep["date_of_sleep"], f"{DAY} 00:00:00")
        self.assertEqual(sleep["deep_minutes"], 60)
        score = self.tables["sleep_score"].iloc[0]
        self.assertEqual(score["timestamp"], f"{DAY} 07:00:00")  # "Z" dropped
        exercise = self.tables["exercise"].iloc[0]
        self.assertEqual(exercise["duration_min"], 30)

    def test_questionnaires(self) -> None:
        wellness = self.tables["wellness"]
        self.assertEqual(len(wellness), 5)
        self.assertEqual(wellness.iloc[0]["timestamp"], f"{DAY} 08:00:00")
        reporting = self.tables["reporting"].iloc[0]
        self.assertEqual(reporting["date"], f"{DAY} 00:00:00")  # DD/MM/YYYY
        self.assertEqual(len(self.tables["srpe"]), 2)

    def test_missing_files_are_skipped(self) -> None:
        for table in ["injury", "reporting", "fitbit_daily", "sleep"]:
            ids = set(self.tables[table].participant_id)
            self.assertEqual(ids, {"p01"}, table)

    def test_food_images_index_paths_only(self) -> None:
        images = self.tables["food_images"]
        self.assertEqual(len(images), 1)  # notes.txt is not an image
        self.assertTrue(images.iloc[0]["image_path"].endswith("IMG_0001.jpg"))
        self.assertTrue(pd.isna(images.iloc[0]["timestamp"]))  # no EXIF

    def test_missing_spreadsheet(self) -> None:
        root = Path(tempfile.mkdtemp())
        try:
            (root / "p07" / "pmsys").mkdir(parents=True)
            PMDataDataset.prepare_metadata(str(root), ["participants"])
            table = pd.read_csv(root / "pmdata-participants.csv")
            self.assertEqual(list(table.participant_id), ["p07"])
            self.assertTrue(table[["age", "gender"]].isna().all(axis=None))
        finally:
            shutil.rmtree(root)


class TestPMDataDataset(unittest.TestCase):
    """The prepared tables load through BaseDataset."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(tempfile.mkdtemp())
        cls.cache = Path(tempfile.mkdtemp())
        _write_raw_pmdata(cls.root)
        cls.dataset = PMDataDataset(root=str(cls.root), cache_dir=cls.cache)
        # Build the event cache once here (BaseDataset starts a Dask cluster
        # for it, which takes a few seconds), so each test runs quickly.
        cls.dataset.unique_patient_ids

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.root)
        shutil.rmtree(cls.cache)

    def test_patients(self) -> None:
        self.assertEqual(self.dataset.dataset_name, "pmdata")
        self.assertEqual(sorted(self.dataset.unique_patient_ids), ["p01", "p02"])

    def test_events(self) -> None:
        patient = self.dataset.get_patient("p01")
        self.assertEqual(len(patient.get_events(event_type="fitbit_hourly")), 24)
        wellness = patient.get_events(event_type="wellness")
        self.assertEqual([e.stress for e in wellness], ["2", "4", "2", "0"])
        self.assertEqual(wellness[0].timestamp, datetime(2019, 11, 1, 8))
        static = patient.get_events(event_type="participants")
        self.assertEqual(len(static), 1)
        self.assertIsNone(static[0].timestamp)

    def test_default_task(self) -> None:
        self.assertIsInstance(self.dataset.default_task, PMDataStressDetection)

    def test_set_task(self) -> None:
        # min_hours=0 keeps p01's Nov 2, which has no wearable data.
        samples = self.dataset.set_task(
            PMDataStressDetection(min_hours=0), num_workers=1
        )
        self.assertEqual(len(samples), 3)  # p01: Nov 1 and 2; p02: Nov 1
        labels = sorted(int(s["stress"].item()) for s in samples)
        self.assertEqual(labels, [0, 1, 1])
        sample = samples[0]
        self.assertEqual(tuple(sample["wearable"].shape), (24, 4))
        n_features = sum(len(v) for v in FEATURE_GROUPS.values())
        self.assertEqual(tuple(sample["survey"].shape), (n_features,))
        self.assertFalse(bool(sample["wearable"].isnan().any()))

    def test_unknown_table(self) -> None:
        with self.assertRaises(ValueError):
            PMDataDataset(root=str(self.root), tables=["heart"])

    def test_missing_root(self) -> None:
        with self.assertRaises(FileNotFoundError):
            PMDataDataset(root=str(self.root / "missing"))

    def test_root_without_participants(self) -> None:
        empty = tempfile.mkdtemp()
        try:
            with self.assertRaises(FileNotFoundError):
                PMDataDataset(root=empty)
        finally:
            shutil.rmtree(empty)


def _patient(
    events: List[Tuple[str, Optional[datetime], Dict[str, Any]]],
    patient_id: str = "p01",
) -> Patient:
    """Builds a Patient from ``(table, timestamp, attributes)`` tuples.

    Attribute values are stored as strings, as when loaded from CSV.
    """
    columns = sorted({f"{t}/{k}" for t, _, attrs in events for k in attrs})
    rows = []
    for table, timestamp, attrs in events:
        row: Dict[str, Any] = {"event_type": table, "timestamp": timestamp}
        for column in columns:
            prefix, key = column.split("/", 1)
            value = attrs.get(key) if prefix == table else None
            row[column] = None if value is None else str(value)
        rows.append(row)
    schema = {"event_type": pl.Utf8, "timestamp": pl.Datetime("ms")}
    schema.update({c: pl.Utf8 for c in columns})
    return Patient(patient_id=patient_id, data_source=pl.DataFrame(rows, schema=schema))


def _day(hours: int = 24, stress: int = 4) -> List[Tuple[str, Any, Dict[str, Any]]]:
    """Events of one synthetic day: hourly data, a report and context."""
    events: List[Tuple[str, Any, Dict[str, Any]]] = [
        ("participants", None, {"age": 30, "height": 180, "gender": "female"}),
        (
            "wellness",
            datetime(2019, 11, 1, 8),
            {
                "stress": stress,
                "fatigue": 2,
                "readiness": 7,
                "sleep_duration_h": 8,
                "sleep_quality": 4,
                "soreness": 1,
                "mood": 3,
            },
        ),
        ("fitbit_daily", datetime(2019, 11, 1), {"sedentary_minutes": 500}),
        (
            "sleep",
            datetime(2019, 10, 31, 23),
            {
                "main_sleep": "True",
                "date_of_sleep": "2019-11-01 00:00:00",
                "minutes_asleep": 420,
            },
        ),
        (
            "sleep",
            datetime(2019, 11, 1, 14),
            {
                "main_sleep": "False",
                "date_of_sleep": "2019-11-01 00:00:00",
                "minutes_asleep": 30,
            },
        ),
        ("sleep_score", datetime(2019, 11, 1, 7), {"overall_score": 80}),
        (
            "srpe",
            datetime(2019, 11, 1, 18),
            {"duration_min": 30, "perceived_exertion": 6},
        ),
        (
            "srpe",
            datetime(2019, 11, 1, 20),
            {"duration_min": 15, "perceived_exertion": 8},
        ),
        (
            "reporting",
            datetime(2019, 11, 1),
            {
                "meals": "Breakfast, Lunch",
                "weight": 70,
                "glasses_of_fluid": 8,
                "alcohol_consumed": "No",
            },
        ),
        ("injury", datetime(2019, 11, 1, 8), {"injuries": "{}"}),
    ]
    for hour in range(hours):
        events.append(
            (
                "fitbit_hourly",
                datetime(2019, 11, 1, hour),
                {"steps": hour, "heart_rate": 60 + hour, "calories": 2, "distance": ""},
            )
        )
    return events


class TestPMDataStressDetection(unittest.TestCase):
    """Task logic on in-memory patients."""

    def _features(self, task: PMDataStressDetection, sample: Dict[str, Any]) -> Dict:
        return dict(zip(task.feature_names, sample["survey"]))

    def test_schema(self) -> None:
        task = PMDataStressDetection()
        self.assertEqual(task.task_name, "PMDataStressDetection")
        self.assertEqual(set(task.input_schema), {"wearable", "survey"})
        self.assertEqual(task.output_schema, {"stress": "binary"})

    def test_sample_and_label(self) -> None:
        task = PMDataStressDetection()
        samples = task(_patient(_day(stress=4)))
        self.assertEqual(len(samples), 1)
        sample = samples[0]
        self.assertEqual(sample["patient_id"], "p01")
        self.assertEqual(sample["date"], "2019-11-01")
        self.assertEqual(sample["stress_score"], 4)
        self.assertEqual(sample["stress"], 1)
        self.assertEqual(task(_patient(_day(stress=3)))[0]["stress"], 0)

    def test_threshold(self) -> None:
        task = PMDataStressDetection(stress_threshold=2)
        self.assertEqual(task(_patient(_day(stress=3)))[0]["stress"], 1)

    def test_invalid_scores_are_skipped(self) -> None:
        task = PMDataStressDetection()
        self.assertEqual(task(_patient(_day(stress=0))), [])
        events = [e for e in _day() if e[0] != "wellness"]
        self.assertEqual(task(_patient(events)), [])

    def test_wearable_matrix(self) -> None:
        task = PMDataStressDetection(wearable_channels=["steps", "heart_rate"])
        wearable = task(_patient(_day()))[0]["wearable"]
        self.assertEqual(len(wearable), 24)
        self.assertEqual(wearable[5], [5.0, 65.0])

    def test_empty_values_are_nan(self) -> None:
        task = PMDataStressDetection(wearable_channels=["distance", "steps"])
        wearable = task(_patient(_day()))[0]["wearable"]
        self.assertTrue(math.isnan(wearable[0][0]))

    def test_min_hours(self) -> None:
        events = _day(hours=20)
        self.assertEqual(PMDataStressDetection()(_patient(events)), [])
        samples = PMDataStressDetection(min_hours=20)(_patient(events))
        self.assertEqual(len(samples), 1)
        self.assertTrue(all(math.isnan(v) for v in samples[0]["wearable"][23]))

    def test_daily_features(self) -> None:
        task = PMDataStressDetection()
        features = self._features(task, task(_patient(_day()))[0])
        self.assertEqual(len(features), len(task.feature_names))
        self.assertEqual(features["gender_female"], 1.0)
        self.assertEqual(features["sedentary_minutes"], 500)
        self.assertEqual(features["minutes_asleep"], 420)  # main sleep only
        self.assertEqual(features["overall_score"], 80)
        self.assertEqual(features["training_minutes"], 45)
        self.assertEqual(features["max_perceived_exertion"], 8)
        self.assertEqual(features["had_breakfast"], 1.0)
        self.assertEqual(features["had_dinner"], 0.0)
        self.assertEqual(features["alcohol_consumed"], 0.0)
        self.assertEqual(features["had_injury"], 0.0)
        self.assertEqual(features["readiness"], 7)
        self.assertNotIn("mood", features)
        self.assertTrue(math.isnan(features["resting_heart_rate"]))

    def test_day_without_training_has_zero_training(self) -> None:
        events = [e for e in _day() if e[0] in {"wellness", "fitbit_hourly"}]
        task = PMDataStressDetection()
        features = self._features(task, task(_patient(events))[0])
        self.assertEqual(features["training_minutes"], 0.0)
        self.assertEqual(features["max_perceived_exertion"], 0.0)
        self.assertTrue(math.isnan(features["age"]))  # no participants row

    def test_feature_groups(self) -> None:
        task = PMDataStressDetection(feature_groups=["demographics", "training"])
        self.assertEqual(
            task.feature_names,
            FEATURE_GROUPS["demographics"] + FEATURE_GROUPS["training"],
        )
        sample = task(_patient(_day()))[0]
        self.assertEqual(len(sample["survey"]), 5)

    def test_last_report_of_day_is_used(self) -> None:
        events = _day(stress=2)
        events.append(
            ("wellness", datetime(2019, 11, 1, 21), {"stress": 5, "fatigue": 1})
        )
        samples = PMDataStressDetection()(_patient(events))
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["stress_score"], 5)

    def test_invalid_arguments(self) -> None:
        with self.assertRaises(ValueError):
            PMDataStressDetection(feature_groups=["vitals"])
        with self.assertRaises(ValueError):
            PMDataStressDetection(feature_groups=[])
        with self.assertRaises(ValueError):
            PMDataStressDetection(wearable_channels=["temperature"])
        with self.assertRaises(ValueError):
            PMDataStressDetection(min_hours=25)


if __name__ == "__main__":
    unittest.main()
