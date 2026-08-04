import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Iterable

from hicpg.schema import ManifestRecord


REQUIRED_COLUMNS = ("path", "label", "dataset", "patient_id")


def read_manifest(path: Path) -> list[ManifestRecord]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or any(name not in reader.fieldnames for name in REQUIRED_COLUMNS):
            raise ValueError("manifest columns are incomplete")
        records = [
            ManifestRecord(Path(row["path"]), row["label"].strip(), row["dataset"].strip(), row["patient_id"].strip())
            for row in reader
        ]
    validate_manifest(records)
    return records


def validate_manifest(records: Iterable[ManifestRecord], require_files: bool = True) -> None:
    materialized = list(records)
    if not materialized:
        raise ValueError("manifest is empty")
    identities: set[tuple[str, str]] = set()
    for record in materialized:
        if not record.label or not record.dataset or not record.patient_id:
            raise ValueError("manifest contains an empty field")
        if require_files and not record.path.is_file():
            raise FileNotFoundError(record.path)
        identity = (record.dataset, str(record.path))
        if identity in identities:
            raise ValueError(f"duplicate image entry: {record.path}")
        identities.add(identity)


def write_manifest(path: Path, records: Iterable[ManifestRecord]) -> None:
    rows = list(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        for record in rows:
            writer.writerow({"path": record.path, "label": record.label, "dataset": record.dataset, "patient_id": record.patient_id})
    temporary.replace(path)


def stratified_patient_split(
    records: Iterable[ManifestRecord],
    evaluation_fraction: float = 0.20,
    seed: int = 42,
) -> tuple[list[ManifestRecord], list[ManifestRecord]]:
    if not 0 < evaluation_fraction < 1:
        raise ValueError("evaluation fraction must be between zero and one")
    groups: dict[tuple[str, str], list[ManifestRecord]] = {}
    for record in records:
        groups.setdefault((record.dataset, record.label), []).append(record)
    rng = random.Random(seed)
    calibration: list[ManifestRecord] = []
    evaluation: list[ManifestRecord] = []
    for group in groups.values():
        patient_ids = sorted({record.patient_id for record in group})
        rng.shuffle(patient_ids)
        count = max(1, round(len(patient_ids) * evaluation_fraction))
        held_out = set(patient_ids[:count])
        calibration.extend(record for record in group if record.patient_id not in held_out)
        evaluation.extend(record for record in group if record.patient_id in held_out)
    return calibration, evaluation


def manifest_summary(records: Iterable[ManifestRecord]) -> dict[str, object]:
    rows = list(records)
    return {
        "images": len(rows),
        "datasets": dict(Counter(record.dataset for record in rows)),
        "labels": dict(Counter(record.label for record in rows)),
        "patients": len({(record.dataset, record.patient_id) for record in rows}),
    }


def file_sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def write_summary(path: Path, records: Iterable[ManifestRecord]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest_summary(records), indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)
