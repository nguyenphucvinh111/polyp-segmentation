"""Small standard-library helpers shared by the runner and report command."""

import csv
import hashlib
import json
import math
from pathlib import Path

from experiments.pid_v2_ablation import METRICS, TEST_COUNTS


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_metrics(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not set(METRICS).issubset(reader.fieldnames):
            raise ValueError(f'Missing metric columns: {path}')
        dataset_column = reader.fieldnames[0]
        rows = {}
        for row in reader:
            name = row[dataset_column]
            if name in rows:
                raise ValueError(f'Duplicate dataset {name}: {path}')
            values = {metric: float(row[metric]) for metric in METRICS}
            if any(not math.isfinite(value) or value < 0 or
                   (metric != 'HD95' and value > 1)
                   for metric, value in values.items()):
                raise ValueError(f'Invalid metric values: {path}, {name}')
            rows[name] = values
    if set(rows) != set(TEST_COUNTS) | {'Mean'}:
        raise ValueError(f'Expected all five datasets plus Mean: {path}')
    for metric in METRICS:
        mean = sum(rows[name][metric] for name in TEST_COUNTS) / len(TEST_COUNTS)
        if not math.isclose(mean, rows['Mean'][metric], abs_tol=1.1e-4):
            raise ValueError(f'Mean is not the five-dataset macro average: {path}')
    return rows


def write_csv(path, rows, fields):
    with Path(path).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
