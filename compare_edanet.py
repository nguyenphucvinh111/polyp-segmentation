"""Compare an existing benchmark CSV to the user's EDANet reference table.

This reads finished results only. It never selects checkpoints, changes model
predictions or optimizes thresholds. Numeric wins are not statistical claims.
"""

import argparse
import csv
import json
import math
from pathlib import Path


DEFAULT_REFERENCE = Path(__file__).resolve().parent / 'benchmarks' / 'edanet_lidp2026.json'


def compare(metrics_path, reference_path=DEFAULT_REFERENCE):
    with open(reference_path, encoding='utf-8') as handle:
        reference = json.load(handle)
    with open(metrics_path, encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError('Metrics CSV is empty')
        dataset_column = reader.fieldnames[0]
        measurements = {}
        for row in reader:
            dataset = row[dataset_column]
            if dataset in measurements:
                raise ValueError(f'Duplicate dataset: {dataset}')
            measurements[dataset] = row

    comparisons = []
    for dataset, targets in reference['datasets'].items():
        if dataset not in measurements:
            raise ValueError(f'Missing test dataset: {dataset}')
        for metric, target in targets.items():
            try:
                value = float(measurements[dataset][metric])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f'Missing/invalid {metric} for {dataset}') from error
            if not math.isfinite(value):
                raise ValueError(f'Non-finite {metric} for {dataset}')
            direction = reference['directions'][metric]
            if direction not in ('lower', 'higher'):
                raise ValueError(f'Unknown metric direction: {direction}')
            gain = target - value if direction == 'lower' else value - target
            comparisons.append(dict(dataset=dataset, metric=metric, value=value,
                                    target=target, gain=gain, exceeds=gain > 1e-12))
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics', type=Path, required=True, help='Finished run metrics.csv')
    parser.add_argument('--reference', type=Path, default=DEFAULT_REFERENCE)
    args = parser.parse_args()
    try:
        comparisons = compare(args.metrics, args.reference)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print('Numerical comparison to supplied PDF (rounded values; protocol must match).')
    print(f"{'Dataset':<20} {'Metric':<7} {'Run':>9} {'EDANet':>9} {'Gain':>10}  Result")
    for row in comparisons:
        verdict = 'better' if row['exceeds'] else ('tie' if abs(row['gain']) <= 1e-12 else 'worse')
        print(f"{row['dataset']:<20} {row['metric']:<7} {row['value']:9.4f} "
              f"{row['target']:9.4f} {row['gain']:+10.4f}  {verdict}")
    passed = sum(row['exceeds'] for row in comparisons)
    print(f'Strictly better numerical values in {passed}/{len(comparisons)} dataset/metric pairs. '
          'Positive gain means improvement, including lower MAE.')
    print('Recall, Precision, Specificity, HD95 and wFb are absent from this PDF reference.')


if __name__ == '__main__':
    main()
