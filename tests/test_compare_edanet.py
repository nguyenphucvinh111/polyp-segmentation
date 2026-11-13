import csv
import json
import tempfile
import unittest
from pathlib import Path

from compare_edanet import DEFAULT_REFERENCE, compare


class ReferenceComparisonTests(unittest.TestCase):
    def write_metrics(self, path, improve=False, omit=None):
        reference = json.loads(DEFAULT_REFERENCE.read_text(encoding='utf-8'))
        with path.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=['dataset', *reference['directions']])
            writer.writeheader()
            for dataset, metrics in reference['datasets'].items():
                if dataset == omit:
                    continue
                values = dict(metrics)
                if improve:
                    for metric in values:
                        values[metric] += -.001 if metric == 'MAE' else .001
                writer.writerow(dict(dataset=dataset, **values))

    def test_all_metrics_and_correct_mae_direction(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'metrics.csv'
            self.write_metrics(path, improve=True)
            rows = compare(path)
            self.assertEqual(len(rows), 25)
            self.assertTrue(all(row['exceeds'] for row in rows))

    def test_ties_do_not_count_as_wins_and_missing_dataset_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'metrics.csv'
            self.write_metrics(path)
            self.assertFalse(any(row['exceeds'] for row in compare(path)))
            self.write_metrics(path, omit='CVC-300')
            with self.assertRaisesRegex(ValueError, 'CVC-300'):
                compare(path)


if __name__ == '__main__':
    unittest.main()
