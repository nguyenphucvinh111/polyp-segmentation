import contextlib
import csv
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from experiments.ablation_io import file_hash, read_json, read_metrics, write_json
from experiments.pid_v2_ablation import METRICS, TEST_COUNTS, VARIANTS, run_tag
from run_pid_v2_ablation import ensure_study, reconcile_log, run_state, train_command, validate_resume
from summarize_pid_v2_ablation import summarize


class AblationWorkflowTests(unittest.TestCase):
    def make_completed(self, root, variant, seed, score, study):
        tag = run_tag(variant, seed)
        directory = root / 'results' / tag
        directory.mkdir(parents=True)
        path = directory / 'metrics.csv'
        with path.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(['', *METRICS])
            for dataset in [*TEST_COUNTS, 'Mean']:
                writer.writerow([dataset, *([score] * len(METRICS))])
        (directory / 'efficiency.csv').write_text('Model,Hardware,Params (M)\ntest,CPU,0.835\n')
        write_json(root / 'ablation' / tag / 'completed.json', {
            'variant': variant, 'seed': seed, 'tag': tag, 'study_sha256': study['identity_sha256'],
            'best_epoch': 8, 'last_epoch': 100, 'best_val_mDice': .9,
            'artifacts': {path.relative_to(root).as_posix(): file_hash(path)}})

    def test_manifest_accepts_same_study_and_rejects_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identity = {'protocol': {'epochs': 100}, 'sources': {'train.py': 'sha'}}
            first = ensure_study(root, identity, {'git_revision': 'a'})
            self.assertEqual(first, ensure_study(root, identity, {'git_revision': 'b'}))
            with self.assertRaisesRegex(ValueError, 'changed'):
                ensure_study(root, {'protocol': {'epochs': 50}}, {})
            self.assertEqual(first, read_json(root / 'ablation_study.json'))

    def test_resume_and_completed_artifact_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            study = ensure_study(root, {'protocol': {}}, {})
            self.assertEqual(run_state(root, 'full', 42, study['identity_sha256']), ('new', None))
            checkpoint = root / 'snapshots' / run_tag('full', 42) / 'last.pth'
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b'placeholder')
            self.assertEqual(run_state(root, 'full', 42, study['identity_sha256']), ('resume', checkpoint))
            self.make_completed(root, 'full', 42, .8, study)
            self.assertEqual(run_state(root, 'full', 42, study['identity_sha256']), ('complete', None))
            metrics = root / 'results' / run_tag('full', 42) / 'metrics.csv'
            metrics.write_text('modified')
            with self.assertRaisesRegex(ValueError, 'missing/modified'):
                run_state(root, 'full', 42, study['identity_sha256'])

    def test_partial_output_without_checkpoint_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / 'logs' / run_tag('full', 42) / 'train_log.csv'
            log.parent.mkdir(parents=True)
            log.touch()
            with self.assertRaisesRegex(ValueError, 'no checkpoint'):
                run_state(root, 'full', 42, 'sha')

    def test_protocol_commands_and_resume(self):
        args = SimpleNamespace(epochs=100, batchsize=8, output_root='outputs', data_root='data')
        for variant in VARIANTS:
            command = train_command(args, variant, 3407, 'last.pth')
            for flag, expected in [('--lr', '0.0003'), ('--loss', 'structure_loss'),
                                   ('--boundary_weight', '0.2'), ('--seeds', '3407'),
                                   ('--scheduler', 'cosine'), ('--resume', 'last.pth')]:
                self.assertEqual(command[command.index(flag) + 1], expected)
            self.assertNotIn('--tta', command)
            self.assertNotIn('--fast', command)

    def test_mean_sample_std_paired_deltas_and_missing_not_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            study = ensure_study(root, {'protocol': {}}, {})
            for variant, scores in [('full', [.8, .9]), ('no_global', [.75, .85])]:
                for seed, score in zip([42, 3407], scores):
                    self.make_completed(root, variant, seed, score, study)
            report, completed, missing = summarize(root, [42, 3407])
            self.assertEqual(completed, 4)
            self.assertEqual(len(missing), 8)
            with (report / 'summary.csv').open(newline='') as handle:
                rows = {(r['variant'], r['dataset'], r['metric']): r for r in csv.DictReader(handle)}
            full = rows['full', 'CVC-300', 'mDice']
            self.assertAlmostEqual(float(full['mean']), .85)
            self.assertAlmostEqual(float(full['std']), .1 / 2**.5)
            variant = rows['no_global', 'CVC-300', 'mDice']
            self.assertAlmostEqual(float(variant['paired_delta_vs_full']), -.05)
            self.assertAlmostEqual(float(variant['paired_delta_std']), 0)
            self.assertEqual(rows['no_derivative', 'Mean', 'mDice']['mean'], '')
            # Request an absent third seed: partial means retained, paired claim withheld.
            summarize(root, [42, 3407, 1234])
            with (report / 'summary.csv').open(newline='') as handle:
                self.assertTrue(all(row['paired_delta_vs_full'] == '' for row in csv.DictReader(handle)))
            summarize(root, [42])
            with (report / 'summary.csv').open(newline='') as handle:
                self.assertTrue(all(row['std'] == '' for row in csv.DictReader(handle)))

    def test_incomplete_benchmark_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'metrics.csv'
            path.write_text(',mDice\nCVC-300,0.8\n')
            with self.assertRaisesRegex(ValueError, 'columns'):
                read_metrics(path)

    def test_resume_rejects_wrong_seed_even_with_matching_architecture(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / 'last.pth'
            torch.save({'model_name': run_tag('full', 1234),
                        'model_architecture': 'pid_unet_v2_ablation.v1.full'}, checkpoint)
            with self.assertRaisesRegex(ValueError, 'another run/variant'):
                validate_resume(root, 'full', 42, checkpoint, {})

    def test_interrupted_save_log_reconciliation_keeps_backup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tag = run_tag('full', 42)
            log = root / 'logs' / tag / 'train_log.csv'
            log.parent.mkdir(parents=True)
            original = 'epoch,loss\n1,4.0\n2,3.0\n2,2.9\n3,2.5\n'
            log.write_text(original)
            reconcile_log(root, tag, 2)
            with log.open(newline='') as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([r['epoch'] for r in rows], ['1', '2'])
            self.assertEqual(rows[-1]['loss'], '2.9')
            backups = list((root / 'ablation' / tag).glob('log_before_resume_*.csv'))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_text(), original)


if __name__ == '__main__':
    unittest.main()
