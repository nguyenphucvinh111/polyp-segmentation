import contextlib
import io
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from utils.checkpoint import Checkpointer
from utils.run_paths import check_run_destination, output_directories
from utils.trainer import Trainer


class RunPersistenceTests(unittest.TestCase):
    def test_explicit_output_root_and_overwrite_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            logs, snapshots, results = output_directories(Path(directory) / 'saved_runs')
            for path in (logs, snapshots, results):
                self.assertTrue(path.is_dir())
                self.assertTrue(path.is_relative_to(Path(directory).resolve()))
            check_run_destination(logs, snapshots, 'run')
            run_dir = snapshots / 'run'
            run_dir.mkdir()
            checkpoint = run_dir / 'best.pth'
            checkpoint.write_bytes(b'existing checkpoint')
            with self.assertRaises(FileExistsError):
                check_run_destination(logs, snapshots, 'run')
            check_run_destination(logs, snapshots, 'run', str(checkpoint))
            with self.assertRaises(FileNotFoundError):
                check_run_destination(logs, snapshots, 'run', str(run_dir / 'missing.pth'))

    def test_unmounted_colab_drive_is_rejected(self):
        with patch('utils.run_paths.Path.is_dir', return_value=False):
            with self.assertRaisesRegex(ValueError, 'Mount Google Drive'):
                output_directories('/content/drive/MyDrive/polyp_runs')

    def test_last_preserves_best_and_restores_training_rng(self):
        model = torch.nn.Conv2d(3, 1, 1)
        optimizer = torch.optim.AdamW(model.parameters())
        trainer = Trainer(model, optimizer, None, SimpleNamespace(amp=False), None, None)
        inputs = torch.ones(1, 3, 2, 2, device=trainer.device)
        model(inputs).square().mean().backward()
        optimizer.step()
        optimizer.zero_grad()
        expected_weight = model.weight.detach().clone()
        expected_moment = optimizer.state[model.weight]['exp_avg'].clone()
        with tempfile.TemporaryDirectory() as directory:
            manager = Checkpointer(directory, 'run')
            with contextlib.redirect_stdout(io.StringIO()):
                manager.save(model, optimizer, 1, {'mDice': .9}, trainer.training_state_dict())
                state = trainer.training_state_dict()
                expected = (random.random(), float(np.random.rand()), torch.rand(3))
                manager.save(model, optimizer, 2, {'mDice': .8}, state)
                last = Path(manager.save_dir) / 'last.pth'
                model(inputs).square().mean().backward()
                optimizer.step()
                epoch, metrics = Checkpointer.load(model, last, device='cpu',
                                                   optimizer=optimizer, trainer=trainer)
            self.assertEqual(epoch, 2)
            self.assertEqual(metrics['best_dice'], .9)
            self.assertEqual(metrics['best_epoch'], 1)
            self.assertEqual(expected[0], random.random())
            self.assertEqual(expected[1], float(np.random.rand()))
            torch.testing.assert_close(expected[2], torch.rand(3), rtol=0, atol=0)
            torch.testing.assert_close(expected_weight, model.weight, rtol=0, atol=0)
            torch.testing.assert_close(expected_moment, optimizer.state[model.weight]['exp_avg'], rtol=0, atol=0)
            best = torch.load(manager.get_best_path(), map_location='cpu', weights_only=True)
            self.assertEqual(best['epoch'], 1)
            resumed = Checkpointer(directory, 'run')
            resumed.restore_best(last, epoch, metrics)
            with contextlib.redirect_stdout(io.StringIO()):
                resumed.save(model, optimizer, 3, {'mDice': .85})
            best = torch.load(manager.get_best_path(), map_location='cpu', weights_only=True)
            self.assertEqual(best['epoch'], 1)

    def test_failed_write_preserves_existing_best(self):
        with tempfile.TemporaryDirectory() as directory:
            model = torch.nn.Conv2d(3, 1, 1)
            manager = Checkpointer(directory, 'run')
            with contextlib.redirect_stdout(io.StringIO()):
                manager.save(model, None, 1, {'mDice': .8})
            previous = Path(manager.get_best_path()).read_bytes()
            with patch('utils.checkpoint.torch.save', side_effect=OSError('write interrupted')):
                with self.assertRaises(OSError):
                    manager.save(model, None, 2, {'mDice': .9})
            self.assertEqual(Path(manager.get_best_path()).read_bytes(), previous)
            self.assertEqual(manager.best_dice, .8)

    def test_resume_keeps_newer_best_when_last_write_was_interrupted(self):
        with tempfile.TemporaryDirectory() as directory:
            model = torch.nn.Conv2d(3, 1, 1)
            manager = Checkpointer(directory, 'run')
            original_save = manager._save_file

            def interrupted_save(state, path):
                if Path(path).name == 'last.pth':
                    raise OSError('last write interrupted')
                original_save(state, path)

            with contextlib.redirect_stdout(io.StringIO()):
                manager.save(model, None, 1, {'mDice': .8})
                with patch.object(manager, '_save_file', side_effect=interrupted_save):
                    with self.assertRaises(OSError):
                        manager.save(model, None, 2, {'mDice': .9})
                last = Path(manager.save_dir) / 'last.pth'
                epoch, metrics = Checkpointer.load(model, last, device='cpu')
                resumed = Checkpointer(directory, 'run')
                resumed.restore_best(last, epoch, metrics)
                resumed.save(model, None, 3, {'mDice': .85})
            self.assertEqual(epoch, 1)
            self.assertEqual(resumed.best_dice, .9)
            best = torch.load(resumed.get_best_path(), weights_only=True)
            self.assertEqual(best['epoch'], 2)


if __name__ == '__main__':
    unittest.main()
