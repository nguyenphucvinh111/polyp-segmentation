import contextlib
import io
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from loss.structure_loss import structure_loss
from models import get_model
from models._pid_family import PID_V2_MODELS
from models.pid_unet_v2 import PIDUNetV2
from models.pid_unet_v2_os8 import PIDUNetV2OS8
from utils.checkpoint import Checkpointer
from utils.trainer import Trainer


class PIDUNetV2OS8Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(42)

    @staticmethod
    def make_trainer(model):
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        cfg = SimpleNamespace(model='pid_unet_v2_os8', amp=False, scheduler='cosine',
                              lr=3e-4, warmup_epochs=5, epochs=100, fast=True,
                              trainsize=64, clip=.5, boundary_weight=.2)
        return Trainer(model, optimizer, structure_loss, cfg, None, None)

    @staticmethod
    def batches():
        target = torch.zeros(2, 1, 64, 64)
        target[0, :, 16:48, 12:52] = 1
        target[1, :, 28:33, 28:33] = 1
        return [(torch.randn(2, 3, 64, 64), target),
                (torch.randn(1, 3, 64, 64), target[1:])]

    def test_v2_initialization_and_parameter_budget_preserved(self):
        base = PIDUNetV2()
        torch.manual_seed(42)
        model = get_model('pid_unet_v2_os8')
        self.assertIsInstance(model, PIDUNetV2OS8)
        self.assertIn('pid_unet_v2_os8', PID_V2_MODELS)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 835_373)
        self.assertEqual(set(base.state_dict()), set(model.state_dict()))
        for key, value in base.state_dict().items():
            torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
        self.assertEqual(base.down4.conv_path[0].stride, (2, 2))
        self.assertEqual(base.down4.pool_path[0].stride, 2)
        self.assertEqual([b[3].conv1.dilation[0] for b in base.bottleneck[0].branches],
                         [1, 3, 6, 12])

    def test_dense_semantic_features_and_multiscale_output_contract(self):
        model = PIDUNetV2OS8().eval()
        observed = []
        handle = model.bottleneck.register_forward_hook(
            lambda module, inputs, output: observed.append(tuple(output.shape)))
        try:
            with torch.inference_mode():
                for height, width in ((256, 256), (352, 352), (448, 448), (320, 384)):
                    outputs, boundary = model(torch.randn(1, 3, height, width), return_aux=True)
                    self.assertEqual(observed[-1], (1, 192, height // 8, width // 8))
                    self.assertEqual(len(outputs), 4)
                    for output in outputs:
                        self.assertEqual(output.shape, (1, 1, height, width))
                        self.assertTrue(torch.isfinite(output).all())
                    self.assertEqual(boundary.shape, (1, 1, height // 2, width // 2))
                    self.assertTrue(torch.isfinite(boundary).all())
        finally:
            handle.remove()

    def test_existing_loss_updates_encoder_decoder_and_boundary(self):
        model = PIDUNetV2OS8()
        trainer = self.make_trainer(model)
        watched = [model.inc[0].weight, model.down4.conv_path[0].weight,
                   model.bottleneck[0].branches[0][3].conv1.weight,
                   model.bottleneck[1].net[-1].weight, model.bag4.proj_i[0].weight,
                   model.head1.weight, model.boundary_branch.fuse[-1].weight]
        before = [p.detach().clone() for p in watched]
        with contextlib.redirect_stderr(io.StringIO()):
            stats = trainer.train_one_epoch(self.batches(), 1)
        self.assertTrue(all(math.isfinite(value) for value in stats.values()))
        # Positive contribution confirms the existing boundary objective is active.
        self.assertGreater(stats['loss'], sum(stats[f'loss{i}'] for i in (2, 3, 4, 5)))
        for previous, parameter in zip(before, watched):
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
            self.assertFalse(torch.equal(previous, parameter))
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_autocast_backward_with_empty_and_full_masks(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dtype = torch.float16 if device == 'cuda' else torch.bfloat16
        model = PIDUNetV2OS8().to(device).train()
        with torch.autocast(device, dtype=dtype):
            outputs = model(torch.randn(2, 3, 64, 64, device=device))
            target = torch.cat((torch.zeros_like(outputs[-1][:1]),
                                torch.ones_like(outputs[-1][:1]))).float()
            loss = sum(structure_loss(output.float(), target) for output in outputs)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertGreater(model.down4.conv_path[0].weight.grad.abs().sum().item(), 0)

    def test_checkpoint_resume_reproduces_next_optimizer_step(self):
        model = PIDUNetV2OS8()
        trainer = self.make_trainer(model)
        batches = self.batches()
        with tempfile.TemporaryDirectory() as directory, \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            trainer.train_one_epoch(batches, 1)
            manager = Checkpointer(directory, 'custom_run_name')
            manager.save(model, trainer.optimizer, 1, {'mDice': .5}, trainer.training_state_dict())
            last = Path(manager.save_dir) / 'last.pth'
            saved = torch.load(last, map_location='cpu', weights_only=True)
            self.assertEqual(saved['model_architecture'], model.checkpoint_architecture)
            trainer.train_one_epoch(batches, 2)
            expected = {key: value.detach().clone() for key, value in model.state_dict().items()}
            restored = PIDUNetV2OS8()
            resumed_trainer = self.make_trainer(restored)
            epoch, metrics = Checkpointer.load(restored, last, device='cpu',
                                               optimizer=resumed_trainer.optimizer, trainer=resumed_trainer)
            self.assertEqual(epoch, 1)
            self.assertEqual(metrics['best_epoch'], 1)
            resumed_manager = Checkpointer(directory, 'resumed_run')
            resumed_manager.restore_best(last, epoch, metrics, model=restored)
            self.assertEqual(Path(resumed_manager.get_best_path()).read_bytes(),
                             Path(manager.get_best_path()).read_bytes())
            resumed_trainer.train_one_epoch(batches, 2)
            for key, value in expected.items():
                torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)

    def test_same_shaped_v2_and_os8_checkpoints_cannot_be_interchanged(self):
        base, model = PIDUNetV2(), PIDUNetV2OS8()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pth'
            for checkpoint, target in (
                ({'state_dict': base.state_dict()}, model),
                (base.state_dict(), model),
                ({'state_dict': model.state_dict(), 'model_architecture': 'wrong'}, model),
                ({'state_dict': model.state_dict(), 'model_architecture': model.checkpoint_architecture}, base),
            ):
                torch.save(checkpoint, path)
                before = target.inc[0].weight.detach().clone()
                optimizer = torch.optim.AdamW(target.parameters(), lr=.001)
                with self.assertRaisesRegex(ValueError, 'same architecture'):
                    Checkpointer.load(target, path, device='cpu', optimizer=optimizer)
                torch.testing.assert_close(before, target.inc[0].weight, rtol=0, atol=0)
                self.assertEqual(optimizer.param_groups[0]['lr'], .001)
                self.assertFalse(optimizer.state)

    def test_legacy_v2_and_prefixed_os8_loads_remain_supported(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            path = Path(directory) / 'checkpoint.pth'
            base = PIDUNetV2()
            torch.save(base.state_dict(), path)
            Checkpointer.load(PIDUNetV2(), path, device='cpu')
            source, target = PIDUNetV2OS8(), PIDUNetV2OS8()
            state = {'module.' + key: value for key, value in source.state_dict().items()}
            torch.save({'state_dict': state, 'model_architecture': source.checkpoint_architecture}, path)
            Checkpointer.load(target, path, device='cpu')
            for key, value in source.state_dict().items():
                torch.testing.assert_close(value, target.state_dict()[key], rtol=0, atol=0)
            del state['module.head1.weight']
            torch.save({'state_dict': state, 'model_architecture': source.checkpoint_architecture}, path)
            with self.assertRaisesRegex(ValueError, 'same architecture'):
                Checkpointer.load(target, path, device='cpu')

    def test_resume_rejects_wrong_best_before_copying_or_updating_score(self):
        model = PIDUNetV2OS8()
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            source = Checkpointer(directory, 'source')
            source.save(model, None, 1, {'mDice': .5})
            last = Path(source.save_dir) / 'last.pth'
            wrong = {'state_dict': PIDUNetV2().state_dict(), 'epoch': 2, 'metrics': {'mDice': .9}}
            torch.save(wrong, source.get_best_path())
            destination = Checkpointer(directory, 'destination')
            with self.assertRaisesRegex(ValueError, 'same architecture'):
                destination.restore_best(last, 1, {'mDice': .5}, model=model)
            self.assertFalse(Path(destination.get_best_path()).exists())
            self.assertEqual(destination.best_dice, -1)
            torch.save(wrong, destination.get_best_path())
            original = Path(destination.get_best_path()).read_bytes()
            with self.assertRaisesRegex(ValueError, 'same architecture'):
                destination.restore_best(last, 1, {'mDice': .5}, model=model)
            self.assertEqual(Path(destination.get_best_path()).read_bytes(), original)
            self.assertEqual(destination.best_dice, -1)


if __name__ == '__main__':
    unittest.main()
