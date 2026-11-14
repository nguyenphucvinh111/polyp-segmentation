import contextlib
import io
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from loss.structure_loss import structure_loss
from models import get_model
from models._pid_family import PID_V2_MODELS
from models.pid_unet_v2 import PIDUNetV2
from models.pid_unet_v8 import PIDUNetV8, IntegralControlledPIDBlock
from models.pid_unet_v9 import PIDUNetV9, InstanceBatchNorm2d
from utils.checkpoint import Checkpointer
from utils.trainer import Trainer


VARIANTS = (('pid_unet_v8', PIDUNetV8), ('pid_unet_v9', PIDUNetV9))


class PIDUNetV8V9Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(42)

    def test_v8_preserves_initial_v2_weights_and_predictions(self):
        base = PIDUNetV2().eval()
        torch.manual_seed(42)
        model = PIDUNetV8().eval()
        for key, value in base.state_dict().items():
            torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
        extras = set(model.state_dict()) - set(base.state_dict())
        self.assertTrue(extras)
        self.assertTrue(all(key.startswith(('bag1.integral_gate.', 'bag2.integral_gate.'))
                            for key in extras))
        x = torch.randn(2, 3, 64, 80)
        with torch.no_grad():
            expected, expected_edge = base(x, return_aux=True)
            actual, edge = model(x, return_aux=True)
        for first, second in zip(expected, actual):
            torch.testing.assert_close(first, second, rtol=0, atol=0)
        torch.testing.assert_close(expected_edge, edge, rtol=0, atol=0)

    def test_v8_controls_both_integral_routes_and_keeps_a_floor(self):
        block = IntegralControlledPIDBlock(16, 24, 16).eval()
        # Isolate the integral term in fusion and the integral shortcut.
        block.se = nn.Identity()
        block.fuse = nn.Identity()
        p, integral, edge = torch.randn(2, 16, 16, 20), torch.randn(2, 24, 8, 10), torch.randn(2, 1, 8, 10)
        zero, one = torch.zeros(2, 1, 1, 1), torch.ones(2, 1, 1, 1)
        with patch.object(block.pid_gate, 'forward', return_value=(zero, one, zero)), torch.no_grad():
            initial = block(p, integral, edge)
            self.assertGreater(initial.abs().sum().item(), 0)
            block.integral_gate.local[-1].bias.fill_(-1000)
            suppressed = block(p, integral, edge)
            block.integral_gate.local[-1].bias.fill_(1000)
            enhanced = block(p, integral, edge)
        # Leaving either route ungated would produce a different ratio.
        torch.testing.assert_close(suppressed, .5 * initial, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(enhanced, 1.5 * initial, rtol=1e-6, atol=1e-6)
        self.assertTrue(torch.isfinite(suppressed).all() and torch.isfinite(enhanced).all())

    def test_v9_only_changes_five_shallow_branch_normalizations(self):
        base = PIDUNetV2()
        torch.manual_seed(42)
        model = PIDUNetV9()
        locations = {name for name, layer in model.named_modules()
                     if isinstance(layer, InstanceBatchNorm2d)}
        expected = {f'enc{stage}.branches.{branch}.1'
                    for stage, count in ((1, 2), (2, 3)) for branch in range(count)}
        self.assertEqual(locations, expected)
        for key, value in base.state_dict().items():
            if not any(key.startswith(name + '.') for name in expected):
                torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
        self.assertEqual(sum(p.numel() for p in base.parameters()),
                         sum(p.numel() for p in model.parameters()))

    def test_instance_half_uses_per_image_statistics_at_inference(self):
        layer = InstanceBatchNorm2d(5).eval()
        x = torch.randn(2, 5, 12, 16)
        scale = torch.tensor([2., 3.]).view(2, 1, 1, 1)
        offset = torch.tensor([-4., 7.]).view(2, 1, 1, 1)
        with torch.no_grad():
            original = layer(x)
            shifted = layer(x * scale + offset)
        self.assertEqual(original.shape, x.shape)
        torch.testing.assert_close(original[:, :2], shifted[:, :2], rtol=1e-4, atol=1e-4)
        self.assertGreater((original[:, 2:] - shifted[:, 2:]).abs().mean().item(), 1)
        self.assertIsNone(layer.instance.running_mean)
        layer.train()
        layer(x[:1])  # Last training batch may contain only one image.
        self.assertEqual(layer.batch.num_batches_tracked.item(), 1)

    def test_disabling_each_change_recovers_v2(self):
        base = PIDUNetV2().eval()
        x = torch.randn(1, 3, 64, 80)
        with torch.no_grad():
            expected = base(x)
            for model in (PIDUNetV8(integral_gating=False), PIDUNetV9(use_ibn=False)):
                model.load_state_dict(base.state_dict(), strict=True)
                actual = model.eval()(x)
                for first, second in zip(expected, actual):
                    torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_registry_recipe_budget_and_multiscale_shapes(self):
        for name, constructor in VARIANTS:
            with self.subTest(model=name):
                self.assertIn(name, PID_V2_MODELS)
                model = get_model(name).eval()
                self.assertIsInstance(model, constructor)
                self.assertTrue(model.strict_checkpoint_loading)
                self.assertLess(sum(p.numel() for p in model.parameters()), 1_000_000)
                with torch.inference_mode():
                    for height, width in ((256, 256), (352, 352), (448, 448), (320, 384)):
                        outputs, edge = model(torch.randn(1, 3, height, width), return_aux=True)
                        self.assertEqual(len(outputs), 4)
                        for output in outputs:
                            self.assertEqual(output.shape, (1, 1, height, width))
                            self.assertTrue(torch.isfinite(output).all())
                        self.assertEqual(edge.shape, (1, 1, height // 2, width // 2))
                        self.assertTrue(torch.isfinite(edge).all())

    def test_actual_trainer_updates_new_paths_with_existing_loss(self):
        for name, constructor in VARIANTS:
            with self.subTest(model=name):
                model = constructor()
                optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
                cfg = SimpleNamespace(model=name, amp=False, scheduler='cosine', lr=3e-4,
                                      warmup_epochs=5, epochs=100, fast=True, trainsize=64,
                                      clip=.5, boundary_weight=.2)
                trainer = Trainer(model, optimizer, structure_loss, cfg, None, None)
                if name == 'pid_unet_v8':
                    parameters = []
                    for block in (model.bag1, model.bag2):
                        gate = block.integral_gate
                        parameters.extend((gate.skip.weight, gate.context.weight,
                                           gate.difference.weight, gate.boundary.weight,
                                           gate.local[-1].weight))
                else:
                    parameters = [parameter for layer in model.modules()
                                  if isinstance(layer, InstanceBatchNorm2d)
                                  for parameter in (layer.instance.weight, layer.batch.weight)]
                before = [parameter.detach().clone() for parameter in parameters]
                target = torch.zeros(2, 1, 64, 64)
                target[0, :, 12:48, 16:44] = 1
                target[1, :, 30:35, 30:35] = 1
                batches = [(torch.randn(2, 3, 64, 64), target) for _ in range(2)]
                with contextlib.redirect_stderr(io.StringIO()):
                    stats = trainer.train_one_epoch(batches, 1)
                self.assertTrue(all(math.isfinite(value) for value in stats.values()))
                for previous, parameter in zip(before, parameters):
                    self.assertIsNotNone(parameter.grad)
                    self.assertGreater(parameter.grad.abs().sum().item(), 0)
                    self.assertFalse(torch.equal(previous, parameter))
                self.assertGreater(model.boundary_branch.fuse[-1].weight.grad.abs().sum().item(), 0)
                self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all()
                                    for p in model.parameters()))

    def test_autocast_backward_with_empty_and_full_targets(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dtype = torch.float16 if device == 'cuda' else torch.bfloat16
        for name, constructor in VARIANTS:
            with self.subTest(model=name):
                model = constructor().to(device).train()
                if name == 'pid_unet_v8':
                    with torch.no_grad():
                        for block in (model.bag1, model.bag2):
                            block.integral_gate.local[-1].weight.normal_(std=.01)
                with torch.autocast(device, dtype=dtype):
                    outputs = model(torch.randn(2, 3, 64, 64, device=device))
                    target = torch.cat((torch.zeros_like(outputs[-1][:1]),
                                        torch.ones_like(outputs[-1][:1]))).float()
                    loss = sum(structure_loss(output.float(), target) for output in outputs)
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all()
                                    for p in model.parameters()))

    def test_checkpoint_round_trip_and_architecture_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pth'
            for name, constructor in VARIANTS:
                with self.subTest(model=name):
                    model, restored = constructor().eval(), constructor().eval()
                    torch.save({'state_dict': model.state_dict(), 'epoch': 7}, path)
                    with contextlib.redirect_stdout(io.StringIO()):
                        epoch, _ = Checkpointer.load(restored, path, device='cpu')
                    self.assertEqual(epoch, 7)
                    for key, value in model.state_dict().items():
                        torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
                    other = PIDUNetV9 if constructor is PIDUNetV8 else PIDUNetV8
                    for wrong in (PIDUNetV2, other):
                        torch.save({'state_dict': wrong().state_dict()}, path)
                        before = restored.inc[0].weight.detach().clone()
                        with self.assertRaisesRegex(ValueError, 'same architecture'):
                            Checkpointer.load(restored, path, device='cpu')
                        torch.testing.assert_close(before, restored.inc[0].weight, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
