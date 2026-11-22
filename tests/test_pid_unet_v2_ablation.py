"""Scientific controls, actual trainer integration, and checkpoint separation."""

import contextlib
import io
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from experiments.pid_v2_ablation import MODEL_NAMES, VARIANTS
from loss.structure_loss import structure_loss
from models import get_model, list_models
from models._pid_family import PID_V2_MODELS
from models.pid_unet_v2 import PIDUNetV2
from models.pid_unet_v2_ablation import PIDUNetV2Ablation
from utils.checkpoint import Checkpointer
from utils.trainer import Trainer


class V2AblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_control_exactly_matches_original_and_common_initialization(self):
        torch.manual_seed(42)
        original = PIDUNetV2().eval()
        state, rng = original.state_dict(), torch.get_rng_state()
        x = torch.randn(2, 3, 64, 80)
        with torch.no_grad():
            expected, expected_boundary = original(x, return_aux=True)
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                torch.manual_seed(42)
                model = PIDUNetV2Ablation(variant).eval()
                torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
                for key, value in model.state_dict().items():
                    torch.testing.assert_close(value, state[key], rtol=0, atol=0)
                self.assertLess(sum(p.numel() for p in model.parameters()), 1_000_000)
                if variant == 'full':
                    with torch.no_grad():
                        actual, boundary = model(x, return_aux=True)
                    for left, right in zip(expected, actual):
                        torch.testing.assert_close(left, right, rtol=0, atol=0)
                    torch.testing.assert_close(boundary, expected_boundary, rtol=0, atol=0)

    def test_all_registered_with_original_training_recipe(self):
        # Force discovery before relying on the lazily populated registry.
        import models
        models._auto_import()
        for variant, name in MODEL_NAMES.items():
            self.assertIn(name, list_models())
            self.assertIn(name, PID_V2_MODELS)
            self.assertEqual(get_model(name).variant, variant)

    def test_v2_keeps_historical_backbone_and_handles_singleton_batch(self):
        from models._hardgate_base import EnhancedTinyPIDUNet
        model = PIDUNetV2().train()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 835373)
        self.assertFalse(model.bottleneck[0].use_global)
        rates = [branch[3].conv1.dilation[0] for branch in model.bottleneck[0].branches]
        self.assertEqual(rates, [1, 3, 6, 12])
        outputs, boundary = model(torch.randn(1, 3, 64, 64), return_aux=True)
        (sum(output.square().mean() for output in outputs) + boundary.square().mean()).backward()
        self.assertTrue(torch.isfinite(model.inc[0].weight.grad).all())
        # Preserve the independently updated unversioned pid_unet default.
        latest = EnhancedTinyPIDUNet()
        self.assertTrue(latest.bottleneck.use_global)
        self.assertEqual([branch[3].conv1.dilation[0] for branch in latest.bottleneck.branches],
                         [1, 2, 4, 8])

    def test_boundary_guidance_removed_only_where_requested(self):
        for variant in ('full', 'no_derivative', 'no_boundary_gate', 'no_gate_floor'):
            with self.subTest(variant=variant):
                model = PIDUNetV2Ablation(variant).eval()
                p = torch.randn(2, 32, 16, 16)
                i = torch.randn(2, 64, 8, 8)
                boundary = torch.randn(2, 1, 8, 8, requires_grad=True)
                model.bag1(p, i, boundary).square().mean().backward()
                if variant in ('no_derivative', 'no_boundary_gate'):
                    self.assertIsNone(boundary.grad)
                else:
                    self.assertGreater(boundary.grad.abs().sum().item(), 0)

    def test_block_matches_isolated_equation(self):
        from torch.nn import functional as F
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                block = PIDUNetV2Ablation(variant).eval().bag1
                p_feat, i_feat = torch.randn(2, 32, 8, 8), torch.randn(2, 64, 4, 4)
                b = torch.randn(2, 1, 4, 4)
                with torch.no_grad():
                    actual = block(p_feat, i_feat, b)
                    i_up = F.interpolate(i_feat, (8, 8), mode='bilinear', align_corners=True)
                    b_up = F.interpolate(b, (8, 8), mode='bilinear', align_corners=True)
                    p, i = block.proj_p(p_feat), block.proj_i(i_up)
                    sal = block.saliency_gate(i_up)
                    floor = 0 if variant == 'no_gate_floor' else .6 * block.alpha.sigmoid()
                    gate = floor + (1 - floor) * sal
                    if variant == 'fixed_gains':
                        gains = [i.new_full((2, 1, 1, 1), 1 / 3)] * 3
                    else:
                        gains = block.pid_gate(i_up)
                    d = (p - F.avg_pool2d(p, 3, 1, 1)).abs()
                    if variant == 'no_derivative':
                        d = d * 0
                    d = d * gate * (1 if variant == 'no_boundary_gate' else b_up.sigmoid())
                    expected = block.fuse(block.se(gains[0] * p * gate + gains[1] * i + gains[2] * d)) + i
                    torch.testing.assert_close(actual, expected)

    def test_multiscale_shapes_and_finite_outputs(self):
        for variant in VARIANTS:
            model = PIDUNetV2Ablation(variant).eval()
            for size in (256, 352, 448):
                with self.subTest(variant=variant, size=size), torch.inference_mode():
                    outputs, boundary = model(torch.randn(1, 3, size, size), return_aux=True)
                    self.assertEqual(len(outputs), 4)
                    for output in outputs:
                        self.assertEqual(output.shape, (1, 1, size, size))
                        self.assertTrue(torch.isfinite(output).all())
                    self.assertEqual(boundary.shape, (1, 1, size // 2, size // 2))

    def test_existing_trainer_keeps_boundary_supervision_for_all_variants(self):
        target = torch.zeros(2, 1, 64, 64)
        target[:, :, 16:43, 24:48] = 1
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                model = PIDUNetV2Ablation(variant)
                cfg = SimpleNamespace(model=MODEL_NAMES[variant], amp=False, scheduler='cosine',
                                      lr=3e-4, warmup_epochs=5, epochs=100, fast=True,
                                      trainsize=64, clip=.5, boundary_weight=.2)
                optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
                trainer = Trainer(model, optimizer, structure_loss, cfg, None, None)
                before = model.head1.weight.detach().clone()
                device = trainer.device
                with contextlib.redirect_stderr(io.StringIO()):
                    stats = trainer.train_one_epoch([(torch.randn(2, 3, 64, 64), target)], 1)
                self.assertTrue(all(math.isfinite(v) for v in stats.values()))
                self.assertFalse(torch.equal(before.to(device), model.head1.weight))
                self.assertGreater(model.boundary_branch.fuse[-1].weight.grad.abs().sum().item(), 0)
                self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all()
                                    for p in model.parameters()))

    def test_same_shape_wrong_variant_checkpoint_rejected_before_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            saved = PIDUNetV2Ablation('no_derivative')
            other = PIDUNetV2Ablation('no_boundary_gate')
            self.assertEqual(set(saved.state_dict()), set(other.state_dict()))
            checkpointer = Checkpointer(directory, 'test')
            with contextlib.redirect_stdout(io.StringIO()):
                checkpointer.save(saved, None, 1, {'mDice': .4})
            before = other.inc[0].weight.clone()
            with self.assertRaisesRegex(ValueError, 'same architecture'):
                Checkpointer.load(other, checkpointer.get_best_path(), device='cpu')
            torch.testing.assert_close(before, other.inc[0].weight, rtol=0, atol=0)
            restored = PIDUNetV2Ablation('no_derivative')
            with contextlib.redirect_stdout(io.StringIO()):
                epoch, _ = Checkpointer.load(restored, checkpointer.get_best_path(), device='cpu')
            self.assertEqual(epoch, 1)
            for key, value in saved.state_dict().items():
                torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)

    def test_bfloat16_backward_all_variants(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                model = PIDUNetV2Ablation(variant)
                with torch.autocast('cpu', dtype=torch.bfloat16):
                    outputs = model(torch.randn(2, 3, 64, 64))
                    target = torch.zeros_like(outputs[-1]).float()
                    target[1] = 1
                    loss = sum(structure_loss(output.float(), target) for output in outputs)
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all()
                                    for p in model.parameters()))


if __name__ == '__main__':
    unittest.main()
