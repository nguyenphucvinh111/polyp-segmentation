"""CPU regression tests; CUDA AMP is also checked when a GPU is available."""

import contextlib
import io
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from loss.structure_loss import structure_loss
from models.pid_unet_v2 import PIDUNetV2
from models.pid_unet_v2_plus import ForegroundBackgroundContext, PIDUNetV2Plus
from utils.checkpoint import Checkpointer
from utils.trainer import Trainer


class PIDUNetV2PlusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(42)

    def test_v2_initial_outputs_and_base_weights_are_preserved(self):
        baseline = PIDUNetV2().eval()
        candidate = PIDUNetV2Plus().eval()
        missing, unexpected = candidate.load_state_dict(baseline.state_dict(), strict=False)
        self.assertFalse(unexpected)
        self.assertTrue(missing)
        self.assertTrue(all(key.startswith(('context1.', 'context2.', 'context3.'))
                            for key in missing))
        x = torch.randn(2, 3, 64, 80)
        with torch.no_grad():
            expected, expected_boundary = baseline(x, return_aux=True)
            actual, actual_boundary = candidate(x, return_aux=True)
        for left, right in zip(expected, actual):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        torch.testing.assert_close(expected_boundary, actual_boundary, rtol=0, atol=0)

    def test_shapes_and_parameter_budget(self):
        model = PIDUNetV2Plus().eval()
        self.assertLess(sum(p.numel() for p in model.parameters()), 1_000_000)
        with torch.inference_mode():
            for height, width in ((256, 256), (352, 352), (448, 448), (320, 384)):
                with self.subTest(size=(height, width)):
                    outputs, boundary = model(torch.randn(1, 3, height, width), return_aux=True)
                    self.assertEqual(len(outputs), 4)
                    for output in outputs:
                        self.assertEqual(output.shape, (1, 1, height, width))
                        self.assertTrue(torch.isfinite(output).all())
                    self.assertEqual(boundary.shape, (1, 1, height // 2, width // 2))

    def test_region_context_distinguishes_foreground_from_background(self):
        # Opposite source regions and queries: foreground affinity must follow
        # appearance, and reversing soft region assignments reverses affinity.
        source = torch.tensor([[[[1., -1.]], [[-1., 1.]]]])
        logits = torch.tensor([[[[12., -12.]]]])
        context, affinity = ForegroundBackgroundContext.region_context(source, source, logits)
        self.assertGreater(affinity[0, 0, 0, 0].item(), .99)
        self.assertLess(affinity[0, 0, 0, 1].item(), .01)
        _, inverted = ForegroundBackgroundContext.region_context(source, source, -logits)
        torch.testing.assert_close(inverted, 1 - affinity, atol=1e-6, rtol=1e-5)
        self.assertTrue(torch.isfinite(context).all())

    def test_extreme_coarse_logits_are_finite(self):
        for value in (-1000., 0., 1000.):
            with self.subTest(logit=value):
                query = torch.randn(2, 24, 8, 8, requires_grad=True)
                source = torch.randn(2, 24, 4, 4, requires_grad=True)
                logits = torch.full((2, 1, 4, 4), value, requires_grad=True)
                context, affinity = ForegroundBackgroundContext.region_context(query, source, logits)
                (context.square().mean() + affinity.square().mean()).backward()
                for tensor in (context, affinity, query.grad, source.grad, logits.grad):
                    self.assertTrue(torch.isfinite(tensor).all())

    def test_real_trainer_updates_context_with_existing_loss(self):
        model = PIDUNetV2Plus()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        cfg = SimpleNamespace(
            model='pid_unet_v2_plus', amp=False, scheduler='cosine', lr=3e-4,
            warmup_epochs=5, epochs=100, fast=True, trainsize=64,
            clip=.5, boundary_weight=.2,
        )
        trainer = Trainer(model, optimizer, structure_loss, cfg, None, None)
        target = torch.zeros(2, 1, 64, 64)
        target[0, :, 12:42, 16:44] = 1
        target[1, :, 30:34, 30:34] = 1
        batches = [(torch.randn(2, 3, 64, 64), target) for _ in range(2)]
        with contextlib.redirect_stderr(io.StringIO()):
            stats = trainer.train_one_epoch(batches, 1)
        self.assertTrue(all(math.isfinite(value) for value in stats.values()))
        for name in ('context1', 'context2', 'context3'):
            module = getattr(model, name)
            for parameter in (module.query[0].weight, module.key_value[0].weight,
                              module.refine[-1].weight):
                self.assertIsNotNone(parameter.grad)
                self.assertGreater(parameter.grad.abs().sum().item(), 0)
                self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(model.boundary_branch.fuse[-1].weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all()
                            for p in model.parameters()))

    def test_checkpoint_round_trip_and_wrong_architecture_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            model = PIDUNetV2Plus().eval()
            path = Path(directory) / 'plus.pth'
            torch.save({'state_dict': model.state_dict(), 'epoch': 7}, path)
            restored = PIDUNetV2Plus().eval()
            with contextlib.redirect_stdout(io.StringIO()):
                epoch, _ = Checkpointer.load(restored, path, device='cpu')
            self.assertEqual(epoch, 7)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
            torch.save({'state_dict': PIDUNetV2().state_dict()}, path)
            before = restored.inc[0].weight.detach().clone()
            with self.assertRaisesRegex(ValueError, 'same architecture'):
                Checkpointer.load(restored, path, device='cpu')
            torch.testing.assert_close(before, restored.inc[0].weight, rtol=0, atol=0)

    def test_bfloat16_autocast_and_full_empty_targets(self):
        model = PIDUNetV2Plus().train()
        # Nonzero projection exercises the complete attention backward path.
        with torch.no_grad():
            for module in (model.context1, model.context2, model.context3):
                module.refine[-1].weight.normal_(std=.01)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            outputs = model(torch.randn(2, 3, 64, 64))
            target = torch.cat((torch.zeros_like(outputs[-1][:1]),
                                torch.ones_like(outputs[-1][:1])))
            loss = sum(structure_loss(output.float(), target.float()) for output in outputs)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all()
                            for p in model.parameters()))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA GPU not available')
    def test_cuda_fp16_backward(self):
        model = PIDUNetV2Plus().cuda().train()
        with torch.no_grad():
            for module in (model.context1, model.context2, model.context3):
                module.refine[-1].weight.normal_(std=.01)
        with torch.autocast('cuda', dtype=torch.float16):
            outputs = model(torch.randn(2, 3, 256, 256, device='cuda'))
            target = (torch.rand_like(outputs[-1]) > .5).float()
            loss = sum(structure_loss(output, target) for output in outputs)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all()
                            for p in model.parameters()))


if __name__ == '__main__':
    unittest.main()
