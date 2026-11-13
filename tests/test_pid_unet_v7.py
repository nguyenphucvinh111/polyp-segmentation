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
from models.pid_unet_v7 import PIDUNetV7, SemanticPIDBlock
from utils.checkpoint import Checkpointer
from utils.trainer import Trainer


class PIDUNetV7Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(42)

    def test_disabled_changes_recover_v2_function(self):
        base = PIDUNetV2().eval()
        model = PIDUNetV7(multiscale=False, semantic_guidance=False).eval()
        missing, unexpected = model.load_state_dict(base.state_dict(), strict=False)
        self.assertFalse(unexpected)
        self.assertTrue(all('.local_' in name for name in missing))
        x = torch.randn(2, 3, 64, 80)
        with torch.no_grad():
            expected, edge = base(x, return_aux=True)
            actual, actual_edge = model(x, return_aux=True)
        for a, b in zip(expected, actual):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(edge, actual_edge, rtol=0, atol=0)

    def test_multiscale_shapes_and_budget(self):
        model = PIDUNetV7().eval()
        self.assertLess(sum(p.numel() for p in model.parameters()), 1_000_000)
        with torch.inference_mode():
            for height, width in ((256, 256), (352, 352), (448, 448), (320, 384)):
                with self.subTest(size=(height, width)):
                    outputs, edge = model(torch.randn(1, 3, height, width), return_aux=True)
                    self.assertEqual(len(outputs), 4)
                    for output in outputs:
                        self.assertEqual(output.shape, (1, 1, height, width))
                        self.assertTrue(torch.isfinite(output).all())
                    self.assertEqual(edge.shape, (1, 1, height // 2, width // 2))

    def test_coarse_predictions_control_gate_and_receive_gradients(self):
        block = SemanticPIDBlock(16, 16, 16).eval()
        p, integral = torch.randn(2, 16, 16, 16), torch.randn(2, 16, 16, 16)
        prior = torch.full((2, 1, 16, 16), .5)
        coarse = torch.full((2, 1, 8, 8), 4., requires_grad=True)
        positive = block.semantic_mask(p, integral, prior, coarse)
        negative = block.semantic_mask(p, integral, prior, -coarse)
        self.assertTrue((positive > negative).all())
        positive.mean().backward()
        self.assertGreater(coarse.grad.abs().sum().item(), 0)
        for value in (-1000., 1000.):
            mask = block.semantic_mask(p, integral, prior,
                                       torch.full((2, 1, 8, 8), value))
            self.assertTrue(torch.isfinite(mask).all())
            self.assertTrue(((mask >= 0) & (mask <= 1)).all())

    def test_actual_trainer_updates_multiscale_and_local_paths(self):
        model = PIDUNetV7()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        cfg = SimpleNamespace(model='pid_unet_v7', amp=False, scheduler='cosine', lr=3e-4,
                              warmup_epochs=5, epochs=100, fast=True, trainsize=64,
                              clip=.5, boundary_weight=.2)
        trainer = Trainer(model, optimizer, structure_loss, cfg, None, None)
        target = torch.zeros(2, 1, 64, 64)
        target[0, :, 12:48, 16:44] = 1
        target[1, :, 30:35, 30:35] = 1
        batches = [(torch.randn(2, 3, 64, 64), target) for _ in range(2)]
        with contextlib.redirect_stderr(io.StringIO()):
            stats = trainer.train_one_epoch(batches, 1)
        self.assertTrue(all(math.isfinite(value) for value in stats.values()))
        for name in ('bag1', 'bag2', 'bag3', 'bag4'):
            block = getattr(model, name)
            parameters = [branch[0].weight for branch in block.fuse.branches]
            parameters += [block.local_gate[-1].weight, block.local_p.weight]
            for parameter in parameters:
                self.assertIsNotNone(parameter.grad)
                self.assertGreater(parameter.grad.abs().sum().item(), 0)
        self.assertGreater(model.boundary_branch.fuse[-1].weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all()
                            for p in model.parameters()))

    def test_autocast_final_head_loss_reaches_coarse_heads(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dtype = torch.float16 if device == 'cuda' else torch.bfloat16
        model = PIDUNetV7().to(device).train()
        with torch.autocast(device, dtype=dtype):
            outputs = model(torch.randn(2, 3, 64, 64, device=device))
            target = (torch.rand_like(outputs[-1]) > .5).float()
            loss = structure_loss(outputs[-1].float(), target)
        loss.backward()
        for head in (model.head2, model.head3, model.head4):
            self.assertIsNotNone(head.weight.grad)
            self.assertGreater(head.weight.grad.abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all()
                            for p in model.parameters()))

    def test_checkpoint_round_trip_and_v2_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            model = PIDUNetV7().eval()
            path = Path(directory) / 'v7.pth'
            torch.save({'state_dict': model.state_dict(), 'epoch': 7}, path)
            restored = PIDUNetV7().eval()
            with contextlib.redirect_stdout(io.StringIO()):
                epoch, _ = Checkpointer.load(restored, path, device='cpu')
            self.assertEqual(epoch, 7)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
            before = restored.inc[0].weight.detach().clone()
            torch.save({'state_dict': PIDUNetV2().state_dict()}, path)
            with self.assertRaisesRegex(ValueError, 'same architecture'):
                Checkpointer.load(restored, path, device='cpu')
            torch.testing.assert_close(before, restored.inc[0].weight, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
