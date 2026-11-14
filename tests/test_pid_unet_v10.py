import contextlib
import io
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from loss.structure_loss import structure_loss
from models import get_model
from models._pid_family import PID_V2_MODELS
from models.pid_unet_v2 import PIDUNetV2
from models.pid_unet_v8 import PIDUNetV8
from models.pid_unet_v9 import PIDUNetV9
from models.pid_unet_v10 import PIDUNetV10, PooledContextMixer, SpatialAttention
from utils.checkpoint import Checkpointer
from utils.trainer import Trainer


class PIDUNetV10Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(42)

    def test_original_v2_weights_normalization_and_initial_outputs_preserved(self):
        base = PIDUNetV2().eval()
        torch.manual_seed(42)
        model = PIDUNetV10().eval()
        for key, value in base.state_dict().items():
            torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
        extras = set(model.state_dict()) - set(base.state_dict())
        self.assertTrue(extras)
        self.assertTrue(all(key.startswith('context_bridge.') for key in extras))
        for name, module in base.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                self.assertIsInstance(model.get_submodule(name), nn.BatchNorm2d)
        images = torch.randn(2, 3, 64, 80)
        with torch.no_grad():
            expected, expected_edge = base(images, return_aux=True)
            actual, edge = model(images, return_aux=True)
        for first, second in zip(expected, actual):
            torch.testing.assert_close(first, second, rtol=0, atol=0)
        torch.testing.assert_close(expected_edge, edge, rtol=0, atol=0)

    def test_disabling_bridge_is_exact_v2_architecture(self):
        base = PIDUNetV2().eval()
        model = PIDUNetV10(context_bridge=False).eval()
        model.load_state_dict(base.state_dict(), strict=True)
        images = torch.randn(1, 3, 64, 80)
        with torch.inference_mode():
            for first, second in zip(base(images), model(images)):
                torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_registry_recipe_budget_and_multiscale_interfaces(self):
        model = get_model('pid_unet_v10').eval()
        self.assertIsInstance(model, PIDUNetV10)
        self.assertIn('pid_unet_v10', PID_V2_MODELS)
        self.assertTrue(model.strict_checkpoint_loading)
        self.assertLess(sum(parameter.numel() for parameter in model.parameters()), 1_000_000)
        observed = []
        handle = model.context_bridge.mixers[0].attention.register_forward_pre_hook(
            lambda module, inputs: observed.append((inputs[0].shape[-2], inputs[1].shape[-2])))
        with torch.inference_mode():
            for height, width in ((256, 256), (352, 352), (448, 448), (320, 384)):
                outputs, edge = model(torch.randn(1, 3, height, width), return_aux=True)
                self.assertEqual(len(outputs), 4)
                for output in outputs:
                    self.assertEqual(output.shape, (1, 1, height, width))
                    self.assertTrue(torch.isfinite(output).all())
                self.assertEqual(edge.shape, (1, 1, height // 2, width // 2))
                self.assertTrue(torch.isfinite(edge).all())
                self.assertEqual(observed[-1], (height * width // 16, 85))
        handle.remove()

    def test_attention_matches_reference_and_accounts_for_matrix_macs(self):
        layer = SpatialAttention()
        query = torch.randn(2, 4, 19, 16, requires_grad=True)
        key = torch.randn(2, 4, 7, 16, requires_grad=True)
        value = torch.randn(2, 4, 7, 16, requires_grad=True)
        expected = ((query @ key.transpose(-2, -1)) / 4).softmax(-1) @ value
        actual = layer(query, key, value)
        torch.testing.assert_close(actual, expected)
        actual.square().mean().backward()
        for tensor in (query, key, value):
            self.assertTrue(torch.isfinite(tensor.grad).all())
            self.assertGreater(tensor.grad.abs().sum().item(), 0)
        layer.register_buffer('total_ops', torch.zeros(1, dtype=torch.float64))
        layer.count_macs(layer, (query, key, value), actual)
        self.assertEqual(layer.total_ops.item(), 2 * 4 * 19 * 7 * (16 + 16))

    def test_context_can_communicate_beyond_local_receptive_field(self):
        mixer = PooledContextMixer(width=16).eval()
        features = torch.randn(1, 16, 24, 32)
        changed = features.clone()
        changed[:, 0, -3:, -3:] += 20
        with torch.inference_mode():
            result, other = mixer(features), mixer(changed)
        # The opposite corner cannot be reached by the two local 3x3 filters.
        # Its change must come through the pooled spatial attention.
        self.assertGreater((result[:, :, :2, :2] - other[:, :, :2, :2]).abs().max().item(), 1e-6)

    def test_profiler_includes_attention_products(self):
        from utils.edge_profiler import EdgeProfiler, HAS_THOP
        if not HAS_THOP:
            self.skipTest('THOP is not installed')
        import thop
        model = PIDUNetV10().eval()
        dummy = torch.randn(1, 3, 64, 64)
        without_attention, _ = thop.profile(model, inputs=(dummy,), verbose=False)
        with_attention = EdgeProfiler(model, testsize=64, device=torch.device('cpu')).measure_flops() * 1e9
        # Two blocks, two matrix multiplies, 16x16 queries, 85 keys, 64 channels.
        self.assertAlmostEqual(with_attention - without_attention, 2 * 2 * 256 * 85 * 64, places=3)

    def test_existing_trainer_trains_bridge_and_boundary_with_same_loss(self):
        model = PIDUNetV10()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        cfg = SimpleNamespace(model='pid_unet_v10', amp=False, scheduler='cosine', lr=3e-4,
                              warmup_epochs=5, epochs=100, fast=True, trainsize=64,
                              clip=.5, boundary_weight=.2)
        trainer = Trainer(model, optimizer, structure_loss, cfg, None, None)
        watched = [layer.weight for layer in model.context_bridge.inputs]
        watched += [layer.weight for layer in model.context_bridge.outputs]
        for mixer in model.context_bridge.mixers:
            watched += [mixer.query.weight, mixer.key_value.weight, mixer.feed_forward[0].weight]
        watched += [model.inc[0].weight, model.boundary_branch.fuse[-1].weight]
        before = [parameter.detach().clone() for parameter in watched]
        target = torch.zeros(2, 1, 64, 64)
        target[0, :, 16:48, 12:52] = 1
        target[1, :, 28:33, 28:33] = 1
        # A zero output projection opens the internal gradient path after step 1.
        # Include batch size 1, as the real training loader retains its last batch.
        batches = [(torch.randn(2, 3, 64, 64), target),
                   (torch.randn(1, 3, 64, 64), target[1:])]
        with contextlib.redirect_stderr(io.StringIO()):
            stats = trainer.train_one_epoch(batches, 1)
        self.assertTrue(all(math.isfinite(value) for value in stats.values()))
        for previous, parameter in zip(before, watched):
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
            self.assertFalse(torch.equal(previous, parameter))
        self.assertTrue(all(parameter.grad is None or torch.isfinite(parameter.grad).all()
                            for parameter in model.parameters()))

    def test_autocast_backward_empty_and_full_masks_after_bridge_activation(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dtype = torch.float16 if device == 'cuda' else torch.bfloat16
        model = PIDUNetV10().to(device).train()
        with torch.no_grad():
            for layer in model.context_bridge.outputs:
                layer.weight.normal_(std=.01)
        with torch.autocast(device, dtype=dtype):
            outputs = model(torch.randn(2, 3, 64, 64, device=device))
            target = torch.cat((torch.zeros_like(outputs[-1][:1]),
                                torch.ones_like(outputs[-1][:1]))).float()
            loss = sum(structure_loss(output.float(), target) for output in outputs)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for parameter in model.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(model.context_bridge.mixers[0].query.weight.grad.abs().sum().item(), 0)

    def test_strict_checkpoint_roundtrip_and_wrong_model_rejection(self):
        model, restored = PIDUNetV10().eval(), PIDUNetV10().eval()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pth'
            torch.save({'state_dict': model.state_dict(), 'epoch': 9}, path)
            with contextlib.redirect_stdout(io.StringIO()):
                epoch, _ = Checkpointer.load(restored, path, device='cpu')
            self.assertEqual(epoch, 9)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
            for constructor in (PIDUNetV2, PIDUNetV8, PIDUNetV9):
                torch.save({'state_dict': constructor().state_dict()}, path)
                previous = restored.inc[0].weight.detach().clone()
                with self.assertRaisesRegex(ValueError, 'same architecture'):
                    Checkpointer.load(restored, path, device='cpu')
                torch.testing.assert_close(previous, restored.inc[0].weight, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
