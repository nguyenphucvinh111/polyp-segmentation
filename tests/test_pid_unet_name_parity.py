"""The public model name must preserve hardgate initialization and computation."""
import unittest
import torch
from models.pid_unet import PIDUnet
from models.pid_unet_ablation import PIDUNetV2_HardGate


class NameParityTest(unittest.TestCase):
    def test_seed_weights_rng_and_outputs(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        reference = PIDUNetV2_HardGate().eval()
        rng = torch.get_rng_state().clone()
        torch.manual_seed(42)
        renamed = PIDUnet().eval()
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(reference.state_dict().keys(), renamed.state_dict().keys())
        for key, value in reference.state_dict().items():
            self.assertTrue(torch.equal(value, renamed.state_dict()[key]), key)
        self.assertEqual(sum(p.numel() for p in renamed.parameters()), 814573)
        with torch.no_grad():
            x = torch.randn(1, 3, 352, 352)
            heads_a, boundary_a = reference(x, return_aux=True)
            heads_b, boundary_b = renamed(x, return_aux=True)
        for a, b in zip((*heads_a, boundary_a), (*heads_b, boundary_b)):
            self.assertTrue(torch.equal(a, b))


if __name__ == '__main__':
    unittest.main()
