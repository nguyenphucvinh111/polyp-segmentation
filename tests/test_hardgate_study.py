import unittest
from types import SimpleNamespace
from pathlib import Path
import torch

from models import get_model
from models.hardgate_study import VARIANTS, HardgateStudy
from experiments.hardgate_ablation import MODEL_NAMES
from run_hardgate_ablation import train_command


class HardgateStudyTests(unittest.TestCase):
    def test_all_variants_registered_and_parameters_under_1m(self):
        x = torch.randn(2, 3, 352, 352)
        for variant in VARIANTS:
            model_name = MODEL_NAMES[variant]
            model = get_model(model_name)
            self.assertIsInstance(model, HardgateStudy)
            self.assertEqual(model.checkpoint_architecture, f"hardgate_study.v1.{variant}")
            
            param_count = sum(p.numel() for p in model.parameters())
            self.assertLess(param_count, 1_000_000, f"{model_name} exceeded 1M params: {param_count}")
            
            # Forward pass
            out, boundary = model(x, return_aux=True)
            self.assertEqual(len(out), 4)
            self.assertEqual(out[-1].shape, (2, 1, 352, 352))
            self.assertEqual(boundary.shape, (2, 1, 176, 176))

    def test_no_boundary_loss_train_command_uses_zero_weight(self):
        args = SimpleNamespace(epochs=100, batchsize=8, output_root='.', data_root='./data')
        cmd_full = train_command(args, 'full', 42)
        cmd_nobound = train_command(args, 'no_boundary_loss', 42)
        
        bw_idx_full = cmd_full.index('--boundary_weight') + 1
        bw_idx_nobound = cmd_nobound.index('--boundary_weight') + 1
        
        self.assertEqual(cmd_full[bw_idx_full], '0.2')
        self.assertEqual(cmd_nobound[bw_idx_nobound], '0.0')

    def test_internal_branch_modulations(self):
        # Test no_derivative
        m_nod = get_model('pid_hardgate_ablation_no_derivative')
        self.assertEqual(m_nod.bag1.variant, 'no_derivative')

        # Test no_integral
        m_noi = get_model('pid_hardgate_ablation_no_integral')
        self.assertEqual(m_noi.bag1.variant, 'no_integral')

        # Test fixed_gains
        m_fixed = get_model('pid_hardgate_ablation_fixed_gains')
        self.assertEqual(m_fixed.bag1.variant, 'fixed_gains')


if __name__ == '__main__':
    unittest.main()
