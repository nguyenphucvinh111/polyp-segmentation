"""Predeclared, one-factor-at-a-time study of the original PID-UNet v2."""

VARIANTS = {
    'full': 'Frozen hardgate hybrid-bottleneck control',
    'no_global': 'Remove additive global context at bottleneck',
    'no_derivative': 'Set D contribution to zero; keep three-way gain softmax',
    'no_integral': 'Set I contribution to zero; evaluate Proportional + Derivative only',
    'fixed_gains': 'Replace adaptive gains with Kp = Ki = Kd = 1/3',
    'no_boundary_gate': 'Replace boundary multiplier in D with one',
    'soft_floor': 'Add a learned bounded floor initialized at 0.35',
    'no_boundary_loss': 'Train full control without boundary supervision (boundary_weight=0.0)',
}
MODEL_NAMES = {key: 'pid_hardgate_ablation_' + key for key in VARIANTS}
TEST_COUNTS = {'CVC-300': 60, 'CVC-ClinicDB': 62, 'CVC-ColonDB': 380,
               'ETIS-LaribPolypDB': 196, 'Kvasir': 100}
METRICS = ('mDice', 'mIoU', 'Recall', 'Precision', 'Specificity',
           'HD95', 'wFb', 'Sm', 'Em', 'MAE')
LOWER_IS_BETTER = {'HD95', 'MAE'}
PROTOCOL = {
    'version': 1, 'loss': 'structure_loss', 'boundary_weight': 0.2,
    'lr': 3e-4, 'weight_decay': 1e-4, 'scheduler': 'cosine',
    'warmup_epochs': 5, 'resolution': 352, 'clip': 0.5,
    'multiscale': [0.75, 1.0, 1.25], 'amp': True, 'tta': False, 'drop_last': True,
    'pretrained': False, 'split_seed': 42, 'validation_fraction': 0.1,
    'checkpoint_selection': 'validation_mDice',
}


def run_tag(variant, seed):
    return f'hgate_{variant}_seed{seed}'
