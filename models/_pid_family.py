"""Models sharing the original v2 training recipe and boundary supervision."""

from experiments.pid_v2_ablation import MODEL_NAMES

PID_V2_MODELS = frozenset({
    "pid_unet_v2", "pid_unet_v3", "pid_unet_v4", "pid_unet_v5", "pid_unet_v6",
    "pid_unet_v2_plus", "pid_unet_v7", "pid_unet_v8", "pid_unet_v9", "pid_unet_v10",
    "pid_unet_v2_os8",
    # Ablation variants
    "pid_unet_v2_nogc", "pid_unet_v2_nobound", "pid_unet_v2_nod", "pid_unet_v2_noi",
    "pid_unet_v2_static", "pid_unet_v2_hardgate", "pid_unet_v2_stddown",
}) | frozenset(MODEL_NAMES.values()) | frozenset(
    'pid_hardgate_ablation_' + variant for variant in
    ('full', 'soft_floor', 'no_derivative', 'fixed_gains', 'no_boundary_gate', 'no_global'))
