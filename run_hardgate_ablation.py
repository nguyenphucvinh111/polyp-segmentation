"""Run/resume the controlled v2 ablation study; inspect first with --dry_run."""

import argparse
import csv
import os
from pathlib import Path
import shlex
import subprocess
import sys

from experiments.ablation_io import file_hash, object_hash, read_json, read_metrics, write_json
from experiments.hardgate_ablation import MODEL_NAMES, PROTOCOL, TEST_COUNTS, VARIANTS, run_tag

REPO = Path(__file__).resolve().parent


def source_fingerprint():
    paths = [REPO / name for name in ('train.py', 'test.py', 'run_hardgate_ablation.py')]
    for directory in ('models', 'utils', 'loss', 'experiments'):
        paths.extend((REPO / directory).rglob('*.py'))
    return {path.relative_to(REPO).as_posix(): file_hash(path) for path in sorted(paths)}


def dataset_identity(data_root):
    """Hash paired image/mask bytes, including test sets, without using test scores."""
    import numpy as np
    from utils.dataloader import PolypDataset

    root = Path(data_root)
    identities = {}
    split = None
    for name, relative, expected in [('TrainDataset', Path('TrainDataset'), 1450)] + [
            (name, Path('TestDataset') / name, count) for name, count in TEST_COUNTS.items()]:
        dataset = PolypDataset(str(root / relative / 'images'),
                               str(root / relative / 'masks'), 352, is_train=False)
        if len(dataset) != expected:
            raise ValueError(f'{name}: expected {expected} pairs, got {len(dataset)}; check dataset.')
        pairs = [{'image': Path(image).name, 'mask': Path(mask).name,
                  'image_sha256': file_hash(image), 'mask_sha256': file_hash(mask)}
                 for image, mask in zip(dataset.images, dataset.gts)]
        identities[name] = {'count': len(pairs), 'pairs_sha256': object_hash(pairs)}
        if name == 'TrainDataset':
            indices = np.random.default_rng(42).permutation(len(dataset)).tolist()
            nval = max(1, round(len(dataset) * .1))
            split = {'train': [Path(dataset.images[i]).name for i in indices[nval:]],
                     'validation': [Path(dataset.images[i]).name for i in indices[:nval]]}
    return {'datasets': identities, 'split': split}


def ensure_study(root, identity, provenance):
    path = root / 'ablation_study.json'
    if path.exists():
        saved = read_json(path)
        if saved['identity'] != identity:
            raise ValueError('Study protocol, source, data or runtime changed. '
                             'Restore the original setup or use a new --output_root.')
        return saved
    if any((root / name).exists() and any((root / name).iterdir())
           for name in ('logs', 'snapshots', 'results')):
        raise ValueError('Outputs exist without an ablation manifest; use a fresh --output_root.')
    study = {'identity': identity, 'identity_sha256': object_hash(identity),
             'provenance': provenance, 'variants': VARIANTS}
    write_json(path, study)
    return study


def train_command(args, variant, seed, resume=None):
    boundary_weight = 0.0 if variant == 'no_boundary_loss' else PROTOCOL['boundary_weight']
    command = [sys.executable, '-u', str(REPO / 'train.py'),
               '--model', MODEL_NAMES[variant], '--seeds', str(seed),
               '--run_name', run_tag(variant, seed), '--loss', PROTOCOL['loss'],
               '--epochs', str(args.epochs), '--batchsize', str(args.batchsize),
               '--lr', str(PROTOCOL['lr']), '--weight_decay', str(PROTOCOL['weight_decay']),
               '--scheduler', PROTOCOL['scheduler'], '--warmup_epochs', '5',
               '--boundary_weight', str(boundary_weight),
               '--output_root', str(Path(args.output_root).resolve()),
               '--data_root', str(Path(args.data_root).resolve())]
    if resume:
        command.extend(['--resume', str(resume)])
    return command


def run_state(root, variant, seed, study_hash):
    tag = run_tag(variant, seed)
    directory = root / 'snapshots' / tag
    done = root / 'ablation' / tag / 'completed.json'
    metrics = root / 'results' / tag / 'metrics.csv'
    if done.exists():
        record = read_json(done)
        if (record['study_sha256'] != study_hash or record['variant'] != variant
                or record['seed'] != seed or record['tag'] != tag):
            raise ValueError(f'Completion record does not match this study: {tag}')
        for relative, digest in record['artifacts'].items():
            path = root / relative
            if not path.is_file() or file_hash(path) != digest:
                raise ValueError(f'Completed artifact missing/modified: {path}')
        read_metrics(metrics)
        return 'complete', None
    for name in ('last.pth', 'best.pth'):
        checkpoint = directory / name
        if checkpoint.is_file():
            return 'resume', checkpoint
    if (root / 'logs' / tag / 'train_log.csv').exists() or metrics.exists():
        raise ValueError(f'{tag} has partial output but no checkpoint. '
                         'Keep these files and use a new output root for a fresh study.')
    return 'new', None


def mark_complete(root, variant, seed, study):
    import torch

    tag = run_tag(variant, seed)
    metrics_path = root / 'results' / tag / 'metrics.csv'
    read_metrics(metrics_path)
    config_path = root / 'logs' / f'{tag}_config.json'
    config = read_json(config_path)
    actual_split = {key: [Path(name).name for name in config[key]]
                    for key in ('train', 'validation')}
    if actual_split != study['identity']['data']['split']:
        raise ValueError(f'Training split differs from the study manifest: {tag}')
    best_path = root / 'snapshots' / tag / 'best.pth'
    last_path = root / 'snapshots' / tag / 'last.pth'
    best = torch.load(best_path, map_location='cpu', weights_only=True)
    last = torch.load(last_path, map_location='cpu', weights_only=True)
    expected_architecture = f'hardgate_study.v1.{variant}'
    if any(checkpoint.get('model_architecture') != expected_architecture
           or checkpoint.get('model_name') != tag for checkpoint in (best, last)):
        raise ValueError(f'Unexpected checkpoint architecture/run: {tag}')
    if last['epoch'] != study['identity']['protocol']['epochs']:
        raise ValueError(f'Run did not finish the planned epochs: {tag}')
    paths = [metrics_path, config_path, best_path, last_path,
             root / 'results' / tag / 'efficiency.csv',
             root / 'logs' / tag / 'train_log.csv']
    record = {'variant': variant, 'seed': seed, 'tag': tag,
              'study_sha256': study['identity_sha256'],
              'best_epoch': best['epoch'], 'best_val_mDice': best['metrics']['mDice'],
              'last_epoch': last['epoch'],
              'artifacts': {path.relative_to(root).as_posix(): file_hash(path) for path in paths}}
    write_json(root / 'ablation' / tag / 'completed.json', record)


def validate_resume(root, variant, seed, checkpoint_path, study):
    """Same-shaped checkpoints from another seed must not masquerade as a resume."""
    import torch

    tag = run_tag(variant, seed)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if (checkpoint.get('model_name') != tag or checkpoint.get('model_architecture') !=
            f'hardgate_study.v1.{variant}'):
        raise ValueError(f'Resume checkpoint belongs to another run/variant: {checkpoint_path}')
    if not 1 <= checkpoint['epoch'] <= study['identity']['protocol']['epochs']:
        raise ValueError(f'Resume epoch outside the planned schedule: {checkpoint_path}')
    if not checkpoint.get('optimizer') or not checkpoint.get('training_state'):
        raise ValueError('Resume requires the full optimizer/RNG checkpoint, not only weights.')
    config = read_json(root / 'logs' / f'{tag}_config.json')
    if config['config']['model'] != MODEL_NAMES[variant] or config['config']['seed'] != seed:
        raise ValueError(f'Resume config belongs to another model/seed: {tag}')
    split = {key: [Path(name).name for name in config[key]] for key in ('train', 'validation')}
    if split != study['identity']['data']['split']:
        raise ValueError(f'Resume split differs from the study manifest: {tag}')
    return checkpoint['epoch']


def reconcile_log(root, tag, checkpoint_epoch):
    """A crash after logging but before checkpoint save must not duplicate epochs."""
    path = root / 'logs' / tag / 'train_log.csv'
    if not path.exists():
        return
    with path.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        fields, rows = reader.fieldnames, list(reader)
    kept = {}
    for row in rows:
        epoch = int(row['epoch'])
        if epoch <= checkpoint_epoch:
            kept[epoch] = row
    cleaned = [kept[epoch] for epoch in sorted(kept)]
    if rows != cleaned:
        backup = root / 'ablation' / tag / f'log_before_resume_{file_hash(path)[:12]}.csv'
        backup.parent.mkdir(parents=True, exist_ok=True)
        backup.write_bytes(path.read_bytes())
        temporary = path.with_suffix('.csv.tmp')
        with temporary.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(cleaned)
        temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variants', nargs='+', choices=list(VARIANTS), default=list(VARIANTS))
    parser.add_argument('--seeds', nargs='+', type=int, default=[42])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batchsize', type=int, default=8)
    parser.add_argument('--output_root', default='ablation_runs/hardgate')
    parser.add_argument('--data_root', default='data')
    parser.add_argument('--dry_run', action='store_true', help='Print commands; no downloads or writes')
    args = parser.parse_args()
    if args.epochs < 1 or args.batchsize < 1 or any(seed < 0 or seed >= 2**32 for seed in args.seeds):
        parser.error('Epochs/batchsize must be positive; seeds must be in [0, 2**32).')
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.variants)) != len(args.variants):
        parser.error('Seeds and variants must be unique.')
    print(f'{len(args.variants)} variants x {len(args.seeds)} seeds; '
          'fixed v2 losses, validation-selected checkpoints, five held-out test sets.', flush=True)
    if args.dry_run:
        for seed in args.seeds:
            for variant in args.variants:
                print(shlex.join(train_command(args, variant, seed)))
        return

    import torch
    import torchvision
    import numpy as np
    from utils.run_paths import output_directories
    from utils.setup_dataset import setup_dataset

    if not torch.cuda.is_available():
        raise RuntimeError('Full study requires a CUDA GPU. Use --dry_run here or run the Colab notebook.')
    log_root, _, _ = output_directories(args.output_root)  # also checks Drive is mounted
    root = log_root.parent
    args.data_root = setup_dataset(str(Path(args.data_root).resolve()))
    print('Checking dataset contents and fixed split...', flush=True)
    identity = {'protocol': {**PROTOCOL, 'epochs': args.epochs, 'batchsize': args.batchsize},
                'sources': source_fingerprint(), 'data': dataset_identity(args.data_root),
                'runtime': {'python': sys.version.split()[0], 'torch': str(torch.__version__),
                            'torchvision': str(torchvision.__version__), 'numpy': np.__version__,
                            'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
                            'num_workers': min(4, os.cpu_count() or 1)}}
    git = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO, capture_output=True, text=True)
    study = ensure_study(root, identity, {'git_revision': git.stdout.strip(),
                                        'gpu': torch.cuda.get_device_name(0)})
    # Reject damaged completed runs before launching any new training jobs.
    for seed in args.seeds:
        for variant in args.variants:
            status, resume = run_state(root, variant, seed, study['identity_sha256'])
            if resume:
                validate_resume(root, variant, seed, resume, study)
    for seed in args.seeds:
        for variant in args.variants:
            status, resume = run_state(root, variant, seed, study['identity_sha256'])
            tag = run_tag(variant, seed)
            print(f'[{status}] {tag}: {VARIANTS[variant]}', flush=True)
            if status == 'complete':
                continue
            if resume:
                epoch = validate_resume(root, variant, seed, resume, study)
                reconcile_log(root, tag, epoch)
            command = train_command(args, variant, seed, resume)
            write_json(root / 'ablation' / tag / 'launch.json',
                       {'command': command, 'study_sha256': study['identity_sha256'],
                        'gpu': torch.cuda.get_device_name(0), 'resumed_from': str(resume) if resume else None})
            subprocess.run(command, cwd=REPO, check=True)
            mark_complete(root, variant, seed, study)
    subprocess.run([sys.executable, str(REPO / 'summarize_hardgate_ablation.py'),
                    '--output_root', str(root), '--seeds', *map(str, args.seeds)],
                   cwd=REPO, check=True)


if __name__ == '__main__':
    main()
