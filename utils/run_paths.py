"""Resolve persistent output locations before downloading data or training."""

from pathlib import Path


def output_directories(output_root='.'):
    requested = Path(output_root).expanduser()
    normalized = requested.as_posix()
    if normalized == '/content/drive' or normalized.startswith('/content/drive/'):
        if not Path('/content/drive/MyDrive').is_dir():
            raise ValueError("Mount Google Drive first: drive.mount('/content/drive')")
    root = requested.resolve()
    directories = tuple(root / name for name in ('logs', 'snapshots', 'results'))
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    return directories


def check_run_destination(log_root, snapshot_root, tag, resume=None):
    if resume:
        if not Path(resume).is_file():
            raise FileNotFoundError(f'Resume checkpoint not found: {resume}')
        return
    occupied = [Path(log_root) / tag / 'train_log.csv',
                Path(snapshot_root) / tag / 'best.pth',
                Path(snapshot_root) / tag / 'last.pth']
    if any(path.exists() for path in occupied):
        raise FileExistsError(
            f'Run {tag!r} already contains training output. '
            'Use a different --run_name for a fresh run, or --resume to continue it.'
        )
