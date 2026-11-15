"""Report completed, matched v2 ablations; never treat absent runs as zeros."""

import argparse
import csv
from pathlib import Path
from statistics import mean, stdev

from experiments.ablation_io import read_json, read_metrics, write_csv
from experiments.pid_v2_ablation import LOWER_IS_BETTER, METRICS, TEST_COUNTS, VARIANTS, run_tag
from run_pid_v2_ablation import run_state


def summarize(root, seeds):
    root = Path(root)
    study = read_json(root / 'ablation_study.json')
    runs, missing, details, efficiency = {}, [], [], []
    for variant in VARIANTS:
        runs[variant] = {}
        for seed in seeds:
            tag = run_tag(variant, seed)
            done = root / 'ablation' / tag / 'completed.json'
            if not done.exists():
                missing.append(tag)
                continue
            run_state(root, variant, seed, study['identity_sha256'])
            record = read_json(done)
            runs[variant][seed] = read_metrics(root / 'results' / tag / 'metrics.csv')
            details.append({'variant': variant, 'seed': seed, 'best_epoch': record['best_epoch'],
                            'best_val_mDice': record['best_val_mDice'],
                            'last_epoch': record['last_epoch']})
            with (root / 'results' / tag / 'efficiency.csv').open(newline='', encoding='utf-8-sig') as handle:
                for row in csv.DictReader(handle):
                    efficiency.append({'variant': variant, 'seed': seed, **row})

    rows, lookup = [], {}
    for variant in VARIANTS:
        available = [seed for seed in seeds if seed in runs[variant]]
        paired = len(available) == len(seeds) and all(seed in runs['full'] for seed in seeds)
        for dataset in [*TEST_COUNTS, 'Mean']:
            for metric in METRICS:
                values = [runs[variant][seed][dataset][metric] for seed in available]
                delta = [runs[variant][seed][dataset][metric] - runs['full'][seed][dataset][metric]
                         for seed in seeds] if paired else []
                row = {'variant': variant, 'dataset': dataset, 'metric': metric,
                       'direction': 'lower' if metric in LOWER_IS_BETTER else 'higher',
                       'n': len(values), 'expected_n': len(seeds),
                       'seeds': ','.join(map(str, available)),
                       'mean': mean(values) if values else '',
                       'std': stdev(values) if len(values) > 1 else '',
                       'paired_delta_vs_full': mean(delta) if delta else '',
                       'paired_delta_std': stdev(delta) if len(delta) > 1 else ''}
                rows.append(row)
                lookup[variant, dataset, metric] = row

    report = root / 'ablation_report'
    report.mkdir(parents=True, exist_ok=True)
    write_csv(report / 'summary.csv', rows, list(rows[0]))
    write_csv(report / 'runs.csv', details,
              ['variant', 'seed', 'best_epoch', 'best_val_mDice', 'last_epoch'])
    if efficiency:
        fields = list(dict.fromkeys(key for row in efficiency for key in row))
        write_csv(report / 'efficiency_runs.csv', efficiency, fields)

    def cell(variant, dataset, metric, latex=False):
        row = lookup[variant, dataset, metric]
        if not row['n']:
            return '--'
        value = f"{row['mean']:.4f}"
        if row['std'] != '':
            separator = r' $\pm$ ' if latex else ' ± '
            value += separator + f"{row['std']:.4f}"
        return value

    lines = ['# PID-UNet v2 ablation study', '',
             f"Requested seeds: {seeds}. Study: `{study['identity_sha256']}`.", '',
             'Mean ± sample standard deviation across seeds (ddof=1). '
             'One seed has no estimated standard deviation. Values use the 0–1 scale except HD95.', '',
             'Checkpoint selected by internal validation mDice. Test metrics do not select epochs. '
             'Mean is the unweighted mean of the five dataset scores within each seed.', '',
             '| Variant | n / planned | ' + ' | '.join([*TEST_COUNTS, 'Mean']) + ' |',
             '|---|---|' + '---|' * 6]
    for variant in VARIANTS:
        lines.append(f'| {variant} | {len(runs[variant])}/{len(seeds)} | ' +
                     ' | '.join(cell(variant, dataset, 'mDice') for dataset in [*TEST_COUNTS, 'Mean']) + ' |')
    lines += ['', '## CVC-300 details', '',
              '| Variant | Dice ↑ | IoU ↑ | Precision ↑ | Recall ↑ | HD95 ↓ | MAE ↓ | Δ Dice vs full |',
              '|---|---|---|---|---|---|---|---|']
    for variant in VARIANTS:
        delta = lookup[variant, 'CVC-300', 'mDice']['paired_delta_vs_full']
        delta_text = f'{delta:+.4f}' if delta != '' else '--'
        lines.append(f'| {variant} | ' + ' | '.join(cell(variant, 'CVC-300', metric) for metric in
                     ('mDice', 'mIoU', 'Precision', 'Recall', 'HD95', 'MAE')) + f' | {delta_text} |')
    lines += ['', 'Deltas are variant minus full, computed only when BOTH rows have ALL requested '
              'seeds. A negative Dice delta means the ablation performed worse. All ten metrics '
              'and paired delta standard deviations are in summary.csv. No significance claim '
              'is inferred from three seeds.', '',
              'Efficiency is retained per run in efficiency_runs.csv; compare latency only on '
              'the same hardware/settings. Existing profiler FLOPs (G) uses its THOP convention. '
              'Checkpoint file size includes training state, not only deployment weights.', '',
              '## Missing or unfinished runs', '']
    lines += [f'- `{tag}`' for tag in missing] if missing else ['None.']
    (report / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

    latex = [r'\begin{tabular}{lrrrrrrr}', r'\hline',
             r'Variant & $n$ & CVC-300 & ClinicDB & ColonDB & ETIS & Kvasir & Mean \\', r'\hline']
    for variant in VARIANTS:
        name = variant.replace('_', r'\_')
        latex.append(f'{name} & {len(runs[variant])}/{len(seeds)} & ' + ' & '.join(
            cell(variant, dataset, 'mDice', latex=True) for dataset in [*TEST_COUNTS, 'Mean']) + r' \\')
    latex += [r'\hline', r'\end{tabular}',
              '% Dice: mean +/- sample SD across requested seeds; n=1 has no SD estimate.',
              '% Partial rows are preliminary; do not claim matched-seed superiority.']
    (report / 'dice_table.tex').write_text('\n'.join(latex) + '\n', encoding='utf-8')
    return report, len(details), missing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_root', default='ablation_runs/pid_unet_v2')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 3407, 1234])
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error('Seeds must be unique.')
    report, completed, missing = summarize(args.output_root, args.seeds)
    print(f'Report: {report}\nCompleted: {completed}; missing/unfinished: {len(missing)}')


if __name__ == '__main__':
    main()
