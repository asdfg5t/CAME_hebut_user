"""CPU-only independent full ten-seed audit; never changes historical run artifacts."""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))


def main():
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from came_repro.atomic_io import read_json_file, atomic_write_json
    from came_repro.author_loader import fingerprint, verify_originals
    from audit_two_lp_confirmation import audit_confirmation
    verify_originals()
    ledger_path = ROOT / 'artifacts/workflow_state/verified_job07_sports_delivery_repeats_current.json'
    ledger = read_json_file(ledger_path)
    if ledger['state'] != 'completed_delivered_two_lp_repeats':
        raise ValueError('Ten Sports seeds must be complete')
    for name, digest in ledger['identity'].items():
        if fingerprint(name) != digest:
            raise ValueError('Sports frozen source changed')
    delivery = read_json_file(ROOT / 'artifacts/remaining_main_pilots/two_lp_priority/verified_job07_sports_delivery.json')
    first = delivery['tasks']['sports-copurchase']
    rows, audits, curves = [], [], []
    for seed in range(10):
        if seed == 0:
            run = Path(first['confirmation_run'])
            lock = ROOT / 'artifacts/remaining_main_pilots/hard_lp/sports-copurchase_selection_lock.json'
        else:
            rec = ledger['records'][f'sports-copurchase_confirmation_s{seed}']
            assert rec['state'] == 'completed'
            for name, digest in rec['artifact_sha256'].items():
                if fingerprint(name) != digest: raise ValueError('Sports saved artifact changed')
            run = Path(rec['run_directory'])
            lock = ROOT / f'artifacts/verified_job07_sports_delivery_repeats/sports-copurchase_s{seed}_selection_lock.json'
        audit = audit_confirmation(run, lock)
        audit['seed'] = seed
        audits.append(audit)
        row = {'seed': seed, 'test_mrr_pct': audit['test_mrr_pct'],
               'gap_pp': audit['gap_pp'], 'within_two_pp': audit['within_two_pp'],
               'selected_epoch': audit['selected_epoch'], 'completed_epochs': audit['full_epoch_count'],
               'confirmation_run': str(run)}
        rows.append(row)
        selected = Path(audit['validation_run'])
        curves.append([json.loads(line) for line in (selected / 'logs/metrics.jsonl').read_text().splitlines()])
    values = np.array([r['test_mrr_pct'] for r in rows])
    expected = ledger['statistics']['sports-copurchase']
    np.testing.assert_allclose(values, np.array(expected['scores_pct']), rtol=0, atol=2e-12)
    mean, sd = float(values.mean()), float(values.std(ddof=1))
    np.testing.assert_allclose([mean, sd], [expected['mean_mrr_pct'], expected['sample_sd_pp']], rtol=0, atol=2e-12)
    directory = ROOT / 'artifacts/final_results/sports_ten_seeds'
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'results.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(range(10), values, 'o-', label='All ten fixed seeds')
    axes[0].axhline(35.68, color='orange', ls='--', label='Paper minus 2 pp')
    axes[0].axhline(mean, color='navy', ls=':', label='Mean')
    axes[0].set(xlabel='Downstream seed', ylabel='Official test MRR (%)', xticks=range(10))
    axes[0].legend(fontsize=8)
    axes[1].errorbar([0], [mean], yerr=[sd], fmt='o', capsize=5, label='Reconstruction mean +/- sample SD')
    axes[1].scatter([1], [37.68], marker='D', color='black', label='Paper reported mean')
    axes[1].set(xticks=[0, 1], xticklabels=['Reconstruction', 'Paper'], ylabel='Official test MRR (%)', ylim=(35.3, 38.0))
    axes[1].legend(fontsize=8)
    fig.suptitle('Sports: full official test, 150 fixed negatives; one frozen encoder')
    fig.text(.03, .02, f'Mean {mean:.6f}%, SD {sd:.6f} pp; gap {mean-37.68:+.6f} pp. All seeds retained; engineered LP training.', fontsize=8)
    fig.tight_layout(rect=(0, .07, 1, .93))
    fig.savefig(directory / 'paper_comparison.png', dpi=180); plt.close(fig)
    fig, axes = plt.subplots(5, 2, figsize=(12, 14))
    for seed, axis in enumerate(axes.flat):
        history = curves[seed]
        axis.plot([r['step'] for r in history], [100*r['validation_mrr'] for r in history])
        axis.axvline(rows[seed]['selected_epoch'], ls='--', color='black')
        axis.set(title=f'Seed {seed}: test {values[seed]:.4f}%', xlabel='Complete train-positive epoch', ylabel='Validation MRR (%)')
    fig.suptitle('All ten validation histories; saved-head selection shown by dashed line')
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(directory / 'validation_histories.png', dpi=130); plt.close(fig)
    report = {'dataset': 'sports-copurchase', 'complete_seeds': 10, 'fixed_seed_ids': list(range(10)),
              'mean_test_mrr_pct': mean, 'sample_sd_pp': sd, 'paper_mrr_pct': 37.68,
              'gap_pp': mean-37.68, 'mean_within_two_pp': abs(mean-37.68) <= 2,
              'individual_seeds_within_two_pp': int(sum(r['within_two_pp'] for r in rows)),
              'individual_outside_target_seeds': [r['seed'] for r in rows if not r['within_two_pp']],
              'no_seeds_discarded': True, 'official_test_queries_per_seed': 37413,
              'official_fixed_negatives': 150, 'all_full_validation_test_ranks_recomputed': True,
              'all_CSV_JSONL_TensorBoard_checkpoint_parent_hashes_verified': True,
              'encoder_independent_repeats': 1, 'paper_reproduction_complete': False,
              'CPU_only_no_GPU_scoring_or_retraining': True, 'source_ledger_sha256': fingerprint(ledger_path),
              'per_seed_results': rows, 'per_seed_audits': audits,
              'figures': {str(p): fingerprint(p) for p in directory.glob('*.png')},
              'aggregate_figures_visually_verified': False}
    atomic_write_json(directory / 'verified_results.json', report)
    print(json.dumps({k: report[k] for k in ('complete_seeds', 'mean_test_mrr_pct', 'sample_sd_pp', 'gap_pp', 'individual_outside_target_seeds')}))


if __name__ == '__main__': main()
