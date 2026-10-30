"""I/O-only optimizations; no arithmetic, optimizer or evaluation-set changes."""
import time
from pathlib import Path

import numpy as np
import torch

from .author_loader import fingerprint
from .atomic_io import read_json_file


class BufferedTensorBoardWriter:
    """Keep every scalar; avoid waiting for the event queue on every metric row."""
    def __init__(self,writer,interval=5.0,clock=time.monotonic):
        if interval<=0:raise ValueError('Positive flush interval required')
        self.writer,self.interval,self.clock=writer,interval,clock
        self.last_flush=clock()

    def __getattr__(self,name):return getattr(self.writer,name)

    def flush(self,force=False):
        now=self.clock()
        if force or now-self.last_flush>=self.interval:
            self.writer.flush();self.last_flush=now

    def close(self):
        try:self.flush(force=True)
        finally:self.writer.close()


def verified_validation_replay(lock,valid,data_audit):
    """Reuse core.train_rounds' complete selected-head replay, with immutable proof."""
    if not lock.get('selected_head_validation_replayed') or lock.get('data_audit')!=data_audit:
        raise ValueError('Matching complete selected-head replay provenance required')
    run=Path(lock['validation_run_directory'])
    paths={Path(lock['classifier_checkpoint']):lock['classifier_sha256'],
        run/'validation_summary.json':lock['validation_summary_sha256'],
        run/'validation_link_ranks.npz':lock['validation_ranks_sha256']}
    if not lock['locked_before_test'] or lock['test_used_for_selection']:
        raise ValueError('Validation-only selection lock required')
    for path,digest in paths.items():
        if fingerprint(path)!=digest:raise ValueError('Validation replay artifact changed: '+str(path))
    if read_json_file(run/'status.json')['status']!='completed_matched_lp_validation':
        raise ValueError('Incomplete validation phase cannot skip GPU replay')
    summary=read_json_file(run/'validation_summary.json')
    selected=torch.load(lock['classifier_checkpoint'],map_location='cpu',weights_only=False)
    if selected['selected_round']!=summary['selected_round'] or selected['validation_mrr']!=summary['validation_mrr']:
        raise ValueError('Saved head selection differs from complete validation')
    with np.load(run/'validation_link_ranks.npz') as values:
        for key in ['source_node','target_node']:np.testing.assert_array_equal(values[key],np.asarray(valid[key]))
        ranks=np.asarray(values['rank'],dtype=np.float64)
        if (ranks.shape!=(len(valid['source_node']),) or not np.isfinite(ranks).all()
                or not ((ranks>=1)&(ranks<=151)).all()):raise ValueError('Complete finite official validation ranks required')
        if float(np.mean(1./ranks))!=summary['validation_mrr']:raise ValueError('Independent selected-head validation MRR mismatch')
    return summary,selected
