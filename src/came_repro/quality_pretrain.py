"""One disclosed continuation round; preserve model, Adam, RNG and fixed seed pools."""
import argparse
import copy
import sys
from pathlib import Path

import torch
from . import joint_train, subset_offline_train
from .quality_protocol import extension_signature


class ContinuationSchedule(subset_offline_train.FixedSubsetSchedule):
    def load_state_dict(self, state):
        super().load_state_dict(state)
        # The eight-round trainer stopped after advancing epoch to 8 and did
        # not shuffle a ninth order. Prepare that order exactly once using the
        # restored generator; an in-progress ninth round keeps its cursor.
        if self.epoch == 8 and all(self.cursors[name] == len(order) for name, order in self.orders.items()):
            self.new_epoch()


class ContinuationLogFilter(subset_offline_train.SubsetLogFilter):
    def filter(self, record):
        super().filter(record)
        if isinstance(record.msg,str):
            record.msg=record.msg.replace('eight configured subset rounds;',
                                         'nine configured subset rounds (eight completed parent rounds plus one extension);')
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--resume', type=Path, required=True)
    args = parser.parse_args()
    original_validate, original_signature = joint_train.validate_joint, joint_train.signature
    original_schedule = subset_offline_train.FixedSubsetSchedule
    original_filter = subset_offline_train.SubsetLogFilter
    original_args = sys.argv

    def validate_extension(config):
        validated = copy.deepcopy(config)
        if config['training']['epochs'] != 9:
            raise ValueError('Only the explicitly declared eight-to-nine subset-round extension is allowed')
        validated['training']['epochs'] = 8
        original_validate(validated)
        parent = torch.load(args.resume, map_location='cpu', weights_only=False)
        if parent.get('completed_subset_rounds', 0) not in (8, 9) or parent['schedule']['epoch'] not in (8, 9):
            raise ValueError('A completed eight-round parent or its extension resume is required')
        if extension_signature(parent['config']) != extension_signature(config):
            raise ValueError('Continuation changes model, pools, optimizer, data protocol or storage')
        if parent['config']['training']['epochs'] not in (8, 9):
            raise ValueError('Unexpected parent round budget')

    joint_train.validate_joint = validate_extension
    joint_train.signature = extension_signature
    subset_offline_train.FixedSubsetSchedule = ContinuationSchedule
    subset_offline_train.SubsetLogFilter = ContinuationLogFilter
    sys.argv = [original_args[0], '--config', str(args.config), '--resume', str(args.resume)]
    try:
        subset_offline_train.main()
    finally:
        joint_train.validate_joint, joint_train.signature = original_validate, original_signature
        subset_offline_train.FixedSubsetSchedule = original_schedule
        subset_offline_train.SubsetLogFilter = original_filter
        sys.argv = original_args


if __name__ == '__main__':
    main()
