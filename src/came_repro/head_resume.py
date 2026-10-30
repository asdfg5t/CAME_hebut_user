"""Identity checks for epoch-boundary downstream resumes into a new run."""
import copy
import torch
from .experiment import restore_rng


def head_signature(config):
    value=copy.deepcopy(config)
    for key in ('resume_head','epochs'):
        value.get('evaluation',{}).pop(key,None)
    return value


def restore_head(path,schema,experiment,model,optimizer,generator):
    state=torch.load(path,map_location='cpu',weights_only=False)
    if (state['schema']!=schema or head_signature(state['config'])!=head_signature(experiment.config)
            or state['data_audit']!=experiment.data_audit):
        raise ValueError('Downstream resume changed model/data/protocol')
    model.load_state_dict(state['classifier' if schema=='came_linear_probe_v1' else 'head'])
    optimizer.load_state_dict(state['optimizer'])
    restore_rng(state['rng'])
    generator.bit_generator.state=state['sampling_generator']
    for row in state['history_rows']:
        experiment.metric(row)
    experiment.logger.info('Restored completed epoch=%d; next epoch=%d',state['completed_epochs'],state['next_epoch'])
    return state
