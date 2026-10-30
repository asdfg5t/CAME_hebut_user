import pytest
from came_repro.main_pipeline import MAIN_STAGES, execute_stages


def test_main_pipeline_trains_correct_corpus_before_its_evaluation():
    visited=[]
    execute_stages(visited.append)
    assert visited==['preflight','pretrain_representation','evaluate_representation','pretrain_few_shot','evaluate_few_shot']


def test_main_pipeline_failure_prevents_all_subsequent_training():
    visited=[]
    def fail_preflight(stage):
        visited.append(stage)
        raise RuntimeError('CUDA preflight failed')
    with pytest.raises(RuntimeError,match='CUDA preflight failed'):
        execute_stages(fail_preflight)
    assert visited==['preflight']


def test_main_pipeline_downstream_failure_prevents_second_corpus_training():
    visited=[]
    def fail_evaluation(stage):
        visited.append(stage)
        if stage=='evaluate_representation':
            raise RuntimeError('Inference exceeded storage budget')
    with pytest.raises(RuntimeError,match='storage budget'):
        execute_stages(fail_evaluation)
    assert 'pretrain_few_shot' not in visited
