import importlib.util
from pathlib import Path
import pytest

path=Path(__file__).resolve().parents[1]/'scripts/run_products_single_validation.py'
spec=importlib.util.spec_from_file_location('products_single_validation',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_inference_commit_must_include_next_contiguous_rows():
    assert module.next_cursor([40,41,42],40,100)==43


@pytest.mark.parametrize('targets,cursor,n',[([41,42],40,100),([40,42],40,100),([],40,100),([99,100],99,100)])
def test_inference_never_skips_or_commits_unexpected_targets(targets,cursor,n):
    with pytest.raises(ValueError):module.next_cursor(targets,cursor,n)


def io_fixture(monkeypatch):
    import types,logging,torch,dgl
    import came_repro.stream_inference as stream
    import came_repro.storage_budget as budget
    class Graph:
        def num_nodes(self):return 4
        def formats(self,_):return self
    class Model:
        def _prep_graph(self,g):return g
        def eval(self):return self
        def parameters(self):return []
    class Sampler:
        def __init__(self,_):pass
        def sample_blocks(self,g,targets):return targets,targets,[]
    # CPU-only filesystem fixture; production CUDA computations are not replaced.
    monkeypatch.setattr(dgl.dataloading,'MultiLayerFullNeighborSampler',Sampler)
    monkeypatch.setattr(stream,'batch_full_neighbor_embeddings',lambda m,d,b,ids:ids.float()[:,None].repeat(1,1024))
    monkeypatch.setattr(budget,'guard_allocation',lambda *a:None)
    return Model(),types.SimpleNamespace(graph=Graph(),audit={'fixture_only':True}),types.SimpleNamespace(logger=logging.getLogger('io_fixture'))


def test_numpy_prefix_pause_resume_preserves_all_rows(tmp_path,monkeypatch):
    import numpy as np
    import came_repro.pause_control as pause
    model,data,experiment=io_fixture(monkeypatch);count=[0]
    def stop_after_one_commit():
        count[0]+=1
        if count[0]==2:raise pause.TrainingPaused('filesystem fixture pause')
    monkeypatch.setattr(pause,'pause_if_requested',stop_after_one_commit)
    with pytest.raises(pause.TrainingPaused):module.exact_inference(model,data,tmp_path,'fixture',experiment,2)
    monkeypatch.setattr(pause,'pause_if_requested',lambda:None)
    rows=module.exact_inference(model,data,tmp_path,'fixture',experiment,2)
    np.testing.assert_array_equal(rows,np.arange(4,dtype=np.float32)[:,None].repeat(1024,axis=1))


def test_corrupt_saved_prefix_is_rejected_before_resume(tmp_path,monkeypatch):
    import numpy as np
    import came_repro.pause_control as pause
    model,data,experiment=io_fixture(monkeypatch);count=[0]
    def stop():
        count[0]+=1
        if count[0]==2:raise pause.TrainingPaused('filesystem fixture pause')
    monkeypatch.setattr(pause,'pause_if_requested',stop)
    with pytest.raises(pause.TrainingPaused):module.exact_inference(model,data,tmp_path,'fixture',experiment,2)
    array=np.load(tmp_path/'fused_embeddings.npy',mmap_mode='r+');array[0,0]=99;array.flush();array._mmap.close()
    monkeypatch.setattr(pause,'pause_if_requested',lambda:None)
    with pytest.raises(ValueError,match='prefix checksum'):module.exact_inference(model,data,tmp_path,'fixture',experiment,2)
