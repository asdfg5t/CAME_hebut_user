"""CPU-only contract tests; these never stand in for actual CUDA throughput acceptance."""
import copy
import json
import sys
from pathlib import Path

import dgl
import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from came_repro.products_sampled_stream import RowCursor,PackedLinearHeads,packed_losses,estimate_seconds,sample_blocks,canonical_blocks,forward_blocks,selected_offsets


def plan():
    return json.loads((ROOT/'configs/products_sampled_stream_20261021.json').read_text())


def test_packed_heads_gradient_and_adam_match_independent_training():
    torch.set_num_threads(1)
    seeds=[0,1,2]
    packed=PackedLinearHeads(8,3,seeds)
    separate=[]
    for i in range(3):
        h=torch.nn.Linear(8,3)
        h.load_state_dict(dict(weight=packed.weight[i*3:(i+1)*3].detach().clone(),bias=packed.bias[i*3:(i+1)*3].detach().clone()))
        separate.append(h)
    opt=torch.optim.Adam(packed.parameters(),lr=.05)
    opts=[torch.optim.Adam(h.parameters(),lr=.05) for h in separate]
    x=torch.randn(13,8)
    y=torch.arange(13)%3
    for _ in range(4):
        opt.zero_grad()
        packed_losses(packed(x),y).sum().backward()
        for i,(h,optimizer) in enumerate(zip(separate,opts)):
            optimizer.zero_grad()
            torch.nn.functional.cross_entropy(h(x),y).backward()
            torch.testing.assert_close(packed.weight.grad[i*3:(i+1)*3],h.weight.grad,rtol=1e-5,atol=2e-7)
            optimizer.step()
        opt.step()
    for i,h in enumerate(separate):
        torch.testing.assert_close(packed.weight[i*3:(i+1)*3],h.weight,rtol=1e-5,atol=5e-7)
        torch.testing.assert_close(packed.bias[i*3:(i+1)*3],h.bias,rtol=1e-5,atol=5e-7)


def test_packed_seed_initialization_preserves_rng_and_matches_individual_seeds():
    torch.manual_seed(511)
    before=torch.get_rng_state().clone()
    packed=PackedLinearHeads(8,3,[0,1,2])
    torch.testing.assert_close(torch.get_rng_state(),before,rtol=0,atol=0)
    for i in range(3):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(i)
            h=torch.nn.Linear(8,3)
        torch.testing.assert_close(packed.weight[i*3:(i+1)*3],h.weight,rtol=0,atol=0)


def test_packed_optimizer_resume_is_bitexact(tmp_path):
    x=torch.randn(9,8)
    y=torch.arange(9)%3
    def run(n,parent=None):
        h=PackedLinearHeads(8,3,[0,1])
        o=torch.optim.Adam(h.parameters(),lr=.05)
        if parent:
            s=torch.load(parent,weights_only=False)
            h.load_state_dict(s['head']);o.load_state_dict(s['optimizer'])
        for _ in range(n):
            o.zero_grad();packed_losses(h(x),y).sum().backward();o.step()
        return h,o
    a,ao=run(4)
    b,bo=run(2)
    path=tmp_path/'cpu_fixture.pt'
    torch.save(dict(head=b.state_dict(),optimizer=bo.state_dict()),path)
    c,co=run(2,path)
    for k,v in a.state_dict().items():torch.testing.assert_close(v,c.state_dict()[k],rtol=0,atol=0)
    for i,state in ao.state_dict()['state'].items():
        for k,v in state.items():torch.testing.assert_close(v,co.state_dict()['state'][i][k],rtol=0,atol=0)


@pytest.mark.parametrize('dtype',['float32','uint8'])
def test_row_cursor_resumes_only_committed_rows_and_preserves_parent(tmp_path,dtype):
    identity=dict(fixture_only=True)
    data=np.arange(18,dtype=dtype).reshape(6,3)
    parent=tmp_path/'parent.json'
    a=RowCursor(parent,identity,data.shape,dtype)
    a.commit(0,data[:2]);old=parent.read_bytes()
    b=RowCursor(tmp_path/'child.json',identity,data.shape,dtype,parent)
    assert b.cursor==2
    b.commit(2,data[2:])
    np.testing.assert_array_equal(b.complete_values(),data)
    assert parent.read_bytes()==old


def test_row_cursor_rejects_incomplete_identity_change_and_corrupted_prefix(tmp_path):
    path=tmp_path/'cursor.json'
    a=RowCursor(path,dict(source='a'),(4,2),'float32')
    a.commit(0,np.ones((2,2)))
    with pytest.raises(ValueError):a.complete_values()
    with pytest.raises(ValueError):RowCursor(tmp_path/'new.json',dict(source='b'),(4,2),'float32',path)
    values=np.load(a.values_path,mmap_mode='r+');values[0,0]=7;values.flush();values._mmap.close()
    with pytest.raises(ValueError):RowCursor(tmp_path/'new.json',dict(source='a'),(4,2),'float32',path)


def test_row_cursor_refuses_nonfinite_and_noncontiguous_rows(tmp_path):
    a=RowCursor(tmp_path/'cursor.json',{},(4,2),'float32')
    with pytest.raises(ValueError):a.commit(1,np.ones((1,2)))
    with pytest.raises(ValueError):a.commit(0,np.array([[np.nan,1]]))
    with pytest.raises(ValueError):a.commit(0,np.ones((1,3)))


def fixture_graph():
    src=np.repeat(np.arange(12),12)
    dst=np.tile(np.arange(12),12)
    return dgl.graph((src,dst),num_nodes=12).long()


def test_sampled_blocks_have_bounded_neighbors_one_loop_and_reproducible_resume():
    torch.set_num_threads(1)
    p=copy.deepcopy(plan());p['inference']['fanouts']=[3,2]
    g=fixture_graph();ids=np.array([3,4,5],dtype=np.int64)
    a,blocks=sample_blocks(g,ids,p,256,2)
    b,replay=sample_blocks(g,ids,p,256,2)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    for i,(x,y) in enumerate(zip(blocks,replay)):
        for k in ['srcdata','dstdata']:
            torch.testing.assert_close(getattr(x,k)[dgl.NID],getattr(y,k)[dgl.NID],rtol=0,atol=0)
        u,v=x.edges(order='eid');ru,rv=y.edges(order='eid')
        torch.testing.assert_close(u,ru,rtol=0,atol=0);torch.testing.assert_close(v,rv,rtol=0,atol=0)
        assert (u==v).sum()==x.num_dst_nodes()
        assert x.in_degrees().max()<=p['inference']['fanouts'][i]+1


def test_full_neighbor_block_adapter_matches_preserved_author_forward_on_CPU():
    from came_repro.author_loader import load_author_model
    from came_repro.prepared_data import PreparedData
    torch.set_num_threads(1);torch.manual_seed(713)
    model=load_author_model()(512,512,hidden_1024=16,out_1024=16,num_heads=4,num_experts=18,num_selected_experts=2)
    model.eval()
    g=dgl.graph(([0,1,1,2,3],[1,0,2,3,3]),num_nodes=6).long()
    values=np.random.default_rng(713).normal(size=(6,512)).astype(np.float32)
    data=PreparedData(g,values,None,{},dict(fixture_only=True))
    full=dgl.add_self_loop(dgl.remove_self_loop(g))
    inputs,targets,blocks=dgl.dataloading.MultiLayerFullNeighborSampler(2).sample_blocks(full,torch.arange(6))
    with torch.no_grad():
        expected=model(g,torch.from_numpy(values),torch.from_numpy(values),getz3=True,compute_loss=False)
    actual=forward_blocks(model,data,inputs,canonical_blocks(blocks),device='cpu')
    torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-5)


def test_two_hour_gate_uses_measured_rate_and_setup_margin():
    p=plan()
    assert estimate_seconds(600,p)<7200
    assert estimate_seconds(500,p)>7200
    for rate in [0,-1,float('nan')]:
        with pytest.raises(ValueError):estimate_seconds(rate,p)
    assert p['head']['seeds']==list(range(10))
    assert p['test_nodes']==2213091 and p['width']==1024 and p['precision']=='fp32'


def test_edge_offsets_are_distinct_and_exhaust_small_degrees():
    degrees=np.arange(100,dtype=np.int64)
    offsets,valid=selected_offsets(np.arange(100),degrees,10,0,217013)
    for i,degree in enumerate(degrees):
        values=offsets[i,valid[i]]
        assert len(values)==min(10,degree) and len(np.unique(values))==len(values)
        assert (values>=0).all() and (values<degree).all()
        if degree<=10:np.testing.assert_array_equal(values,np.arange(degree))


def test_sampling_choices_do_not_change_with_phase_cursor_or_grouping():
    torch.set_num_threads(1)
    p=copy.deepcopy(plan());p['inference']['fanouts']=[3,2]
    g=fixture_graph()
    _,first=sample_blocks(g,np.array([3,4,5]),p,0,1)
    _,second=sample_blocks(g,np.array([3]),p,999,2)
    def incoming(block,target):
        u,v=block.edges(order='eid')
        global_dest=block.dstdata[dgl.NID][v]
        return np.sort(block.srcdata[dgl.NID][u[global_dest==target]].numpy())
    np.testing.assert_array_equal(incoming(first[-1],3),incoming(second[-1],3))


def test_mapped_CSC_sampling_is_bitexact_to_previous_full_int64_DGL_graph(tmp_path):
    from came_repro.compact_csc import build_index,load_graph
    from came_repro.products_memory import MappedCsc
    g=fixture_graph();u,v=g.edges();edges=np.column_stack([u.numpy(),v.numpy()])
    edges=np.concatenate([edges,np.array([[0,3],[0,3],[1,4]])])
    build_index(edges,13,tmp_path,'cpu_fixture')
    old=load_graph(tmp_path).long().formats('csc');mapped=MappedCsc(tmp_path)
    assert isinstance(mapped.indices,np.memmap) and not mapped.indices.flags.writeable
    assert mapped.indices.dtype==np.int32 and mapped.num_nodes()==13
    p=copy.deepcopy(plan());p['inference']['fanouts']=[3,2]
    for targets in [np.array([3,4,12]),np.array([0]),np.array([1,2,3,4,5])]:
        a,left=sample_blocks(old,targets,p,0,1);b,right=sample_blocks(mapped,targets,p,900,3)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        for x,y in zip(left,right):
            assert x.idtype==y.idtype==torch.int64
            for key in ['srcdata','dstdata']:torch.testing.assert_close(getattr(x,key)[dgl.NID],getattr(y,key)[dgl.NID],rtol=0,atol=0)
            for xe,ye in zip(x.edges(order='eid'),y.edges(order='eid')):torch.testing.assert_close(xe,ye,rtol=0,atol=0)


@pytest.mark.parametrize('reason',['available','private'])
def test_RAM_guard_keeps_limits_and_records_actual_pause_reason(tmp_path,monkeypatch,reason):
    from came_repro import products_sampled_stream as core
    from came_repro.pause_control import TrainingPaused
    snapshot=dict(total_RAM_bytes=16*2**30,available_RAM_bytes=2*2**30,process_private_bytes=2*2**30,process_RSS_bytes=2*2**30)
    snapshot['available_RAM_bytes' if reason=='available' else 'process_private_bytes']=int(.5*2**30) if reason=='available' else 14*2**30
    monkeypatch.setattr(core,'OUT',tmp_path);monkeypatch.setattr(core,'memory_snapshot',lambda:snapshot)
    with pytest.raises(TrainingPaused,match='available='):core.resource_guard(plan())
    saved=json.loads((tmp_path/'memory_boundary.json').read_text())
    assert saved['available_RAM_bytes']==snapshot['available_RAM_bytes'] and saved['committed_work_preserved']
    assert saved['thresholds']['minimum_free_RAM_bytes']==2**30


def test_RAM_guard_allows_healthy_snapshot_without_writing_pause(tmp_path,monkeypatch):
    from came_repro import products_sampled_stream as core
    monkeypatch.setattr(core,'OUT',tmp_path)
    monkeypatch.setattr(core,'memory_snapshot',lambda:dict(total_RAM_bytes=16*2**30,available_RAM_bytes=2*2**30,process_private_bytes=2*2**30))
    core.resource_guard(plan());assert not (tmp_path/'memory_boundary.json').exists()


def test_controller_safe_pause_has_clean_exit_and_paused_record(tmp_path,monkeypatch):
    import run_24_products_sampled_stream as entry
    from came_repro import ide_workflow,quality_pipeline,storage_budget,products_controller_storage
    root=tmp_path/'fixture';root.mkdir();release=root/'release.json';release.write_text('{}')
    for key,value in dict(ROOT=root,OUT=root/'out',LEDGER=root/'ledger.json',PAUSE=root/'pause.json',RELEASE=release).items():monkeypatch.setattr(entry,key,value)
    monkeypatch.setattr(entry,'verify',plan);monkeypatch.setattr(sys,'argv',['entry'])
    class Lock:
        def close(self):pass
    monkeypatch.setattr(ide_workflow,'acquire_gpu_lock',lambda *args:Lock())
    monkeypatch.setattr(storage_budget,'check_budget',lambda **kwargs:{})
    monkeypatch.setattr(products_controller_storage,'reserve_external',lambda *args:None)
    def paused(arguments,pause,ledger,ledger_path,record,logger,remaining):
        run=root/'run';run.mkdir();(run/'status.json').write_text('{"status":"paused","error":"RAM fixture"}')
        ledger['active_seconds']+=3
        record.setdefault('attempts',[]).append(dict(run_directory=str(run),returncode=75,elapsed_active_seconds=3))
        return 75,str(run)
    monkeypatch.setattr(quality_pipeline,'run_child',paused)
    assert entry.main() is None
    state=entry.read(entry.LEDGER)
    assert state['state']=='paused' and state['records']['calibration']['state']=='paused'
    assert state['active_seconds']==3 and 'RAM fixture' in state['error']


def test_controller_memory_and_storage_helpers_do_not_import_Torch_or_DGL():
    import subprocess
    result=subprocess.run([sys.executable,'-c',
        "import sys; import came_repro.products_controller_storage; import came_repro.products_memory; assert 'torch' not in sys.modules; assert 'dgl' not in sys.modules"],
        cwd=ROOT,capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr


def test_bounded_feature_reads_are_bitexact_and_do_not_expose_mmap_views(tmp_path):
    from came_repro.products_memory import ReopenedNpyFeatures
    from came_repro.prepared_data import PreparedData
    path=tmp_path/'features.npy';values=np.arange(17*512,dtype=np.float32).reshape(17,512)
    np.save(path,values);old=path.read_bytes();source=ReopenedNpyFeatures(path)
    for index in [np.array([16,0,4,4]),slice(2,8),3]:
        result=source[index];np.testing.assert_array_equal(result,values[index]);result[...] = -1
        assert path.read_bytes()==old
    data=PreparedData(None,source,None,{},dict(fixture=True))
    a,b=data.modality_batch(torch.tensor([1,5,16]))
    torch.testing.assert_close(a,torch.from_numpy(values[[1,5,16]]),rtol=0,atol=0)
    torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_experiment_budget_worker_is_restored_even_when_initialization_fails(monkeypatch):
    from came_repro import products_sampled_stream as core,experiment,storage_budget,products_controller_storage
    original=storage_budget.check_budget
    def fake_init(self,*args,**kwargs):
        assert storage_budget.check_budget is products_controller_storage.external_budget_report
        raise RuntimeError('fixture initialization failure')
    monkeypatch.setattr(experiment.Experiment,'__init__',fake_init)
    with pytest.raises(RuntimeError,match='fixture'):core.ProductsExperiment({}, {})
    assert storage_budget.check_budget is original


def test_exact_FP32_feature_cache_hits_misses_duplicates_and_order_match_source_on_CPU(tmp_path):
    from came_repro.products_memory import ReopenedNpyFeatures,ExactFeatureCache
    path=tmp_path/'clip.npy';matrix=np.arange(11*512,dtype=np.float32).reshape(11,512);np.save(path,matrix)
    cache=ExactFeatureCache(ReopenedNpyFeatures(path),np.array([0,3,8]),device='cpu')
    ids=torch.tensor([8,1,3,1,10,0]);a,b=cache.modality_batch(ids)
    torch.testing.assert_close(a,torch.from_numpy(matrix[ids.numpy()]),rtol=0,atol=0)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert cache.stats()['cache_hits']==3 and cache.stats()['rows_requested']==6
    assert cache.values.dtype==torch.float32 and cache.values.shape==(3,512)


def test_structural_cache_priority_is_label_free_with_deterministic_node_ID_ties(tmp_path):
    from came_repro.products_memory import MappedCsc,cache_node_ids
    from came_repro.compact_csc import build_index
    edges=np.array([[3,0],[3,1],[1,0],[2,0]],dtype=np.int64)
    build_index(edges,5,tmp_path,'fixture')
    graph=MappedCsc(tmp_path)
    np.testing.assert_array_equal(cache_node_ids(graph,3),[1,2,3])
    np.testing.assert_array_equal(cache_node_ids(graph,5),np.arange(5))
    with pytest.raises(ValueError):cache_node_ids(graph,0)
