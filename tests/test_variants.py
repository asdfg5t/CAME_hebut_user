from types import SimpleNamespace
import dgl
import pytest
import torch
from came_repro.author_loader import load_author_model,verify_originals
from came_repro.variants import model_class,adapt_instance,expected_loss_weights
from came_repro.stream_inference import batch_full_neighbor_embeddings


def build(variant):
    author=load_author_model()
    model=model_class(author,variant)(8,8,hidden_1024=16,out_1024=16,num_experts=18,dropout=0,attn_drop=0)
    return adapt_instance(model,author,variant)


def test_loss_ablations_retain_surviving_main_weights():
    base={'w_tv':.5,'w_tf':.1,'w_vf':.5}
    assert expected_loss_weights(base,'without_text_fusion_loss')=={'w_tv':.5,'w_tf':0.,'w_vf':.5}
    assert expected_loss_weights(base,'only_vision_fusion_loss')=={'w_tv':0.,'w_tf':0.,'w_vf':.5}
    assert base=={'w_tv':.5,'w_tf':.1,'w_vf':.5}


def test_no_dimension_gate_is_exact_mean_after_modal_encoders():
    model=build('without_dimension_gate').eval()
    graph=dgl.graph(([0,1,2],[1,2,0]),num_nodes=3)
    observed={}
    handle=model.fuse_mlp.register_forward_pre_hook(lambda module,args:observed.update(value=args[0].detach()))
    with torch.no_grad():
        _,result=model(graph,torch.randn(3,8),torch.randn(3,8),return_embeddings=True,compute_loss=False)
    handle.remove()
    torch.testing.assert_close(observed['value'][:,:16],.5*(result['z1']+result['z2']))


def test_parameter_matched_mlp_uses_real_parameters_without_padding():
    model=build('without_moe')
    report=model.parameter_match_audit
    assert report['relative_difference']<.01
    graph=dgl.graph(([0,1,2],[1,2,0]),num_nodes=3)
    loss=model(graph,torch.randn(3,8),torch.randn(3,8))
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.fuse_mlp.parameters())
    verify_originals()


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_early_mean_ablation_has_one_gnn_and_correct_streamed_inference():
    model=build('without_modality_gnn').cuda().eval()
    assert model.gat_clip is model.gat_t5vit
    graph=dgl.graph(([0,1,2,0],[1,2,0,3]),num_nodes=4)
    text,vision=torch.randn(4,8),torch.randn(4,8)
    data=SimpleNamespace(modality_batch=lambda ids:(text[ids],vision[ids]))
    with torch.no_grad():
        _,expected=model(graph.to('cuda'),text.cuda(),vision.cuda(),return_embeddings=True,compute_loss=False)
    torch.testing.assert_close(expected['z1'],expected['z2'])
    graph=model._prep_graph(graph)
    actual=torch.empty(4,16)
    loader=dgl.dataloading.DataLoader(graph,torch.arange(4),dgl.dataloading.MultiLayerFullNeighborSampler(2),batch_size=2,shuffle=False,num_workers=0)
    for inputs,targets,blocks in loader:
        actual[targets]=batch_full_neighbor_embeddings(model,data,blocks,inputs).cpu()
    torch.testing.assert_close(actual,expected['z3'].cpu(),rtol=1e-5,atol=1e-6)
