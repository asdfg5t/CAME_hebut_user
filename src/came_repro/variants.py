"""Explicit paper ablations on owned model instances; preserved source files are untouched."""
import sys

import torch
from torch import nn

VARIANTS=('came','without_text_vision_loss','without_text_fusion_loss','without_vision_fusion_loss',
          'only_text_vision_loss','only_text_fusion_loss','only_vision_fusion_loss',
          'without_dimension_gate','without_moe','without_modality_gnn')


class HalfGate(nn.Module):
    def forward(self,value):
        return value.new_full((len(value),value.shape[1]//2),.5)


def model_class(author_class,variant):
    if variant!='without_modality_gnn':
        return author_class
    nce=sys.modules[author_class.__module__].info_nce_loss

    class EarlyMeanGraph(author_class):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.gat_clip=self.gat_t5vit

        def forward(self,graph,feature_t5vit,feature_clip,return_embeddings=False,compute_loss=True,**kwargs):
            if kwargs.get('cat_emb'):
                raise ValueError('Layer concatenation is not part of this ablation reconstruction')
            graph=self._prep_graph(graph)
            dtype=next(self.parameters()).dtype
            shared=self.gat_t5vit(graph,.5*(feature_t5vit.to(dtype=dtype)+feature_clip.to(dtype=dtype)))
            # Both expert views receive the output of the ONE shared early-fusion GNN.
            fused=self.fuse_mlp(torch.cat([shared,shared,shared],dim=-1))
            if kwargs.get('getz1') or kwargs.get('getz2'):
                return shared
            if kwargs.get('getz3'):
                return fused
            loss=None
            outputs={'z1':shared,'z2':shared,'z3':fused}
            if compute_loss:
                params={'temperature':self.temperature,'symmetric':self.symmetric_nce,'batch_size':self.loss_batch_size}
                l12=nce(shared,shared,**params)
                l13=nce(shared,fused,**params)
                l23=nce(shared,fused,**params)
                loss=self.w12*l12+self.w13*l13+self.w23*l23
                outputs.update(loss12=l12,loss13=l13,loss23=l23)
            return (loss,outputs) if return_embeddings else loss

    return EarlyMeanGraph


def adapt_instance(model,author_class,variant):
    if variant not in VARIANTS:
        raise ValueError('Unknown paper ablation')
    model.representation_variant=variant
    if variant=='without_dimension_gate':
        model.fuse_gate=HalfGate()
    elif variant=='without_moe':
        target=sum(parameter.numel() for parameter in model.fuse_mlp.parameters())
        hidden=model.gat_t5vit.norms[-1].normalized_shape[0]
        # Author's two-layer MLP: (3d+3+d)*width+d parameters. No inert padding parameters.
        coefficient=4*hidden+3
        width=max(1,round((target-hidden)/coefficient))
        mlp_class=sys.modules[author_class.__module__].MLP
        model.fuse_mlp=mlp_class(3*hidden,width,hidden,dropout=model.gat_t5vit.dropout.p,num_layers=2)
        achieved=sum(parameter.numel() for parameter in model.fuse_mlp.parameters())
        model.parameter_match_audit={'target_fusion_parameters':target,'actual_mlp_parameters':achieved,
                                    'relative_difference':abs(achieved-target)/target,'hidden_width':width,
                                    'policy':'Nearest functional MLP width; no dummy unused parameters added.'}
    return model


def expected_loss_weights(base,variant):
    result=dict(base)
    removed={'without_text_vision_loss':'w_tv','without_text_fusion_loss':'w_tf','without_vision_fusion_loss':'w_vf'}
    if variant in removed:
        result[removed[variant]]=0.
    selected={'only_text_vision_loss':'w_tv','only_text_fusion_loss':'w_tf','only_vision_fusion_loss':'w_vf'}
    if variant in selected:
        for key in ('w_tv','w_tf','w_vf'):
            if key!=selected[variant]:
                result[key]=0.
    return result
