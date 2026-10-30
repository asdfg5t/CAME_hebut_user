import copy
import torch
from came_repro.kg_linear_repair import make_solver, linear_objective


def fixture():
    generator=torch.Generator().manual_seed(541)
    x=torch.randn(40,8,generator=generator)
    y=(x[:,:3]@torch.tensor([[2.,0.,-1.],[0.,2.,1.],[-1.,0.,2.]])).argmax(1)
    return x,y


def steps(head,optimizer,x,y,count):
    for _ in range(count):
        def closure():
            optimizer.zero_grad()
            loss,_,_=linear_objective(head,x,y,.0001)
            loss.backward()
            return loss
        optimizer.step(closure)


def test_fixed_step_solver_reduces_convex_objective_without_changing_feature_width():
    torch.manual_seed(1);x,y=fixture();head=torch.nn.Linear(8,3);solver=make_solver(head.parameters())
    before=linear_objective(head,x,y,.0001)[0].item()
    steps(head,solver,x,y,100)
    after=linear_objective(head,x,y,.0001)[0].item()
    assert after<before*.2
    assert head.weight.shape==(3,8)


def test_quasi_newton_history_restores_continuous_updates():
    torch.manual_seed(1);x,y=fixture();head=torch.nn.Linear(8,3);solver=make_solver(head.parameters())
    steps(head,solver,x,y,5)
    weights=copy.deepcopy(head.state_dict());history=copy.deepcopy(solver.state_dict())
    steps(head,solver,x,y,25)
    restored=torch.nn.Linear(8,3);restored.load_state_dict(weights)
    restored_solver=make_solver(restored.parameters());restored_solver.load_state_dict(history)
    steps(restored,restored_solver,x,y,25)
    for a,b in zip(head.parameters(),restored.parameters()):assert torch.equal(a,b)
