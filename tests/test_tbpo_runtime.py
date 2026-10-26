"""Run original/current TBPO-Q/A tiny forward functions without optional logging/FSDP imports."""
import ast
import copy
import typing
from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch',reason='runtime TBPO equivalence requires PyTorch')
from test_baselines import baseline
from test_reference import ROOT


def extracted(source: str, names: list[str], namespace: dict) -> dict:
    """Compile original functions unchanged, bypassing import-only dependencies (§12)."""
    tree = ast.parse(source)
    nodes = [node for node in ast.walk(tree) if isinstance(node,ast.FunctionDef) and node.name in names]
    assert len(nodes) == len(names)
    ns = dict(namespace)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<tbpo-isolated>','exec'),ns)
    return ns


def test_tbpo_q_a_tiny_forward_equivalence() -> None:
    """Original and current forward scores/rewards/masks match bit-for-bit (§12)."""
    from baseline_head import BaselineHead
    ns = {**vars(typing),'torch':torch,'nn':torch.nn}
    utils = extracted(baseline('utils.py'),['pad_to_length','concatenated_inputs','compute_tbpo_loss_mask'],ns)
    scorers = extracted(baseline('loss/loss_utils.py'),['Q_tbpo_get_batch_logps','compute_entropy','compute_kl'],utils)
    functions = ['Q_tbpo_concatenated_forward','A_tbpo_concatenated_forward']
    before = extracted(baseline('trainers.py'),functions,scorers)
    after = extracted((ROOT/'trainers.py').read_text(),functions,scorers)

    class Backbone(torch.nn.Module):
        """Tuple-output transformer stand-in for inherited Q hook (§12)."""
        def __init__(self) -> None:
            """Tiny shared causal architecture (§12)."""
            super().__init__()
            self.embedding = torch.nn.Embedding(7,8)
            self.gru = torch.nn.GRU(8,8,batch_first=True)
        def forward(self, ids: torch.Tensor) -> tuple:
            """Return causal hidden states in the expected base-model position (§12)."""
            return (self.gru(self.embedding(ids))[0],)

    class LM(torch.nn.Module):
        """Mimic inherited CausalLM.model hook surface (§3)."""
        def __init__(self) -> None:
            """Create tiny pretrained-free fixture (§12)."""
            super().__init__()
            self.model = Backbone()
            self.head = torch.nn.Linear(8,7)
        def forward(self, ids: torch.Tensor, **kwargs: object) -> SimpleNamespace:
            """Expose logits while triggering the original head hook (§12)."""
            return SimpleNamespace(logits=self.head(self.model(ids)[0]))

    torch.manual_seed(2)
    model,reference = LM(),LM().eval().requires_grad_(False)
    fixture = SimpleNamespace(baseline_head=BaselineHead(8,mlp_dim=4),
                 config=SimpleNamespace(loss=SimpleNamespace(beta=.1),model=SimpleNamespace(baseline_l=-10,baseline_u=10)))
    batch = {}
    for side,tokens in [('chosen',[[1,2,3,4],[1,3,4,5]]),('rejected',[[1,2,4,5],[1,3,5,4]])]:
        batch[side+'_input_ids'] = torch.tensor(tokens)
        batch[side+'_attention_mask'] = torch.ones((2,4),dtype=torch.long)
        batch[side+'_labels'] = torch.tensor([[-100,-100,3,4],[-100,-100,4,5]])
    for function in functions:
        original = before[function](copy.deepcopy(fixture),copy.deepcopy(model),reference,batch)
        current = after[function](copy.deepcopy(fixture),copy.deepcopy(model),reference,batch)
        for a,b in zip(original,current):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
        # One tiny training step with the inherited SBA loss and optimizer recipes.
        from loss.loss import bregman_loss
        from loss.h_function import make_h
        updated = []
        for implementation in (before,after):
            policy,owner = copy.deepcopy(model),copy.deepcopy(fixture)
            optimizer = torch.optim.RMSprop(policy.parameters(),lr=5e-7,weight_decay=0)
            head_optimizer = torch.optim.AdamW(owner.baseline_head.parameters(),lr=.001,weight_decay=0)
            outputs = implementation[function](owner,policy,reference,batch)
            losses,_ = bregman_loss(outputs[0],outputs[5],make_h('sba',lam=0.,s=4.))
            losses.mean().backward()
            optimizer.step()
            head_optimizer.step()
            updated.append({**{k:v for k,v in policy.state_dict().items()},
                            **{'head.'+k:v for k,v in owner.baseline_head.state_dict().items()}})
        for key in updated[0]:
            torch.testing.assert_close(updated[0][key],updated[1][key],atol=0,rtol=0)
