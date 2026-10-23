"""Same coupled objective, retained attention graph and MLP-only recomputation."""
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from core import NativeEvidence, representation_objective

class CheckpointEvidence(NativeEvidence):
    def forward(self, tokens):
        x = tokens
        for j, block in enumerate(self.backbone.blocks):
            n = 0
            if self.prompts is not None and j < len(self.prompts):
                n = len(self.prompts[j])
                pt = self.prompts[j][None].expand(len(x), -1, -1)
                x = torch.cat([x[:, :1], pt, x[:, 1:]], 1)
            if torch.is_grad_enabled() and self.prompts is not None:
                x = x + block.drop_path1(block.ls1(block.attn(block.norm1(x))))
                # Checkpoint the MLP and its input norm. Attention is never recomputed.
                def mlp_path(t, current=block):
                    return current.drop_path2(current.ls2(current.mlp(current.norm2(t))))
                x = x + checkpoint(mlp_path, x, use_reentrant=False, preserve_rng_state=False)
            else:
                x = block(x)
            if n:
                x = torch.cat([x[:, :1], x[:, 1+n:]], 1)
        x = self.backbone.norm(x)
        return F.normalize(torch.cat([x[:, 0], x[:, 1:].mean(1)], 1), dim=1)

def graph_update(model, optimizer, tokens, teacher, labels, weights, arm, cfg, cost, temphead=None):
    device = model.prompts.device
    optimizer.zero_grad(set_to_none=True)
    micro = cfg['microbatch']
    z = torch.cat([model(tokens[i:i+micro].to(device).float()) for i in range(0, len(tokens), micro)])
    loss, risk, rc = representation_objective(z, teacher.to(device), labels.to(device), weights.to(device), arm, cfg['ridge'], cfg['consistency_weight'], temphead)
    loss.backward()
    params = [model.prompts] + ([] if temphead is None else [temphead])
    norm = torch.nn.utils.clip_grad_norm_(params, cfg['gradient_clip'])
    assert torch.isfinite(norm)
    optimizer.step()
    for field in ['body_forward_rows', 'body_backward_rows', 'logical_rows']:
        cost[field] = cost.get(field, 0) + len(tokens)
    cost['mlp_checkpoint_recomputed_rows'] = cost.get('mlp_checkpoint_recomputed_rows', 0) + (len(tokens) if isinstance(model, CheckpointEvidence) else 0)
    cost['updates'] = cost.get('updates', 0) + 1
    cost['inner_dual_solves'] = cost.get('inner_dual_solves', 0) + (2 if arm.startswith('RIDGE') else 0)
    return dict(objective=float(loss.detach()), outer_risk=float(risk.detach()), response_consistency=float(rc.detach()), gradient_norm=float(norm))
