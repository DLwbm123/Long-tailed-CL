"""Narrow medical transfer helpers for pinned AIR/APART author implementations."""
import ast
import copy

import torch


def label_map(order):
    if len(set(order)) != len(order):
        raise ValueError('Repeated class in order')
    return {original: column for column, original in enumerate(order)}


def validate_access(config, split, delivery_receipts=None):
    if config['dataset'] not in ('ISIC', 'HK') or config['test_accessed']:
        raise ValueError('Unauthorized dataset or test access')
    if split not in ('train', 'development_validation'):
        raise ValueError('Sealed split')
    if split == 'development_validation' and not config.get('all_four_trained'):
        raise ValueError('Four-method/dataset training barrier not met')
    if delivery_receipts is not None:
        required = ('module_drift10_20261011', 'rfmid_contrast4_20261011')
        for name in required:
            local, remote = delivery_receipts[name]
            for receipt in (local, remote):
                if (receipt.get('status') != 'DELIVERED' or not receipt.get('nas_saved')
                        or receipt.get('anonymous_http_status') != 200):
                    raise ValueError('Protected round not fully delivered')
            if local['commit'] != remote['commit']:
                raise ValueError('Delivery receipt mismatch')


def extend_frequency_embedding(assigner, max_frequency, seed):
    old = assigner.cls_emb
    capacity = max(old.num_embeddings, max_frequency + 1)
    if capacity == old.num_embeddings:
        return
    generator = torch.Generator().manual_seed(seed + 1000003)
    tail = torch.randn(capacity - old.num_embeddings, old.embedding_dim,
                       generator=generator, dtype=old.weight.dtype)
    weight = torch.cat([old.weight.detach().cpu().clone(), tail])
    assigner.cls_emb = torch.nn.Embedding.from_pretrained(weight, freeze=False).to(old.weight.device)
    assert torch.equal(assigner.cls_emb.weight[:old.num_embeddings], old.weight)


def load_apart_pretrained(model, state):
    mapped = {}
    for key, value in state.items():
        if key.startswith('head.'):
            continue
        if '.qkv.' in key:
            for name, part in zip(('q_proj', 'k_proj', 'v_proj'), value.chunk(3, dim=0)):
                mapped[key.replace('qkv', name)] = part
        else:
            mapped[key.replace('.mlp.fc', '.fc')] = value
    own = model.state_dict()
    missing = set(own) - set(mapped)
    expected = {key for key in own if key.startswith(('pool.', 'pool_few.', 'assigner.', 'head.', 'head_few.'))}
    if missing != expected or set(mapped) - set(own):
        raise ValueError('Pretrained core mismatch')
    result = model.load_state_dict(mapped, strict=False)
    assert set(result.missing_keys) == expected and not result.unexpected_keys
    assert all(torch.equal(model.state_dict()[k], v) for k, v in mapped.items())
    for key, param in model.named_parameters():
        param.requires_grad_(key in missing)
    return len(mapped)


def sealed_apart_train(author_method, author_globals):
    """Remove only the author's five-epoch evaluation/log branch; retain its loss/update AST."""
    method = copy.deepcopy(author_method)
    removed = 0
    for node in ast.walk(method):
        for field, value in ast.iter_fields(node):
            if not isinstance(value, list):
                continue
            replacement = []
            for item in value:
                if (isinstance(item, ast.If) and
                        any(isinstance(x, ast.Attribute) and x.attr == '_compute_accuracy'
                            for statement in item.body for x in ast.walk(statement))):
                    assert item.orelse and '5' in ast.unparse(item.test)
                    replacement.extend(item.orelse)
                    removed += 1
                else:
                    replacement.append(item)
            setattr(node, field, replacement)
    assert removed == 1
    assert not any(isinstance(x, ast.Attribute) and x.attr == '_compute_accuracy' for x in ast.walk(method))
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    namespace = dict(author_globals)
    exec(compile(module, '<sealed_author_apart>', 'exec'), namespace)
    return namespace[method.name]


def air_checkpoint(arm):
    # Author cnt is not a registered buffer, so state_dict alone loses its normalization state.
    return dict(state_dict=copy.deepcopy(arm.state_dict()), cnt=arm.cnt.detach().clone(), gamma=arm.gamma)


def restore_air(arm, checkpoint):
    if checkpoint['gamma'] != arm.gamma:
        raise ValueError('AIR regularizer mismatch')
    for key, value in checkpoint['state_dict'].items():
        setattr(arm, key, value.detach().clone())
    arm.cnt = checkpoint['cnt'].detach().clone()
    assert arm.out_features == len(arm.cnt) and (arm.cnt > 0).all()
