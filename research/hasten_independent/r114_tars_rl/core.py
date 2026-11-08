"""Autoregressive recognition with frozen visual cache and language-only LoRA."""
import contextlib
import json
import math
import re
import time
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def normalize(name):
    return re.sub(r'[\s_\-]+', '', name.lower().strip())


def tailness(counts, correct):
    n = torch.as_tensor(counts, dtype=torch.float64)
    error = 1 - torch.as_tensor(correct, dtype=torch.float64) / n
    t = (n.mean() / n * error).pow(.75)
    median = t.quantile(.5)
    w = (t / median.clamp_min(1e-8)).clamp_min(1)
    assert torch.isfinite(w).all()
    return w, float((n * w).sum() / n.sum()), float(median)


def advantages(rewards):
    return (rewards - rewards.mean(-1, keepdim=True)) / (rewards.std(-1, unbiased=False, keepdim=True) + 1e-6)


def surrogate(lp, old, ref, mask, adv, weights):
    # Apply class weight after group normalization, and never to KL.
    ratio = (lp - old).exp()
    a = adv[:, None]
    pg = torch.minimum(ratio * a, ratio.clamp(.8, 1.2) * a)
    delta = ref - lp
    assert torch.isfinite(delta).all() and float(delta.detach().abs().max()) < 80
    kl = torch.expm1(delta) - delta
    lengths = mask.sum(-1).clamp_min(1)
    pg = (pg * mask).sum(-1) / lengths
    kl = (kl * mask).sum(-1) / lengths
    return -(weights * pg).mean() + .04 * kl.mean(), pg.detach(), kl.detach()


class LoRA(nn.Module):
    def __init__(self, base, rank=8):
        super().__init__()
        self.base = base
        self.a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.register_buffer('ref_a', self.a.detach().clone(), persistent=False)
        self.register_buffer('ref_b', self.b.detach().clone(), persistent=False)
        self.reference = False

    def forward(self, x):
        a, b = (self.ref_a, self.ref_b) if self.reference else (self.a, self.b)
        return self.base(x) + (2 * F.linear(F.linear(x.float(), a), b)).to(x.dtype)


class Engine:
    def __init__(self, cfg):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        self.cfg = cfg
        torch.manual_seed(74005)
        self.processor = AutoProcessor.from_pretrained(cfg['model'], local_files_only=True, min_pixels=224**2, max_pixels=224**2)
        self.tokenizer = self.processor.tokenizer
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            cfg['model'], local_files_only=True, torch_dtype=torch.bfloat16,
            attn_implementation='sdpa', low_cpu_mem_usage=True).cuda().eval().requires_grad_(False)
        self.adapters = []
        for layer in self.model.model.layers:
            for name in ['q_proj', 'v_proj']:
                module = LoRA(getattr(layer.self_attn, name))
                setattr(layer.self_attn, name, module)
                self.adapters.append(module)
        self.parameters = [p for a in self.adapters for p in [a.a, a.b]]
        self.eos = self.model.config.eos_token_id
        self.pad = self.tokenizer.pad_token_id
        messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': cfg['prompt']}]}]
        self.text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        self.prompt_ids = self.grid = None

    def snapshot(self):
        return [(a.a.detach().cpu().clone(), a.b.detach().cpu().clone()) for a in self.adapters]

    @torch.no_grad()
    def restore(self, state):
        assert len(state) == len(self.adapters)
        for a, (u, v) in zip(self.adapters, state):
            a.a.copy_(u)
            a.b.copy_(v)

    @torch.no_grad()
    def freeze_reference(self):
        for a in self.adapters:
            a.ref_a.copy_(a.a)
            a.ref_b.copy_(a.b)

    @contextlib.contextmanager
    def reference(self):
        for a in self.adapters:
            a.reference = True
        try:
            yield
        finally:
            for a in self.adapters:
                a.reference = False

    @torch.no_grad()
    def encode(self, images):
        inputs = self.processor(text=[self.text] * len(images), images=[im.resize((224, 224)) for im in images], padding=True, return_tensors='pt').to('cuda')
        assert torch.equal(inputs.image_grid_thw, torch.tensor([[1, 16, 16]], device='cuda').expand(len(images), -1))
        assert all(torch.equal(row, inputs.input_ids[0]) for row in inputs.input_ids)
        self.prompt_ids = inputs.input_ids[:1].detach()
        self.grid = inputs.image_grid_thw[:1].detach()
        features = self.model.visual(inputs.pixel_values.to(self.model.visual.dtype), grid_thw=inputs.image_grid_thw)
        assert features.shape == (len(images) * 64, 2048)
        return features.reshape(len(images), 64, 2048), inputs

    def embeddings(self, ids, features):
        embedded = self.model.get_input_embeddings()(ids)
        mask = ids == self.model.config.image_token_id
        assert int(mask.sum()) == len(ids) * 64
        return embedded.masked_scatter(mask[:, :, None].expand_as(embedded), features.to(embedded.dtype))

    def positions(self, ids, attention):
        return self.model.get_rope_index(ids, self.grid.expand(len(ids), -1), attention_mask=attention)

    def token_logps(self, features, tokens, mask):
        ids = torch.cat([self.prompt_ids.expand(len(tokens), -1), tokens], dim=1)
        attention = torch.cat([torch.ones_like(self.prompt_ids).expand(len(tokens), -1), mask.long()], dim=1)
        position, _ = self.positions(ids, attention)
        hidden = self.model.model(inputs_embeds=self.embeddings(ids, features), position_ids=position,
                                  attention_mask=attention, use_cache=False, return_dict=True).last_hidden_state
        p = self.prompt_ids.shape[1]
        logits = self.model.lm_head(hidden[:, p-1:-1]).float()
        # Only response positions are projected, rather than the whole image prompt.
        lp = -F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.reshape(-1), reduction='none').reshape_as(tokens)
        return lp

    def label_tokens(self, labels):
        items = [self.tokenizer.encode(s, add_special_tokens=False) + [self.eos] for s in labels]
        assert all(len(x) <= 24 for x in items)
        L = max(map(len, items))
        tokens = torch.full((len(items), L), self.pad, dtype=torch.long, device='cuda')
        mask = torch.zeros_like(tokens, dtype=torch.bool)
        for i, item in enumerate(items):
            tokens[i, :len(item)] = torch.tensor(item, device='cuda')
            mask[i, :len(item)] = True
        return tokens, mask

    @torch.no_grad()
    def generate(self, features, generator=None, sample=False):
        n = len(features)
        ids = self.prompt_ids.expand(n, -1)
        attention = torch.ones_like(ids)
        position, delta = self.positions(ids, attention)
        output = self.model.model(inputs_embeds=self.embeddings(ids, features), position_ids=position,
                                  attention_mask=attention, use_cache=True, return_dict=True)
        past = output.past_key_values
        hidden = output.last_hidden_state[:, -1:]
        done = torch.zeros(n, dtype=torch.bool, device='cuda')
        tokens, logs, masks = [], [], []
        for step in range(24):
            lp = self.model.lm_head(hidden).squeeze(1).float().log_softmax(-1)
            token = torch.multinomial(lp.exp(), 1, generator=generator).squeeze(1) if sample else lp.argmax(-1)
            token = torch.where(done, self.pad, token)
            active = ~done
            tokens.append(token)
            logs.append(lp.gather(1, token[:, None]).squeeze(1))
            masks.append(active)
            done = done | (token == self.eos)
            if bool(done.all()) or step == 23:
                break
            attention = torch.cat([attention, active.long()[:, None]], dim=1)
            pos = (ids.shape[1] + step + delta).view(1, n, 1).expand(3, -1, -1)
            output = self.model.model(input_ids=token[:, None], position_ids=pos, attention_mask=attention,
                                      past_key_values=past, use_cache=True, return_dict=True)
            past, hidden = output.past_key_values, output.last_hidden_state
        tokens, logs, masks = torch.stack(tokens, 1), torch.stack(logs, 1), torch.stack(masks, 1)
        text = self.tokenizer.batch_decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return tokens, masks, logs, [normalize(s) for s in text], done


def software_check():
    r = torch.tensor([[0., 0., 0., 0.], [0., 1., 0., 1.]])
    a = advantages(r)
    assert torch.equal(a[0], torch.zeros(4)) and torch.allclose(a[1], torch.tensor([-1., 1., -1., 1.]), atol=3e-6)
    lp = torch.full((4, 3), -1.0, requires_grad=True)
    mask = torch.tensor([[1, 1, 0]] * 4, dtype=torch.bool)
    loss, _, kl = surrogate(lp, lp.detach(), lp.detach(), mask, torch.zeros(4), torch.ones(4) * 7)
    loss.backward()
    assert torch.equal(lp.grad, torch.zeros_like(lp)) and torch.equal(kl, torch.zeros_like(kl))
    lp2 = torch.full((4, 3), -1., requires_grad=True)
    loss2, _, _ = surrogate(lp2, lp2.detach(), lp2.detach() + .1, mask, torch.zeros(4), torch.ones(4))
    loss2.backward()
    assert lp2.grad[:, :2].abs().sum() > 0 and lp2.grad[:, 2].abs().sum() == 0
    w, constant, median = tailness([100, 10, 2], [90, 5, 0])
    assert w[2] > w[1] >= w[0] >= 1 and 1 <= constant <= max(w)
    return {'passed': True, 'zero_advantage_at_reference_exact_zero_gradient': True,
            'off_reference_zero_advantage_KL_gradient_preserved': True, 'padding_gradient_zero': True,
            'positive_class_tailness_order': True, 'real_images': 0}
