"""PyTorch models. H2GCN is an independent port, not the original TF runner."""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class GATLayer(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.linear = nn.Linear(hidden, hidden, bias=False)
        self.source = nn.Parameter(torch.empty(hidden))
        self.target = nn.Parameter(torch.empty(hidden))
        nn.init.normal_(self.source, std=0.1)
        nn.init.normal_(self.target, std=0.1)

    def forward(self, h, edge):
        h = self.linear(h)
        row, col = edge
        scores = F.leaky_relu((h @ self.source)[row] + (h @ self.target)[col], 0.2)
        maximum = torch.full((len(h),), -torch.inf, device=h.device)
        maximum.scatter_reduce_(0, row, scores.detach(), reduce='amax', include_self=True)
        ex = torch.exp(scores - maximum[row])
        denominator = torch.zeros(len(h), device=h.device).index_add_(0, row, ex)
        weight = ex / denominator[row].clamp_min(1e-12)
        return torch.zeros_like(h).index_add_(0, row, weight[:, None] * h[col])


class NonLocal(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        h = cfg['hidden']
        self.q, self.k, self.v = (nn.Linear(h, h, bias=False) for _ in range(3))
        self.block = cfg['attention_block']
        self.recompute = cfg['checkpoint_attention']

    def forward(self, h, ids, inspect_node=None):
        q, key, value = self.q(h), self.k(h), self.v(h)
        inspected = None

        def aggregate(qb, keys, values, index):
            if index is None:
                weights = (qb @ keys.T / math.sqrt(keys.shape[-1])).softmax(-1)
                return weights @ values
            valid = index >= 0
            safe = index.clamp_min(0)
            score = (qb[:, None] * keys[safe]).sum(-1) / math.sqrt(keys.shape[-1])
            # Finite masking plus renormalization handles empty candidate rows.
            weights = score.masked_fill(~valid, -1e9).softmax(-1) * valid
            weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
            return (weights[..., None] * values[safe]).sum(1)

        parts = []
        for b in range(0, len(h), self.block):
            index = None if ids is None else ids[b:b+self.block]
            qb = q[b:b+self.block]
            if self.training and self.recompute:
                parts.append(checkpoint(aggregate, qb, key, value, index, use_reentrant=False))
            else:
                parts.append(aggregate(qb, key, value, index))
        if inspect_node is not None:
            v = inspect_node
            selected = torch.arange(len(h), device=h.device) if ids is None else ids[v][ids[v] >= 0]
            weights = ((q[v] * key[selected]).sum(-1) / math.sqrt(h.shape[-1])).softmax(0)
            inspected = {'ids': selected.detach().cpu(), 'weights': weights.detach().cpu()}
        return torch.cat(parts), inspected


class Classifier(nn.Module):
    def __init__(self, input_dim, classes, cfg):
        super().__init__()
        self.cfg = cfg
        self.kind = cfg['model']
        h = cfg['hidden']
        self.encoder = nn.Linear(input_dim, h, bias=self.kind != 'h2gcn')
        output = 1 if classes == 2 else classes
        if self.kind == 'h2gcn':
            # Ego + all intermediate propagation states; no nonlinearities between hops.
            width = h * (2 ** (cfg['layers'] + 1) - 1)
            self.out = nn.Linear(width, output, bias=False)
            return
        if self.kind == 'mlp':
            self.mlp = nn.ModuleList([nn.Linear(h, h) for _ in range(cfg['layers'])])
        elif self.kind != 'global':
            layer = GATLayer if cfg['backbone'] == 'gat' else lambda d: nn.Linear(d, d, bias=False)
            self.local = nn.ModuleList([layer(h) for _ in range(cfg['layers'])])
            self.norms = nn.ModuleList([nn.LayerNorm(h) if cfg['layernorm'] else nn.Identity()
                                       for _ in range(cfg['layers'])])
        if self.kind in {'global', 'fixed', 'learned', 'proposed', 'concat'}:
            self.nonlocal_branch = NonLocal(cfg)
        if self.kind in {'learned', 'proposed', 'concat'}:
            self.gate = nn.Sequential(nn.Linear(2*h + 1 + (self.kind == 'concat'), 32),
                                      nn.ReLU(), nn.Linear(32, 1))
            if self.kind == 'proposed':
                self.gamma = nn.Parameter(torch.tensor(-5. if cfg['positive_gamma'] else 0.))
        self.out = nn.Linear(h, output)

    def forward(self, data, inspect_node=None):
        cfg = self.cfg
        h = F.relu(self.encoder(data['x']))
        extra = {}
        if self.kind == 'h2gcn':
            states = [h]
            for _ in range(cfg['layers']):
                h = torch.cat([torch.sparse.mm(a, h) for a in data['hops']], dim=-1)
                states.append(h)
            z = F.dropout(torch.cat(states, -1), cfg['dropout'], self.training)
        elif self.kind == 'mlp':
            z = h
            for layer in self.mlp:
                z = F.dropout(F.relu(layer(z)), cfg['dropout'], self.training)
        else:
            local = h
            if self.kind != 'global':
                for layer, norm in zip(self.local, self.norms):
                    prev = local
                    local = (layer(local, data['edge_index']) if cfg['backbone'] == 'gat'
                             else torch.sparse.mm(data['adj'], layer(local)))
                    if cfg['residual']:
                        local = local + prev
                    local = F.dropout(F.relu(norm(local)), cfg['dropout'], self.training)
            if self.kind == 'gcn':
                z = local
            else:
                global_, attention = self.nonlocal_branch(h, data['candidates'], inspect_node)
                if attention is not None:
                    extra['attention'] = attention
                if self.kind == 'global':
                    z = global_
                else:
                    if self.kind == 'fixed':
                        alpha = torch.full((len(h), 1), 0.5, device=h.device)
                    else:
                        inputs = [data['degree'][:, None], local, global_]
                        if self.kind == 'concat':
                            inputs.append(data['proxy'][:, None])
                        gate = self.gate(torch.cat(inputs, -1))
                        if self.kind == 'proposed':
                            gamma = F.softplus(self.gamma) if cfg['positive_gamma'] else self.gamma
                            gate = gate + gamma * data['proxy'][:, None]
                        alpha = gate.sigmoid()
                    z = alpha * local + (1-alpha) * global_
                    extra['alpha'] = alpha[:, 0]
        logits = self.out(z)
        return logits, extra

