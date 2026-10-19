import itertools
import random

from .utils import digest

DEFAULTS = dict(hidden=64, layers=2, dropout=0.3, lr=0.001, l2=0., k=8,
                backbone='gcn', residual=False, layernorm=False, retrieval='exact',
                retrieval_seed=42, exclude_hops=1, positive_gamma=False, shuffle_proxy=False,
                retrieval_block=64, attention_block=256, checkpoint_attention=True,
                max_twohop_edges=20000000, full_max_nodes=12000)
MODELS = ['mlp', 'gcn', 'global', 'fixed', 'learned', 'proposed', 'h2gcn']
NONLOCAL = {'global', 'fixed', 'learned', 'proposed', 'concat'}


def validate_plan(plan):
    from .data import DATASETS
    if not set(plan['datasets']) <= set(DATASETS):
        raise ValueError('Unknown dataset')
    if not set(plan['models']) <= set(MODELS):
        raise ValueError('Unknown main model')
    if 'proposed' not in plan['models']:
        raise ValueError('Ablation protocol requires proposed model')
    for key in ['tune_splits', 'final_splits']:
        ids = plan[key]
        if not ids or len(set(ids)) != len(ids) or any(i not in range(10) for i in ids):
            raise ValueError(f'Invalid {key}')
    if min(plan['trials'], plan['epochs'], plan['patience']) < 1:
        raise ValueError('Budgets must be positive')


def configurations(plan, model):
    base = dict(DEFAULTS, **plan.get('defaults', {}), model=model)
    search = plan.get('search', {}) if plan['trials'] > 1 else {}
    applicable = {'hidden', 'lr', 'l2'}
    if model != 'global':
        applicable.update(['layers', 'dropout'])
    if model in NONLOCAL:
        applicable.add('k')
    if model not in {'mlp', 'global', 'h2gcn'}:
        applicable.update(['residual', 'layernorm'])
    if model == 'h2gcn':
        search = plan.get('h2_search', search) if plan['trials'] > 1 else {}
    search = {k: v for k, v in search.items() if k in applicable}
    names = sorted(search)
    combos = [dict(zip(names, values)) for values in itertools.product(*(search[n] for n in names))]
    # Common randomized order for shared hyperparameter spaces, never using outcomes.
    random.Random(plan['search_seed']).shuffle(combos)
    if len(combos) < plan['trials']:
        raise ValueError(f'Not enough unique configurations for {model}')
    return [dict(base, **x) for x in combos[:plan['trials']]]


def variants(plan, selected):
    rows = []
    for dataset in plan['datasets']:
        for model in plan['models']:
            rows.append(dict(dataset=dataset, group='main', name=model, config=selected[dataset][model]))
        proposed = selected[dataset]['proposed']
        changes = {'proposed_control': {}, 'no_proxy': {'model': 'learned'},
                   'shuffled_proxy': {'shuffle_proxy': True}, 'random_k': {'retrieval': 'random'},
                   'concat_proxy': {'model': 'concat'}}
        for name, delta in changes.items():
            rows.append(dict(dataset=dataset, group='ablation', name=name, config=dict(proposed, **delta)))
        if plan.get('sensitivity_k'):
            for k in plan['sensitivity_k']:
                rows.append(dict(dataset=dataset, group='sensitivity', name=f'k_{k}', config=dict(proposed, k=k)))
        for name in plan.get('extensions', []):
            deltas = {'gat': {'backbone': 'gat'}, 'twohop_exclusion': {'exclude_hops': 2},
                      'positive_gamma': {'positive_gamma': True}, 'ann': {'retrieval': 'ann'},
                      'full_attention': {'retrieval': 'full'}}
            if name not in deltas:
                raise ValueError(f'Unknown extension {name}')
            rows.append(dict(dataset=dataset, group='extension', name=name, config=dict(proposed, **deltas[name])))
    return rows


def run_identity(dataset_hash, cfg, split, seed, plan, source_hash):
    return digest(dict(dataset=dataset_hash, config=cfg, split=split, seed=seed,
                       epochs=plan['epochs'], patience=plan['patience'], source=source_hash))

