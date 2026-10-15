import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize_scalar
from scipy.special import betainc, betaln
from scipy.stats import beta as beta_distribution
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch import nn
from torch.nn import functional as F


class Quality(nn.Module):
    def __init__(self):
        super().__init__()
        self.factor = 1.
        self.net = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2))

    def forward(self, x):
        return (F.softplus(self.net(x)) + .05) * self.factor


class Policy(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.embedding = nn.Linear(11, 64)
        self.attention = nn.MultiheadAttention(64, 4, batch_first=True)
        self.coordinator = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 2))
        self.groups = [3, 2, 3, 1, 2]
        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(size + 66, 32), nn.ReLU(), nn.Linear(32, 1)) for size in self.groups])
        self.critic = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 51))

    def forward(self, features, context):
        h = F.relu(self.embedding(features))[None]
        attended = self.attention(h, h, h, need_weights=False)[0][0]
        slow = self.coordinator(context).sigmoid()
        if self.c['variant'] == 'flat':
            slow = slow * 0 + .5
        budget = slow[0]
        multiplier = 2 * slow[1]
        scalars = torch.stack([budget, multiplier])[None].expand(len(features), -1)
        contributions = []
        offset = 0
        for i, (size, head) in enumerate(zip(self.groups, self.heads)):
            value = head(torch.cat([features[:, offset:offset + size], attended, scalars], 1)).squeeze(-1)
            if i == 2:
                value = multiplier * value
            elif i == 3:
                value = (1 - budget) * value
            contributions.append(value)
            offset += size
        contribution = torch.stack(contributions, 1)
        return contribution.sum(1), self.critic(attended.mean(0)), contribution


class Fab:
    def __init__(self, c, scenario, seed, quality):
        self.c, self.scenario, self.seed, self.quality = c, scenario, seed, quality
        self.rng = np.random.default_rng(seed)
        self.time = 0.
        self.end = (c['warmup_days'] + c['scored_days']) * 24
        self.arrival_rate = scenario['utilization'] * c['tools'] / c['operations']
        self.next_arrival = 0.
        self.lots = []
        self.queues = [set() for _ in range(c['groups'])]
        self.total_completed = 0
        self.genealogy_nodes = {}
        self.tools = [dict(group=i % c['groups'], finish=np.inf, lot=None, family=-1, starts=0) for i in range(c['tools'])]
        self.wip_area = 0.
        self.context = np.zeros(4, dtype=np.float32)
        self.next_coordinator = 0.
        self.completed = []
        self.violations = 0

    def release(self):
        i = len(self.lots)
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, i, 100]))
        family = int(rng.integers(3))
        route = [(operation + 3 * family) % self.c['groups'] for operation in range(self.c['operations'])]
        processing = rng.lognormal(-.5 * self.c['processing_sigma'] ** 2, self.c['processing_sigma'], self.c['operations'])
        self.lots.append(dict(id=i, family=family, route=route, processing=processing, op=0, arrival=self.time,
                              ready=self.time, busy=False, done=False, due=self.time + self.scenario['due_factor'] * processing.sum(),
                              rework=0, genealogy=0., parent=None, last_tool_usage=0., map=rng.normal(0, .1, 2)))
        self.queues[route[0]].add(i)
        self.genealogy_nodes[(i, 0)] = dict(parent=None, risk=0.)
        self.next_arrival = self.time + self.rng.exponential(1 / self.arrival_rate)

    def quality_input(self, lot):
        family = np.eye(3)[lot['family']]
        return np.r_[lot['op'] / self.c['operations'], (self.time - lot['arrival']) / self.c['operations'],
                     max(0, self.time - lot['ready']) / 24, lot['rework'] / 5,
                     lot['last_tool_usage'] / 100, family, lot['map']].astype(np.float32)

    def advance(self):
        while self.time < self.end:
            for tool in self.tools:
                if tool['lot'] is not None and tool['finish'] <= self.time:
                    lot = self.lots[tool['lot']]
                    lot['busy'] = False
                    lot['ready'] = self.time
                    lot['op'] += 1
                    rng = np.random.default_rng(np.random.SeedSequence([self.seed, lot['id'], lot['op'], lot['rework'], 200]))
                    if rng.random() < self.scenario['rework'] and lot['rework'] < self.c['max_rework']:
                        previous = lot['genealogy']
                        lot['parent'] = (lot['id'], lot['rework'])
                        lot['genealogy'] = 1 - (1 - self.scenario['rework']) * (1 - self.c['propagation'] * previous)
                        lot['rework'] += 1
                        lot['op'] = max(0, lot['op'] - 2)
                    if lot['op'] >= self.c['operations']:
                        lot['done'] = True
                        self.total_completed += 1
                        if self.time >= self.c['warmup_days'] * 24:
                            feature = self.quality_input(lot)
                            a, b = true_quality(feature, self.scenario['yield_shift'])
                            outcome = float(rng.beta(a, b))
                            self.completed.append(dict(cycle_time=self.time - lot['arrival'], tardiness=max(0, self.time - lot['due']),
                                                       quality_loss=1 - outcome, rework=lot['rework']))
                    if not lot['done']:
                        self.queues[lot['route'][lot['op']]].add(lot['id'])
                    tool['lot'], tool['finish'] = None, np.inf
            while self.next_arrival <= self.time:
                self.release()
            for tool_id, tool in enumerate(self.tools):
                if tool['lot'] is not None:
                    continue
                candidates = [self.lots[i] for i in sorted(self.queues[tool['group']])] 
                if candidates:
                    if self.time >= self.next_coordinator:
                        waiting = [lot for lot in self.lots if not lot['done']]
                        self.context = np.array([len(waiting) / self.c['tools'],
                                                 np.mean([max(0, self.time - lot['due']) for lot in waiting]) / self.c['operations'],
                                                 np.mean([lot['genealogy'] for lot in waiting]), self.scenario['utilization']], dtype=np.float32)
                        self.next_coordinator = self.time + 4
                    return tool_id, candidates
            future = min(self.next_arrival, min(tool['finish'] for tool in self.tools), self.end)
            self.wip_area += max(0., future - max(self.time, self.c['warmup_days'] * 24)) * (len(self.lots) - self.total_completed)
            self.time = future
        return None

    def features(self, tool_id, candidates):
        inputs = np.stack([self.quality_input(lot) for lot in candidates])
        with torch.no_grad():
            parameters = self.quality(torch.tensor(inputs)).numpy()
        a, b = parameters[:, 0], parameters[:, 1]
        mean = a / (a + b)
        quantile = beta_distribution.ppf(.05, a, b)
        low = betainc(a, b, .85)
        result = []
        for i, lot in enumerate(candidates):
            parent = lot['parent']
            previous_risk = self.genealogy_nodes[parent]['risk'] if parent in self.genealogy_nodes else 0.
            genealogy = 1 - (1 - low[i]) * (1 - self.c['propagation'] * previous_risk)
            self.genealogy_nodes[(lot['id'], lot['rework'])] = dict(parent=parent, risk=float(genealogy))
            lot['genealogy'] = float(genealogy)
            if self.c['variant'] == 'no-genealogy':
                genealogy = low[i]
            quality = [mean[i], quantile[i], low[i]]
            if self.c['variant'] in ['point-yield', 'mean-hierarchy']:
                quality = [mean[i], mean[i], float(mean[i] < .85)]
            result.append([(self.time - lot['ready']) / 24, lot['processing'][lot['op']], lot['op'] / self.c['operations'],
                           (lot['due'] - self.time) / self.c['operations'], float(lot['due'] < self.time),
                           *quality, genealogy, len(candidates) / 100, float(self.tools[tool_id]['family'] != lot['family'])])
        return np.array(result, dtype=np.float32)

    def dispatch(self, tool_id, lot):
        tool = self.tools[tool_id]
        if lot['busy'] or lot['done'] or lot['route'][lot['op']] != tool['group'] or tool['lot'] is not None:
            self.violations += 1
            raise RuntimeError('Qualification, precedence or nonpreemption rule violated')
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, tool_id, tool['starts'], 300]))
        duration = float(lot['processing'][lot['op']])
        setup = self.c['setup_hours'] if tool['family'] != lot['family'] else 0
        repair = rng.exponential(self.scenario['mttr']) if rng.random() < 1 - np.exp(-duration / self.scenario['mtbf']) else 0
        tool['finish'] = self.time + duration + setup + repair
        tool['lot'], tool['family'] = lot['id'], lot['family']
        tool['starts'] += 1
        lot['busy'] = True
        self.queues[tool['group']].remove(lot['id'])
        lot['last_tool_usage'] = tool['starts']

    def score(self):
        if self.completed:
            cycle = np.mean([r['cycle_time'] for r in self.completed])
            tardy = np.mean([r['tardiness'] for r in self.completed])
            loss = np.mean([r['quality_loss'] for r in self.completed])
        else:
            cycle, tardy, loss = 0., 0., 0.
        scored_hours = max(1, self.time - self.c['warmup_days'] * 24)
        wip = self.wip_area / scored_hours / max(1, self.arrival_rate * self.c['operations'])
        cost = .25 * cycle / self.c['operations'] + .2 * tardy / self.c['operations'] + .1 * wip + .45 * loss
        return dict(cost=float(cost), cycle_time=float(cycle), throughput=len(self.completed) / scored_hours,
                    quality_loss=float(loss), wip=float(wip), completed_lots=len(self.completed), hard_rule_violations=self.violations)


def true_quality(x, shift):
    x = np.asarray(x)
    mean = np.clip(.96 - .03 * x[..., 2] - .04 * x[..., 3] - .01 * np.abs(x[..., 8]) - shift, .1, .99)
    concentration = 30 / (1 + x[..., 2].clip(0) + x[..., 3].clip(0))
    return mean * concentration, (1 - mean) * concentration


def prepare(c, out):
    rng = np.random.default_rng(c['seed'])
    n = sum(c['quality_sizes'])
    x = rng.uniform(0, 1, (n, 10)).astype(np.float32)
    a, b = true_quality(x, 0)
    y = rng.beta(a, b).clip(1e-6, 1 - 1e-6).astype(np.float32)
    split = np.repeat(['train', 'val', 'test'], c['quality_sizes'])
    np.savez_compressed(out / 'quality.npz', x=x, y=y, split=split)
    scenarios = []
    combinations = [(u, failure, repair, rework, due, shift) for u in [.8, .9, .97]
                    for failure in [168, 120, 96] for repair in [6, 10, 14] for rework in [.03, .08, .15]
                    for due in [1.8, 1.5, 1.3] for shift in [0, .05, .1]]
    chosen = rng.choice(len(combinations), 24, replace=False)
    for i, index in enumerate(chosen):
        u, failure, repair, rework, due, shift = combinations[index]
        scenarios.append(dict(id=i, split='train' if i < 12 else 'val' if i < 16 else 'test', utilization=u,
                              mtbf=failure, mttr=repair, rework=rework, due_factor=due, yield_shift=shift))
    (out / 'scenarios.json').write_text(json.dumps(scenarios))


def fit(c, out):
    torch.manual_seed(c['seed'])
    d = np.load(out / 'quality.npz')
    model = Quality()
    optimizer = torch.optim.AdamW(model.parameters(), lr=c['learning_rate'])
    ids = np.flatnonzero(d['split'] == 'train')
    for _ in range(c['quality_epochs']):
        for group in torch.tensor(ids)[torch.randperm(len(ids))].split(c['batch_size']):
            parameters = model(torch.tensor(d['x'][group]))
            a, b = parameters[:, 0], parameters[:, 1]
            y = torch.tensor(d['y'][group])
            nll = -((a - 1) * y.log() + (b - 1) * (1 - y).log() + torch.lgamma(a + b) - torch.lgamma(a) - torch.lgamma(b)).mean()
            optimizer.zero_grad()
            nll.backward()
            optimizer.step()
    torch.save(model.state_dict(), out / 'quality.pt')
    selected = d['split'] == 'val'
    with torch.no_grad():
        parameters = model(torch.tensor(d['x'][selected])).numpy()
    mean = parameters[:, 0] / parameters.sum(1)
    concentration = parameters.sum(1)
    y = d['y'][selected]
    objective = lambda log_scale: -np.mean((mean * concentration * np.exp(log_scale) - 1) * np.log(y) +
                                          ((1 - mean) * concentration * np.exp(log_scale) - 1) * np.log1p(-y) -
                                          betaln(mean * concentration * np.exp(log_scale), (1 - mean) * concentration * np.exp(log_scale)))
    scale = minimize_scalar(objective, bounds=(-2, 2), method='bounded')
    (out / 'quality_calibration.json').write_text(json.dumps(dict(concentration_scale=float(np.exp(scale.x)))))


def load_quality(out):
    model = Quality()
    model.load_state_dict(torch.load(out / 'quality.pt', weights_only=True, map_location='cpu'))
    model.eval()
    model.factor = json.loads((out / 'quality_calibration.json').read_text(encoding='utf-8'))['concentration_scale']
    return model


def train(c, out):
    torch.manual_seed(c['seed'])
    quality = load_quality(out)
    policy = Policy(c)
    optimizer = torch.optim.Adam(policy.parameters(), lr=c['learning_rate'])
    scenarios = [s for s in json.loads((out / 'scenarios.json').read_text(encoding='utf-8')) if s['split'] == 'train']
    history = []
    consumed, episode = 0, 0
    env = None
    previous_cost = 0.
    while consumed < c['train_events']:
        if env is None:
            env = Fab(c, scenarios[episode % len(scenarios)], c['seed'] + episode, quality)
            previous_cost = 0.
        rollout = []
        terminated = False
        while consumed < c['train_events']:
            decision = env.advance()
            if decision is None:
                terminated = True
                break
            tool, candidates = decision
            features = torch.tensor(env.features(tool, candidates))
            context = torch.tensor(env.context)
            with torch.no_grad():
                logits, quantiles, _ = policy(features, context)
                distribution = torch.distributions.Categorical(logits=logits)
                action = distribution.sample()
                logp = distribution.log_prob(action)
            env.dispatch(tool, candidates[int(action)])
            cost = env.score()['cost']
            rollout.append((features, context, int(action), float(logp), quantiles, cost - previous_cost))
            previous_cost = cost
            consumed += 1
            if len(rollout) >= c['rollout']:
                break
        if not rollout:
            env = None
            episode += 1
            continue
        returns = []
        total = 0.
        for item in reversed(rollout):
            total = item[-1] + c['discount'] * total
            returns.append(total)
        returns = torch.tensor(returns[::-1], dtype=torch.float32)
        eta = torch.quantile(returns, .95)
        baselines = torch.stack([item[4] for item in rollout])
        mean_advantage = baselines.mean(1) - returns
        tail_target = eta + F.relu(returns - eta) / .05
        tail_advantage = baselines.topk(3, dim=1).values.mean(1) - tail_target
        advantage = mean_advantage + (0 if c['variant'] in ['risk-neutral', 'mean-hierarchy'] else .2) * tail_advantage
        advantage = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-6)
        for _ in range(c['ppo_epochs']):
            losses = []
            for i, (features, context, action, old_logp, _, _) in enumerate(rollout):
                logits, quantiles, _ = policy(features, context)
                logp = logits.log_softmax(0)[action]
                ratio = (logp - old_logp).exp()
                actor = -torch.minimum(ratio * advantage[i], ratio.clamp(.8, 1.2) * advantage[i])
                residual = returns[i] - quantiles
                levels = (torch.arange(51) + .5) / 51
                huber = torch.where(abs(residual) <= 1, .5 * residual.square(), abs(residual) - .5)
                critic = (abs(levels - (residual.detach() < 0).float()) * huber).mean()
                losses.append(actor + .5 * critic)
            loss = torch.stack(losses).mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1)
            optimizer.step()
        history.append(dict(events=consumed, episode=episode, observed_cost=env.score()['cost']))
        if terminated:
            env = None
            episode += 1
    torch.save(policy.state_dict(), out / 'policy.pt')
    pd.DataFrame(history).to_csv(out / 'learning.csv', index=False)


def run(c, out):
    import time
    quality = load_quality(out)
    policy = Policy(c)
    policy.load_state_dict(torch.load(out / 'policy.pt', weights_only=True, map_location='cpu'))
    policy.eval()
    scenarios = [s for s in json.loads((out / 'scenarios.json').read_text(encoding='utf-8')) if s['split'] == 'test']
    rows = []
    with (out / 'actions.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['method', 'seed', 'scenario', 'time', 'tool', 'lot', 'alternative', 'logit', 'contribution_sum', 'latency_ms'])
        writer.writeheader()
        for seed in c['evaluation_seeds']:
            for scenario in scenarios:
                for method in ['YieldRisk', 'FIFO', 'EDD', 'ATC', 'mean-feature-control']:
                    env = Fab(c, scenario, seed, quality)
                    latencies = []
                    while True:
                        decision = env.advance()
                        if decision is None:
                            break
                        tool, candidates = decision
                        start = time.perf_counter()
                        features = env.features(tool, candidates)
                        if method == 'FIFO':
                            scores = np.array([env.time - lot['ready'] for lot in candidates])
                        elif method == 'EDD':
                            scores = -np.array([lot['due'] for lot in candidates])
                        elif method == 'ATC':
                            durations = np.array([lot['processing'][lot['op']] for lot in candidates])
                            slack = np.array([max(0, lot['due'] - env.time - duration) for lot, duration in zip(candidates, durations)])
                            scores = np.exp(-slack / max(1, 2 * durations.mean())) / durations
                        else:
                            if method == 'mean-feature-control':
                                features[:, 6] = features[:, 5]
                                features[:, 7] = (features[:, 5] < .85)
                            with torch.no_grad():
                                logits, _, contributions = policy(torch.tensor(features), torch.tensor(env.context))
                            scores = logits.numpy()
                        selected = int(np.argmax(scores))
                        alternate = int(np.argsort(scores)[-2]) if len(scores) > 1 else selected
                        latency = (time.perf_counter() - start) * 1000
                        latencies.append(latency)
                        writer.writerow(dict(method=method, seed=seed, scenario=scenario['id'], time=env.time,
                                             tool=tool, lot=candidates[selected]['id'], alternative=candidates[alternate]['id'],
                                             logit=float(scores[selected]), contribution_sum=float(contributions[selected].sum()) if method in ['YieldRisk', 'mean-feature-control'] else float(scores[selected]), latency_ms=latency))
                        env.dispatch(tool, candidates[selected])
                    rows.append(dict(method=method, seed=seed, scenario=scenario['id'], median_latency_ms=np.median(latencies), **env.score()))
    pd.DataFrame(rows).to_csv(out / 'episodes.csv', index=False)


def evaluate(c, out):
    d = pd.read_csv(out / 'episodes.csv')
    denominator = d[d.method == 'FIFO'].cost.mean()
    rows = []
    for method, group in d.groupby('method'):
        cost = group.cost.to_numpy()
        tail = np.sort(cost)[-max(1, int(np.ceil(.05 * len(cost)))):].mean()
        result = group.mean(numeric_only=True).to_dict()
        result.update(method=method, expected_cost_index=100 * cost.mean() / denominator, cvar_index=100 * tail / denominator)
        rows.append(result)
    pd.DataFrame(rows).to_csv(out / 'metrics.csv', index=False)
    data = np.load(out / 'quality.npz')
    model = load_quality(out)
    selected = data['split'] == 'test'
    with torch.no_grad():
        parameters = model(torch.tensor(data['x'][selected])).numpy()
    a, b = parameters[:, 0], parameters[:, 1]
    y = data['y'][selected]
    probability = betainc(a, b, .85)
    label = y < .85
    ece = sum(len(ids) / len(y) * abs(probability[ids].mean() - label[ids].mean()) for ids in np.array_split(np.argsort(probability), 10))
    quality_metrics = [dict(method='beta', accuracy=accuracy_score(label, probability >= .5), macro_f1=f1_score(label, probability >= .5, average='macro'),
                            auroc=roc_auc_score(label, probability), ece=ece,
                            nll=float(-np.mean((a - 1) * np.log(y) + (b - 1) * np.log1p(-y) - betaln(a, b))))]
    fit_ids = data['split'] == 'train'
    for name, learner in [('logistic', LogisticRegression(max_iter=1000)), ('gradient-boosting', HistGradientBoostingClassifier(random_state=c['seed']))]:
        learner.fit(data['x'][fit_ids], data['y'][fit_ids] < .85)
        p = learner.predict_proba(data['x'][selected])[:, 1]
        quality_metrics.append(dict(method=name, accuracy=accuracy_score(label, p >= .5), macro_f1=f1_score(label, p >= .5, average='macro'), auroc=roc_auc_score(label, p)))
    pd.DataFrame(quality_metrics).to_csv(out / 'quality_metrics.csv', index=False)


def plot(out):
    d = pd.read_csv(out / 'metrics.csv')
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(d.expected_cost_index, d.cvar_index)
    for row in d.itertuples():
        ax.annotate(row.method, (row.expected_cost_index, row.cvar_index))
    ax.set(xlabel='Expected cost / FIFO (%)', ylabel='CVaR cost / FIFO mean (%)')
    fig.tight_layout()
    fig.savefig(out / 'tradeoff.png', dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['prepare', 'fit', 'train', 'run', 'evaluate', 'plot'])
    p.add_argument('--config', default='config.json')
    p.add_argument('--output', default='runs/default')
    p.add_argument('--variant')
    args = p.parse_args()
    c = json.loads(Path(args.config).read_text(encoding='utf-8'))
    if args.variant:
        c['variant'] = args.variant
    if c['variant'] not in ['full', 'point-yield', 'no-genealogy', 'risk-neutral', 'flat', 'mean-hierarchy']:
        p.error('Unknown variant')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    settings = out / 'settings.json'
    if args.variant is None and settings.exists():
        c['variant'] = json.loads(settings.read_text(encoding='utf-8')).get('variant', c.get('variant', 'full'))
    settings.write_text(json.dumps(c, indent=2))
    if args.command == 'plot':
        plot(out)
    else:
        globals()[args.command](c, out)


if __name__ == '__main__':
    main()
