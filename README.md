# YieldRisk

Implementation of **Simulation-based risk-aware semiconductor lot dispatching under yield and operational uncertainty.**

## Installation

Requires Python >=3.10.

```bash
python -m pip install -e .
```

Inspection environment: Python 3.13.5, NumPy 2.1.3 and SciPy 1.15.3. Verification covers Python syntax, JSON/TOML and README CLI options against parser definitions; training and evaluation were not run.

## Data and preprocessing

`prepare` generates chronological quality records and disjoint operating conditions for a reentrant discrete-event simulator with four tool groups, 12 tools and 24 operations. The configured routes, processing/failure distributions, two warmup days and seven scored days define a generated factory benchmark.

```bash
python -m yieldrisk prepare --config config.json --output runs/full
```

## Training and inference

```bash
python -m yieldrisk fit --config config.json --output runs/full
python -m yieldrisk train --config config.json --output runs/full
python -m yieldrisk run --config config.json --output runs/full
```

`config.json` records the method settings and seed `13`; the resolved configuration is saved as `runs/full/settings.json`. Keep separate output directories for different seeds and variants.

## Evaluation

```bash
python -m yieldrisk evaluate --config config.json --output runs/full
```

## Ablations

Supported controls: `full`, `point-yield`, `no-genealogy`, `risk-neutral`, `flat`, `mean-hierarchy`.

Repeat the same data and configuration in a separate directory, for example:

```bash
python -m yieldrisk prepare --config config.json --output runs/point-yield --variant point-yield
python -m yieldrisk fit --config config.json --output runs/point-yield --variant point-yield
python -m yieldrisk train --config config.json --output runs/point-yield --variant point-yield
python -m yieldrisk run --config config.json --output runs/point-yield --variant point-yield
python -m yieldrisk evaluate --config config.json --output runs/point-yield --variant point-yield
```

## Outputs

All main outputs are written below `runs/full/`.

`quality.pt` and `quality_calibration.json` define distributional yield. `policy.pt` stores the attention policy/coordinator/quantile critic. `learning.csv` records observed training cost. `actions.csv` records feasible choices, alternatives, additive logit sums and time. `episodes.csv` supplies complete independent evaluations; `metrics.csv` reports expected/CVaR cost indices, flow, throughput, quality, WIP and timing. `quality_metrics.csv` reports beta NLL/ECE and classification controls.

```bash
python -m yieldrisk plot --config config.json --output runs/full
```
