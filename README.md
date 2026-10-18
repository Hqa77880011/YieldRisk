# YieldRisk

Implementation of **Simulation-based risk-aware semiconductor lot dispatching under yield and operational uncertainty.**

## Install

Use Python 3.11 in an isolated environment. Run from the repository root:

```bash
python -m pip install -e .
```

## Data

`prepare` constructs chronological quality-fitting records and disjoint operating-factor combinations. The simulator has reentrant qualified routes, machine setups/failures, repairs and genealogy-linked rework. `groups,tools,operations,warmup_days,scored_days` set scale. The default is a controlled event model with generated routes/processing distributions; plant-specific routes and distributions belong in the simulator configuration/code. No plant logs are required to run the configured benchmark.

## Workflow

```bash
python -m yieldrisk prepare
python -m yieldrisk fit
python -m yieldrisk train
python -m yieldrisk run
python -m yieldrisk evaluate
python -m yieldrisk plot
```

Main commands accept `--config config.json` and `--output runs/default`. Each output directory records `settings.json`. Keep independent seeds, datasets and conditions in separate directories.

## Configuration

`quality_sizes,quality_epochs` define beta-yield fitting. `train_events,rollout,ppo_epochs,discount` define actor/quantile-critic optimization. `propagation,max_rework` control genealogy exposure. `processing_sigma,setup_hours` define event dynamics. Evaluation uses independent `evaluation_seeds` and frozen test operating combinations. Mean and tail risk weights follow the implemented normalized cost objective. Additional optimizer, batch-size and model-width defaults are in `config.json`.

## Outputs

`quality.pt` and `quality_calibration.json` define distributional yield. `policy.pt` stores the attention policy/coordinator/quantile critic. `learning.csv` records observed training cost. `actions.csv` records feasible choices, alternatives, additive logit sums and time. `episodes.csv` supplies complete independent evaluations; `metrics.csv` reports expected/CVaR cost indices, flow, throughput, quality, WIP and timing. `quality_metrics.csv` reports beta NLL/ECE and classification controls.

## Method and comparisons

Beta features contain mean, 5% quantile and probability below the yield threshold. Rework creates persistent parent/pass nodes; complement-product exposure enters local ranking. A slow coordinator modulates five additive policy groups, while the mask enforces qualification, precedence and nonpreemption. PPO combines mean and tail advantage with a 51-quantile critic. The mean-feature control changes feature input to the same trained policy; an architecture-matched mean-only learner is trained separately with `mean-hierarchy`.

Controls: `full`, `point-yield`, `no-genealogy`, `risk-neutral`, `flat`, `mean-hierarchy`. The run command also includes FIFO, EDD, ATC and a mean-feature intervention. Train each objective/architecture ablation with `--variant`.

```bash
python -m yieldrisk.compare --inputs "runs/*/metrics.csv" --reference YieldRisk --metrics cost cycle_time quality_loss --unit seed
```

Comparison averages rows within independent units before bootstrap intervals and paired Wilcoxon/Holm inference. For participant/client analyses choose those independent unit columns instead of counting recordings as replicates.

## Selected reproduction setting

Use the self-contained single-lot, reentrant DES with four tool groups, 12 tools and 24 operations, two warmup days and seven scored days. Quality fitting uses 2048/512/512 generated records; the default learning budget is 50000 events with rollout 512 and 20 quality epochs. Qualification, precedence and nonpreemption are the chosen hard-rule set; batch tools and engineering holds are outside this selected factory. No SMT2020 dataset is required. Use the documented prepare/fit/train/run/evaluate workflow and the same evaluation seeds for all methods.
