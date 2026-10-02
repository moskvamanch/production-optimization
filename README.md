# Simulation-Based Optimisation of a Stochastic Production Line

A discrete-event simulation study of a multi-stage wooden-chair production line. The project investigates how simulation-based optimisation methods allocate a limited capacity budget across intermediate buffers when the objective can only be observed through noisy simulation runs.

## Overview

The model is implemented in [SimPy](https://simpy.readthedocs.io/) and represents the following process:

```
Arrival → Cutting → Drilling → Sanding → Assembly → Batch Painting
        → Drying → Quality Control → Packaging
```

The production line includes finite intermediate buffers, stochastic processing and arrival times, batch painting, quality control, rework, and scrapping. The five buffer capacities are integer decision variables.

## Optimisation problem

For a buffer-capacity vector \(b = (b_1, \ldots, b_5)\), the simulator estimates average throughput \(\bar{T}(b)\). The optimisation objective balances throughput against installed buffer capacity:

\[
J(b) = \alpha\,\bar{T}(b) - \sum_{i=1}^{5} b_i.
\]

Repeated replications are used to reduce simulation noise. The number of objective-function evaluations, rather than iterations alone, is treated as the main computational budget when methods are compared.

## Methods

| Method | Role in the study |
| --- | --- |
| Grid Search | Exhaustive reference search in a restricted space |
| Finite Difference | Local search using numerical gradient estimates |
| Bayesian Optimisation | Gaussian-process surrogate model with Expected Improvement |
| Particle Swarm Optimisation | Population-based, derivative-free search |

The project includes both a two-dimensional diagnostic setting, where buffers 3 and 4 are varied, and the full five-dimensional optimisation problem.

## Repository structure

```
src/
  simulation/       # SimPy production-line model
  optimization/     # Grid Search, Finite Difference, BO and PSO
experiments/        # Reproducible experiment entry points
docs/               # Method documentation and selected figures
results/            # Locally generated outputs (not committed)
```

## Installation

```bash
git clone https://github.com/moskvamanch/production-optimization.git
cd production-optimization
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
python -m pip install -r requirements.txt
```

The optimiser modules expose their experiment configuration in their `__main__` blocks. Generated CSV files and figures are written to `results/`.

## Research focus

This repository focuses on the practical trade-offs of simulation optimisation: solution quality, simulation cost, stochastic variability, integer decision variables, and the difference between local and global search. It does not assume that one method is universally best; performance depends on the evaluation budget and the structure of the search space.

## Documentation

A Bayesian Optimisation flowchart is available in [docs/bo_algorithm_flowchart.pdf](docs/bo_algorithm_flowchart.pdf).
