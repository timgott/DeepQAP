# Deep QAP

This repo contains the code of a Deep Reinforcement Learning agent that is trained to solve [Quadratic Assignment Problems](https://en.wikipedia.org/wiki/Quadratic_assignment_problem) (QAP). 

For details and results see the [thesis paper](./paper/thesis.pdf). As an introduction to the problem and setup, see the [proposal presentation](./paper/presentation-proposal.pdf).

## Setup

Requires python 3.9

Install poetry
```
pip install poetry
```

Setup environment
```
poetry install
poetry shell
```

## Training runs

From within the environment execute either one of the `run_*` scripts or one of the scripts in `scripts/*`.

## Jupyter evaluation notebooks

Plots and small experiments.

Within the environment, run
```
jupyter notebook
```
then open the browser interface and open a notebook in the `evaluation` folder

## Bokeh evaluation server

Shows some information about the trained agent and allows interactive execution of the agents.

```
bokeh serve evaluation/server
```

## Run unit tests

Only cover some critical aspects

```
pytest
```
