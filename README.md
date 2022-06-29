# Bachelor Thesis Tim Göttlicher

Using Deep Reinforcement learning to solve Quadratic Assignment Problems.

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
