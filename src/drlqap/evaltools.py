import torch
import matplotlib.pyplot as plt
from ipywidgets import interact
from pathlib import Path
from ipywidgets import interact
from natsort import natsorted
from drlqap import agent_configs
import drlqap.experiment

def load_checkpoints(path_prefix: Path):
    agents = []
    metadata = drlqap.experiment.load_metadata(path_prefix)
    agent_constructor = agent_configs.agents[metadata['agent_type']]
    for fname in natsorted(path_prefix.glob("checkpoint_*.pth"), key=lambda p: str(p)):
        print(fname)
        a = agent_constructor()
        a.load_checkpoint(fname)
        a.checkpoint_name = fname.stem
        agents.append(a)

    return agents

def load_float_txt(path: Path):
    try:
        with open(path) as f:
            return [float(line) for line in f.readlines()]
    except FileNotFoundError:
        return []

def plot_embedding(ax, embedding):
    with torch.no_grad():
        ax.clear()
        ax.set_ylim(-1,1)
        ax.bar(range(len(embedding)), embedding)


def interactive_embeddings(embeddings):
    fig = plt.figure()
    ax = fig.gca()
    interact(lambda i: plot_embedding(ax, embeddings[i-1]), i=(1,len(embeddings)))
    fig.show()
    
def interactive_embeddings_series(embeddings_series):
    fig = plt.figure()
    ax = fig.gca()
    interact(lambda i,t: plot_embedding(ax, embeddings_series[t][i]), 
             i=range(len(embeddings_series[0])),
             t=(0,len(embeddings_series)-1)
            )
    fig.show()


def plot_matrix(fig, matrix, **kwargs):
    with torch.no_grad():
        fig.clear()
        ax = fig.gca()
        image = ax.imshow(matrix, **kwargs)
        fig.colorbar(image)

def interactive_matrix_series(series, **kwargs):
    fig = plt.figure()
    interact(lambda i: plot_matrix(fig, series[i], **kwargs), 
             i=(0,len(series)-1),
            )
    fig.show()
