import sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def load_float_txt(path: Path):
    with open(path) as f:
        return [float(line) for line in f.readlines()]

def load_gradients(path: Path):
    mean_gradients = dict()
    for i, filename in enumerate(sorted(path.glob("gradient_*"))):
        gradients = load_float_txt(filename)
        segments = np.array_split(gradients, 20)
        name = filename.stem[len("gradient_"):]
        mean_gradients[name] = [np.mean(arr) for arr in segments]

    return mean_gradients


def generate_gradient_grid(gradients_data):
    plt.rc('axes', titlesize=6)

    columns = 8
    rows = len(gradients_data) // columns + 1
    print(columns, " x ", rows)

    fig, axs = plt.subplots(rows, columns, sharey=True, sharex=True, squeeze=False, figsize=(25,25))

    for i, (name, gradients) in enumerate(gradients_data.items()):
        row = i // columns
        col = i % columns
        ind = range(len(gradients))

        axs[row, col].fill_between(ind, gradients)
        axs[row, col].set_title(name, wrap=True, pad=0.8)
        axs[row, col].set_xticks([])

    for i in range(len(gradients_data), rows * columns):
        row = i // columns
        col = i % columns
        axs[row, col].axis('off')

    fig.tight_layout(h_pad=2)

    return fig

data = load_gradients(Path(sys.argv[1]))
fig = generate_gradient_grid(data)
plt.show()
