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

        param_name = filename.stem[len("gradient_"):]
        layer_name, type = param_name.rsplit('.', maxsplit = 1)

        if layer_name not in mean_gradients:
            mean_gradients[layer_name] = dict()

        mean_gradients[layer_name][type] = [np.mean(arr) for arr in segments]

    return mean_gradients


def generate_gradient_grid(layer_data):
    plt.rc('axes', titlesize=6)

    columns = 8
    rows = len(layer_data) // columns + 1
    print(columns, " x ", rows)

    fig, axs = plt.subplots(rows, columns, sharey=True, sharex=True, squeeze=False, figsize=(25,25))

    for i, (name, gradients_data) in enumerate(layer_data.items()):
        row = i // columns
        col = i % columns
        ax = axs[row, col]

        for param, gradients in gradients_data.items():
            ind = range(len(gradients))
            ax.fill_between(ind, gradients, label=param, alpha=0.5)

        for param, gradients in gradients_data.items():
            ind = range(len(gradients))
            ax.plot(ind, gradients, label=param)

        ax.set_title(name, wrap=True, pad=0.8)
        ax.set_xticks([])

        if i == 0:
            ax.legend()

    for i in range(len(layer_data), rows * columns):
        row = i // columns
        col = i % columns
        axs[row, col].axis('off')

    fig.tight_layout(h_pad=2)

    return fig

data = load_gradients(Path(sys.argv[1]))
fig = generate_gradient_grid(data)
plt.show()
