from typing import Union
from pathlib import Path
from bokeh.core.property.container import Seq
from bokeh.core.property.factors import Factor
from bokeh.events import Tap

from bokeh.layouts import column, gridplot, row
from bokeh.models import Button, Select, ColumnDataSource, Slider, LinearColorMapper, Div
from bokeh.plotting import curdoc, figure
from bokeh.transform import transform
from bokeh.palettes import Viridis256
import numpy as np
import torch
import bottleneck

from drlqap.evaltools import load_checkpoints, load_float_txt
from drlqap.qap import GraphAssignmentProblem
from drlqap.qapenv import QAPEnv
from drlqap.reinforce import Categorical2D, ReinforceAgent

from matrix import MatrixDataSource

agent_folders = list(Path("runs").iterdir())
qap_files = list(Path("qapdata").glob("*.dat"))

class AgentState:
    def __init__(self, qap: GraphAssignmentProblem, agent: ReinforceAgent) -> None:
        self.agent = agent
        self.env = QAPEnv(qap)
        self.compute_net_state()

    def has_policy_distribution(self):
        return hasattr(self.agent, "get_policy")

    def get_policy(self):
        with torch.no_grad():
            return self.agent.get_policy(self.env.get_state())

    def compute_net_state(self):
        with torch.no_grad():
            qap = self.env.get_state()
            net = self.agent.policy_net
            probs = net(qap)
            self.probs = probs
            self.embeddings = (net.embeddings_a, net.embeddings_b)

    def assignment_step(self, a, b):
        try:
            i = self.env.unassigned_a.index(a)
            j = self.env.unassigned_b.index(b)
        except ValueError:
            print("invalid assignment")
        else:
            self.env.step((i,j))
            self.compute_net_state()
            print(self.env.reward_sum)


# global state
experiment_path = None
checkpoints = None
qap = None
state: AgentState = None

# plot data
training_results = ColumnDataSource(data=dict(episode=[], value=[], entropy=[], gradient=[]))

# matrix data
probability_matrix_source = MatrixDataSource('a', 'b', ['p', 'l'])
node_embedding_sources = (
    MatrixDataSource('i', 'j', ['base']),
    MatrixDataSource('i', 'j', ['base']),
)

# data loaders
def update_experiment(path):
    # Plot values during training
    value = load_float_txt(path / "value.txt")
    value_mean = bottleneck.move_mean(value, window=200)
    value_median = bottleneck.move_median(value, window=200)
    episodes = list(range(0, len(value)))
    training_results.data = dict(
        episode=episodes,
        value=value,
        value_mean=value_mean,
        value_median=value_median
    )

def update_callback():
    update_experiment(experiment_path)

def reset_state():
    global state, qap
    if qap is not None and checkpoints is not None:
        agent = checkpoints[checkpoint_slider.value]
        agent_name_text.text = agent.checkpoint_name
        checkpoint_slider.show_value
        state = AgentState(qap, agent)
        state_updated()

checkpoint_slider = Slider(title="Checkpoint", start=0, end=1, step=1, value=0)
agent_name_text = Div()

# add a load_button widget and configure with the call back
def agent_selected_callback(attr, old_path, path: str):
    global checkpoints, experiment_path
    experiment_path = Path(path)
    checkpoints = load_checkpoints(experiment_path)
    checkpoint_slider.end = len(checkpoints)-1
    update_experiment(experiment_path)
    reset_state()

def path_select(title, paths, callback):
    options = [(str(f), f.name) for f in paths]
    widget = Select(title=title, value=str(paths[0]), options=options)
    widget.on_change("value", callback)
    callback("value", None, widget.value)
    return widget

# Agent selector
agent_dropdown = path_select("Agent", agent_folders, agent_selected_callback)

# Plot training results
value_plot = figure(title="Training assignment values")
value_plot.cross(x='episode', y='value', source=training_results, alpha=0.6)
value_plot.line(x='episode', y='value_median', source=training_results, color="green", line_width=2)
value_plot.line(x='episode', y='value_mean', source=training_results, color="red", line_width=2)

plot_layout = value_plot
probability_figure = probability_matrix_source.create_plot(
    title="Assignment probability", data_column="p", low=0.0, high=1.0
)

logit_figure = probability_matrix_source.create_plot(
    title="Assignment logit probability", data_column="l", 
)

node_embedding_figures = [
    source.create_plot(title=f"Node embeddings of graph {i}")
    for i, source in enumerate(node_embedding_sources)
]

agent_state_layout = column(
    row(probability_figure, logit_figure),
    row(*node_embedding_figures, width=600, height=400),
)

def flatten_tensor(m: torch.Tensor) -> np.ndarray:
    return m.ravel().numpy()

def state_updated():
    nodes_a = state.env.unassigned_a
    nodes_b = state.env.unassigned_b
    with torch.no_grad():
        if state.has_policy_distribution():
            policy = state.get_policy()
            probs = policy.distribution.probs
            log_probs = policy.distribution.logits
        else:
            probs = state.probs
            log_probs = probs

    probability_matrix_source.set_data(a=nodes_a, b=nodes_b, p=probs, l=log_probs)

    for i in (0,1):
        data_source = node_embedding_sources[i]
        data_source.set_data(base=state.embeddings[i])

def reset_click_callback():
    reset_state()

def qap_selected_callback(attr, old_path, path: str):
    global qap
    text = Path(path).read_text()
    qap = GraphAssignmentProblem.from_qaplib_string(text)
    reset_state()

def checkpoint_slider_callback(attr, old_value, new_value):
    reset_state()

qaps_dropdown = path_select("QAPLIB Problem", qap_files, qap_selected_callback)
reset_button = Button(label="Reset", button_type="primary")
reset_button.on_click(reset_click_callback)
checkpoint_slider.on_change("value", checkpoint_slider_callback)

qap_insight_layout = column(qaps_dropdown, row(checkpoint_slider, agent_name_text), reset_button, agent_state_layout)

def matrix_tapped(event):
    b = int(np.round(event.x))
    a = int(np.round(event.y))
    state.assignment_step(a, b)
    state_updated()

probability_figure.on_event(Tap, matrix_tapped)
logit_figure.on_event(Tap, matrix_tapped)

# put the load_button and plot in a layout and add to the document
curdoc().add_root(column(agent_dropdown, plot_layout, qap_insight_layout))
