
from pathlib import Path
from bokeh.core.property.container import Seq
from bokeh.core.property.factors import Factor
from bokeh.events import Tap

from bokeh.layouts import column, row
from bokeh.models import Button, Select, ColumnDataSource, Slider, LinearColorMapper, Div
from bokeh.models.tickers import SingleIntervalTicker
from bokeh.plotting import curdoc, figure
from bokeh.transform import transform
from bokeh.palettes import Viridis256
import numpy as np
import torch
from torch.distributions.categorical import Categorical

from drlqap.evaltools import load_checkpoints, load_float_txt
from drlqap.qap import GraphAssignmentProblem
from drlqap.reinforce import Categorical2D, ReinforceAgent

agent_folders = list(Path("runs").iterdir())
qap_files = list(Path("qapdata").glob("*.dat"))

class AgentState:
    def __init__(self, qap: GraphAssignmentProblem, agent: ReinforceAgent) -> None:
        self.agent = agent
        self.unassigned_a = list(qap.graph_source.nodes)
        self.unassigned_b = list(qap.graph_target.nodes)
        with torch.no_grad():
            self.net_state = agent.policy_net.initial_step(qap)

    def assignment_step(self, a, b):
        self.unassigned_a.remove(a)
        self.unassigned_b.remove(b)
        self.net_state = self.agent.policy_net.assignment_step(self.net_state, a, b)

# global state
experiment_path = None
checkpoints = None
qap = None
state: AgentState = None

# plot data
training_results = ColumnDataSource(data=dict(episode=[], value=[], entropy=[], gradient=[]))

# matrix data
probability_matrix_source = ColumnDataSource(dict(a=[], b=[], p=[]))

# data loaders
def update_experiment(path):
    # Plot values during training
    values = load_float_txt(path / "value.txt")
    entropies = load_float_txt(path / "entropy_average.txt")
    gradients = load_float_txt(path / "gradient_magnitude.txt")
    episodes = list(range(0, len(values)))
    training_results.data = dict(episode=episodes, value=values, entropy=entropies, gradient=gradients)

def update_callback():
    update_experiment(experiment_path)

def update_matrices():
    update_probability_matrix()

def reset_state():
    global state, qap
    if qap is not None and checkpoints is not None:
        agent = checkpoints[checkpoint_slider.value]
        agent_name_text.text = agent.checkpoint_name
        checkpoint_slider.show_value
        state = AgentState(qap, agent)
        update_matrices()

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
value_plot.scatter(x='episode', y='value', source=training_results)

entropy_plot = figure(title="Average policy entropy per episode during training")
entropy_plot.scatter(x='episode', y='entropy', source=training_results)

plot_layout = column(value_plot, entropy_plot)

# QAP test runs
colormap = LinearColorMapper(Viridis256, low=0.0, high=150.0)
probability_figure = figure(
    title="Assignment probability", 
    tools="hover",
    toolbar_location=None,
    tooltips=[("P", "@p"), ("a", "@a"), ("b", "@b")],
    x_axis_location="above",
    x_axis_label="b",
    y_axis_label="a"
)

probability_figure.axis.ticker = SingleIntervalTicker(interval=1, num_minor_ticks=0)
probability_figure.x_range.range_padding = 0
probability_figure.y_range.range_padding = 0
probability_figure.y_range.flipped = True
probability_figure.grid.visible = False

probability_figure.rect(
    x="b", y="a", # flipped to make it look like matrix notation
    color=transform("p", colormap), 
    source=probability_matrix_source,
    width=1, height=1
)


def update_probability_matrix():
    if not state.unassigned_a:
        probability_matrix_source.data = dict(
            p = [],
            a = [],
            b = []
        )
    else:
        nodes_a = np.array(state.unassigned_a)
        nodes_b = np.array(state.unassigned_b)
        with torch.no_grad():
            logits = state.agent.policy_net.compute_link_probabilities(
                state.net_state,
                nodes_a,
                nodes_b
            ).squeeze()
            policy = Categorical2D(logits=logits)
            probs = policy.distribution.probs.numpy()
            log_probs = logits.ravel().numpy()
            indices = np.indices(logits.shape)
        data = dict(
            p = log_probs,
            a=nodes_a[indices[0].ravel()],
            b=nodes_b[indices[1].ravel()]
        )
        probability_matrix_source.data = data

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
qap_insight_layout = column(qaps_dropdown, row(checkpoint_slider, agent_name_text), reset_button, probability_figure)

def matrix_tapped(event):
    b = int(np.round(event.x))
    a = int(np.round(event.y))
    state.assignment_step(a, b)
    update_probability_matrix()

probability_figure.on_event(Tap, matrix_tapped)

# put the load_button and plot in a layout and add to the document
curdoc().add_root(column(agent_dropdown, plot_layout, qap_insight_layout))