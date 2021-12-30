from typing import Union
from pathlib import Path
from bokeh.core.property.container import Seq
from bokeh.core.property.factors import Factor
from bokeh.events import Tap

from bokeh.layouts import column, gridplot, row
from bokeh.models import Button, Select, ColumnDataSource, Slider, LinearColorMapper, Div
from bokeh.models.tickers import SingleIntervalTicker
from bokeh.plotting import curdoc, figure
from bokeh.transform import transform
from bokeh.palettes import Viridis256
import numpy as np
import torch
from torch._C import dtype
from torch.distributions.categorical import Categorical

from drlqap.evaltools import load_checkpoints, load_float_txt
from drlqap.qap import GraphAssignmentProblem
from drlqap.qapenv import QAPEnv
from drlqap.reinforce import Categorical2D, ReinforceAgent

agent_folders = list(Path("runs").iterdir())
qap_files = list(Path("qapdata").glob("*.dat"))

class AgentState:
    def __init__(self, qap: GraphAssignmentProblem, agent: ReinforceAgent) -> None:
        self.agent = agent
        self.env = QAPEnv(qap)
        self.compute_net_state()

    def get_policy(self):
        with torch.no_grad():
            return self.agent.get_policy(self.env.get_state())

    def compute_net_state(self):
        with torch.no_grad():
            net = self.agent.policy_net
            hdata = net.initial_transformation(self.env.get_state())
            self.net_base_state = (hdata.x_dict['a'], hdata.x_dict['b'])
            if net.message_passing_net:
                node_dict = net.message_passing_net(hdata.x_dict, hdata.edge_index_dict, hdata.edge_attr_dict)
            else:
                node_dict = hdata.x_dict
            self.net_state = (node_dict['a'], node_dict['b'])

    def assignment_step(self, a, b):
        try:
            i = self.env.unassigned_a.index(a)
            j = self.env.unassigned_b.index(b)
        except ValueError:
            print("invalid assignment")
        else:
            self.env.step((i,j))
            self.compute_net_state()


# global state
experiment_path = None
checkpoints = None
qap = None
state: AgentState = None

# plot data
training_results = ColumnDataSource(data=dict(episode=[], value=[], entropy=[], gradient=[]))

# matrix data
probability_matrix_source = ColumnDataSource(dict(a=[], b=[], p=[], l=[]))
node_embedding_sources = (
    ColumnDataSource(dict(i=[], j=[], base=[], mp=[])),
    ColumnDataSource(dict(i=[], j=[], base=[], mp=[]))
)

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
value_plot.scatter(x='episode', y='value', source=training_results)

entropy_plot = figure(title="Average policy entropy per episode during training")
entropy_plot.scatter(x='episode', y='entropy', source=training_results)

plot_layout = column(value_plot, entropy_plot)

# QAP test runs
def create_matrix_plot(title, i_column, j_column, value_column, source, low=None, high=None):
    colormap = LinearColorMapper(Viridis256, low=low, high=high)
    fig = figure(
        title=title, 
        tools="hover",
        toolbar_location=None,
        tooltips=[(value_column, "@" + value_column), (i_column, "@" + i_column), (j_column, "@" + j_column)],
        x_axis_location="above",
        x_axis_label=j_column,
        y_axis_label=i_column
    )

    fig.axis.ticker = SingleIntervalTicker(interval=1, num_minor_ticks=0)
    fig.x_range.range_padding = 0
    fig.y_range.range_padding = 0
    fig.y_range.flipped = True
    fig.grid.visible = True

    fig.rect(
        x=j_column, y=i_column,
        color=transform(value_column, colormap), 
        source=source,
        width=1, height=1,
    )

    return fig

probability_figure = create_matrix_plot(
    title="Assignment probability",
    i_column="a", j_column="b", value_column="p", 
    source=probability_matrix_source, low=0.0, high=1.0
)

logit_figure = create_matrix_plot(
    title="Assignment logit probability",
    i_column="a", j_column="b", value_column="l", 
    source=probability_matrix_source
)

node_embedding_figures = [
    (
        create_matrix_plot(
            title=f"Node embeddings of graph {i} before message passing", 
            i_column='i', j_column='j', value_column='base',
            source=source
        ),
        create_matrix_plot(
            title=f"Node embeddings of graph {i}", 
            i_column='i', j_column='j', value_column='mp',
            source=source),
    )
    for i, source in enumerate(node_embedding_sources)
]

agent_state_layout = column(
    row(probability_figure, logit_figure),
    gridplot(node_embedding_figures, width=600, height=400),
)

def flatten_tensor(m: torch.Tensor) -> np.ndarray:
    return m.ravel().numpy()

def update_probability_matrix():
    if not state.env.unassigned_a:
        probability_matrix_source.data = dict(
            p = [],
            l = [],
            a = [],
            b = []
        )
    else:
        nodes_a = np.array(state.env.unassigned_a)
        nodes_b = np.array(state.env.unassigned_b)
        with torch.no_grad():
            policy = state.get_policy()
            probs = policy.distribution.probs.numpy()
            log_probs = policy.distribution.logits.numpy()
            indices = np.indices(policy.shape)
        
        probability_matrix_source.data = dict(
            p = probs,
            l = log_probs,
            a=nodes_a[indices[0].ravel()],
            b=nodes_b[indices[1].ravel()]
        )

def update_node_embedding_matrix(base_embeddings, mp_embeddings, data_source):
    indices = np.indices(base_embeddings.shape)

    data_source.data =  dict(
        base = flatten_tensor(base_embeddings),
        mp = flatten_tensor(mp_embeddings),
        i = indices[0].ravel(),
        j = indices[1].ravel(),
    )

def state_updated():
    update_probability_matrix()
    for i in (0,1):
        update_node_embedding_matrix(
            state.net_base_state[i], state.net_state[i],
            data_source=node_embedding_sources[i]
        )

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
