from pathlib import Path
from bokeh.layouts import column, row
from bokeh.models import Button, Select, ColumnDataSource, Slider, Div
from bokeh.plotting import curdoc, figure
import bottleneck
from bokeh.layouts import column, row
from bokeh.models import Button, Select, ColumnDataSource, Slider, Div
from bokeh.events import Tap
import numpy as np

import drlqap.taskgenerators
from drlqap.evaltools import load_checkpoints, load_float_txt
from matrix import create_matrix_plot
from agentstate import AgentStateViewModel

runs_root = Path("runs")
agent_folders = list(p.parent for p in runs_root.glob("**/agenttype"))
qap_generators = drlqap.taskgenerators.generators
qap_generator_names = list(qap_generators.keys())

# global state
experiment_path = None
checkpoints = None
agent_state = AgentStateViewModel(curdoc())

# plot data
training_results = ColumnDataSource(data=dict(episode=[], value=[], entropy=[], gradient=[]))

# data loaders
def update_experiment_data(path):
    # Plot values during training
    value = load_float_txt(path / "value.txt")
    window = 500
    value_mean = bottleneck.move_mean(value, window=window)
    value_median = bottleneck.move_median(value, window=window)
    episodes = list(range(0, len(value)))
    training_results.data = dict(
        episode=episodes,
        value=value,
        value_mean=value_mean,
        value_median=value_median
    )

def update_selected_agent():
    agent = checkpoints[checkpoint_slider.value]
    agent_name_text.text = agent.checkpoint_name
    agent_state.set_agent(agent)

# add a load_button widget and configure with the call back
def agent_selected_callback(attr, old_path, path: str):
    global checkpoints, experiment_path, checkpoint_slider
    experiment_path = Path(path)
    checkpoints = load_checkpoints(experiment_path)
    checkpoint_slider.end = len(checkpoints)-1
    checkpoint_slider.value = checkpoint_slider.end
    update_experiment_data(experiment_path)
    update_selected_agent()

def path_select(title, paths, callback, root):
    options = [(str(f), str(f.relative_to(root))) for f in paths]
    widget = Select(title=title, value=str(paths[0]), options=options)
    widget.on_change("value", callback)
    return widget

# Agent selector
agent_dropdown = path_select("Agent", agent_folders, agent_selected_callback, root=runs_root)

# Plot training results
value_plot = figure(title="Training assignment values", width=1000, height=600)
value_plot.cross(x='episode', y='value', source=training_results, alpha=0.6)
value_plot.line(x='episode', y='value_median', source=training_results, color="green", line_width=2)
value_plot.line(x='episode', y='value_mean', source=training_results, color="red", line_width=2)

plot_layout = value_plot

# Agent tester
def qap_selected_callback(attr, old_name, name: str):
    if name:
        qap = drlqap.taskgenerators.generators[name].sample()
        agent_state.set_qap(qap)

def checkpoint_selected_callback(attr, old_name, name: str):
    update_selected_agent()

checkpoint_slider = Slider(title="Checkpoint", start=0, end=1, step=1, value=0)
agent_name_text = Div()
checkpoint_slider.on_change("value", checkpoint_selected_callback)

qaps_dropdown = Select(title="QAPLIB Problem", options=qap_generator_names, value=qap_generator_names[0])
qaps_dropdown.on_change("value", qap_selected_callback)

reset_button = Button(label="Reset", button_type="primary")
reset_button.on_click(lambda: agent_state.reset_state())

probability_figure = create_matrix_plot(
    agent_state.probability_matrix_source,
    title="Assignment probability", data_column="p", low=0.0, high=1.0
)

logit_figure = create_matrix_plot(
    agent_state.probability_matrix_source,
    title="Assignment logit probability", data_column="l", 
)

estimated_value_figure = create_matrix_plot(
    agent_state.qap_viewmodel.estimated_value_source,
    title="Estimated stochastic value of assignment", data_column="average", tooltip_columns=['minimum']
)


node_embedding_figures = [
    create_matrix_plot(source, title=f"Node embeddings of graph {i}")
    for i, source in enumerate(agent_state.node_embedding_sources)
]

qap_A_figure = create_matrix_plot(
    agent_state.qap_viewmodel.matrix_a_source,
    title="QAP A weights", data_column="A", 
)

qap_B_figure = create_matrix_plot(
    agent_state.qap_viewmodel.matrix_b_source,
    title="QAP B weights", data_column="B", 
)

qap_L_figure = create_matrix_plot(
    agent_state.qap_viewmodel.linear_costs_source,
    title="QAP linear weights", data_column="L", 
)

agent_state_layout = column(
    row(probability_figure, logit_figure, estimated_value_figure, sizing_mode='scale_width'),
    row(qap_A_figure, qap_B_figure, qap_L_figure, sizing_mode='scale_width'),
    row(*node_embedding_figures, sizing_mode='stretch_both'),
    sizing_mode='stretch_width'
)

qap_insight_layout = column(
    qaps_dropdown,
    row(checkpoint_slider, agent_name_text),
    reset_button,
    agent_state_layout,
    sizing_mode='stretch_width'
)

def matrix_tapped(event):
    b = int(np.round(event.x))
    a = int(np.round(event.y))
    agent_state.assignment_step(a, b)

probability_figure.on_event(Tap, matrix_tapped)
logit_figure.on_event(Tap, matrix_tapped)
# put the load_button and plot in a layout and add to the document
curdoc().add_root(column(agent_dropdown, plot_layout, qap_insight_layout, sizing_mode='stretch_width', max_width=1200))

# Select first agent
qap_selected_callback("value", None, qaps_dropdown.value)
agent_selected_callback("value", None, agent_dropdown.value)
