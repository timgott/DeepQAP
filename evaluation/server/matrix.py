from bokeh.models.sources import ColumnDataSource
import numpy as np
from bokeh.plotting import figure
from bokeh.transform import transform
from bokeh.palettes import Viridis256
from bokeh.models import LinearColorMapper
from bokeh.models.tickers import SingleIntervalTicker

class MatrixDataSource():
    def __init__(self, i_column, j_column, value_columns):
        super().__init__()

        empty_data = dict()
        empty_data[i_column] = []
        empty_data[j_column] = []
        for column in value_columns:
            empty_data[column] = []

        self.i_column = i_column
        self.j_column = j_column
        self.value_columns = value_columns
        self.data_source = ColumnDataSource(data=empty_data)
        self.empty_data = empty_data

    def set_data(self, **matrices):
        first_matrix = matrices[self.value_columns[0]]
        shape = np.shape(first_matrix)

        if sum(shape) == 0:
            # empty matrix
            self.data_source.data = self.empty_data
            return

        indices = np.indices(shape)
        i = indices[0].ravel()
        j = indices[1].ravel()

        data = dict()
        for name, matrix in matrices.items():
            array = np.array(matrix)
            if name == self.i_column:
                assert array.shape == (shape[0],)
                i = array[i]
            elif name == self.j_column:
                assert array.shape == (shape[1],)
                j = array[j]
            else:
                assert array.shape == shape
                data[name] = array.flatten()

        data[self.i_column] = i
        data[self.j_column] = j

        self.data_source.data = data


def create_matrix_plot(source: MatrixDataSource, title, data_column=None, low=None, high=None, tooltip_columns=[]):
    if len(source.value_columns) == 1:
        data_column = source.value_columns[0]
    elif data_column not in source.value_columns:
        raise ValueError(f"data_column={data_column} not in {source.value_columns}")

    colormap = LinearColorMapper(Viridis256, low=low, high=high)
    fig = figure(
        title=title, 
        tools="hover",
        toolbar_location=None,
        tooltips=[
            (col, "@" + col) 
            for col in [data_column, *tooltip_columns, source.i_column, source.j_column]
        ],
        x_axis_location="above",
        x_axis_label=source.j_column,
        y_axis_label=source.i_column
    )

    fig.axis.ticker = SingleIntervalTicker(interval=1, num_minor_ticks=0)
    fig.x_range.range_padding = 0
    fig.y_range.range_padding = 0
    fig.y_range.flipped = True
    fig.grid.visible = True

    fig.rect(
        x=source.j_column, y=source.i_column,
        color=transform(data_column, colormap), 
        source=source.data_source,
        width=1, height=1,
    )

    return fig

