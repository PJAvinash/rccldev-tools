import pandas as pd
import plotly.graph_objects as go
import json

# Assume you have the flatten_json_to_dataframe and plot_dataframe_3d_interactive functions defined as above

def flatten_json_to_dataframe(json_data):
    """
    Flattens a specific JSON structure into a Pandas DataFrame.
    (Same function as provided previously)
    """
    if isinstance(json_data, str):
        data = json.loads(json_data)
    else:
        data = json_data

    flat_data = []
    for entry in data:
        commit_hash = entry.get('commit')
        for item in entry.get('data', []):
            row = {
                'commit_hash': commit_hash,
                'size': float(item.get('size')),
                'op_time(us)': float(item.get('op_time(us)', 0.0)),
                'op_algbw(GB/s)': float(item.get('op_algbw(GB/s)', 0.0)),
                'op_busbw(GB/s)': float(item.get('op_busbw(GB/s)', 0.0)),
                'ip_time(us)': float(item.get('ip_time(us)', 0.0)),
                'ip_algbw(GB/s)': float(item.get('ip_algbw(GB/s)', 0.0)),
                'ip_busbw(GB/s)': float(item.get('ip_busbw(GB/s)', 0.0)),
            }
            flat_data.append(row)

    return pd.DataFrame(flat_data)

def plot_dataframe_3d_interactive(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    dropdown_z_cols: list,
    x_axis_title: str = None,
    y_axis_title: str = None,
    plot_title: str = "Interactive 3D Plot",
    connect_points: bool = True,
    log_y: bool = True,
    color_by_deviation: bool = True
):
    import numpy as np
    import plotly.graph_objects as go

    if not all(col in df.columns for col in [x_col, y_col] + dropdown_z_cols):
        raise ValueError("One or more specified columns are not in the DataFrame.")

    x_axis_title = x_axis_title if x_axis_title is not None else x_col
    y_axis_title = y_axis_title if y_axis_title is not None else (f"log({y_col})" if log_y else y_col)
    y_data = np.log2(df[y_col]) if log_y else df[y_col]

    data_traces = []

    for i, col in enumerate(dropdown_z_cols):
        visible = [False] * len(dropdown_z_cols)
        visible[i] = True

        z_data = df[col]
        color_vals = None
        if color_by_deviation:
            # Group by (x, y) and calculate average z
            group_max = df.groupby([y_col])[col].transform("max")
            color_vals = group_max - z_data
        else:
            color_vals = z_data  # or just use a default constant if no coloring

        trace = go.Scatter3d(
            x=df[x_col],
            y=y_data,
            z=z_data,
            mode='lines+markers' if connect_points else 'markers',
            name=col,
            marker=dict(
                size=3,
                color=color_vals,
                colorscale='thermal',
                colorbar=dict(title="Deviation" if color_by_deviation else "Z"),
                showscale=True
            ),
            visible=visible[i]
        )
        data_traces.append(trace)

    buttons = []
    for i, col in enumerate(dropdown_z_cols):
        visibility_settings = [False] * len(dropdown_z_cols)
        visibility_settings[i] = True

        button = dict(
            label=col,
            method="update",
            args=[{"visible": visibility_settings},
                  {"scene.zaxis.title": col, "title.text": f"{plot_title}: {col}"}]
        )
        buttons.append(button)

    fig = go.Figure(data=data_traces)

    fig.update_layout(
        updatemenus=[
            dict(
                type="dropdown",
                direction="down",
                x=0.01,
                xanchor="left",
                y=1.15,
                yanchor="top",
                buttons=buttons,
                showactive=True
            )
        ],
        scene=dict(
            xaxis_title=x_axis_title,
            yaxis_title=y_axis_title,
            zaxis_title=dropdown_z_cols[0]
        ),
        title=plot_title,
        height=700
    )

    return fig

# No need for json or numpy imports inside this function,
# as they are usually handled by the calling script/notebook.
import json

def read_json(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data
df_flattened = flatten_json_to_dataframe(read_json("data.json"))
print(df_flattened)
# 2. Generate the Plotly figure
plot_3d_figure = plot_dataframe_3d_interactive(
    df=df_flattened,
    x_col='commit_hash',
    y_col='size',
    dropdown_z_cols=[
        'op_time(us)', 'op_algbw(GB/s)', 'op_busbw(GB/s)',
        'ip_time(us)', 'ip_algbw(GB/s)', 'ip_busbw(GB/s)'
    ],
    x_axis_title='Commit Hash',
    y_axis_title='Data Size (elements)',
    plot_title='Performance Metrics vs. Commit and Data Size',
    log_y=True
)

# 3. Export to HTML
output_html_file = "interactive_3d_plot.html"
plot_3d_figure.write_html(output_html_file)
print(f"Plot saved to {output_html_file}. Open this file in your web browser to view the interactive plot.")
