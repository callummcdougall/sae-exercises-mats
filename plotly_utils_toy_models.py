from copy import deepcopy
import torch as t
from torch import Tensor
from typing import List, Union, Optional
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Tuple, List
from jaxtyping import Float
import einops

from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation

Arr = np.ndarray


def plot_correlated_features(batch: Float[Tensor, "instances batch_size feats"], title: str):
    go.Figure(
        data=[
            go.Bar(y=batch.squeeze()[:, 0].tolist(), name="Feature 0"),
            go.Bar(y=batch.squeeze()[:, 1].tolist(), name="Feature 1"),
        ],
        layout=dict(
            template="simple_white", title=title,
            bargap=0.3, xaxis=dict(tickmode="array", tickvals=list(range(batch.squeeze().shape[0]))),
            xaxis_title="Pair of features", yaxis_title="Feature Values",
            height=400, width=1000,
        )
    ).show()


def plot_features_in_Nd(
    values: Float[Tensor, "instances d_hidden feats"],
    height: int,
    width: int,
    show_wtw: bool = True,
):
    n_instances, d_hidden, n_feats = values.shape

    W = values.detach().cpu()

    W_norm = W / (1e-5 + t.linalg.norm(W, 2, dim=1, keepdim=True))

    # We get interference[i, j] = sum_{j!=i} (W_norm[i] @ W[j]) (ignoring the instance dimension)
    # because then we can calculate superposition by squaring & summing this over j
    interference = einops.einsum(
        W_norm, W,
        "instances hidden feats_i, instances hidden feats_j -> instances feats_i feats_j"
    )
    interference[:, range(n_feats), range(n_feats)] = 0

    # Now take the sum, and sqrt (we could just as well not sqrt)
    # Heuristic: polysemanticity is zero if it's orthogonal to all else, one if it's perfectly aligned with any other single vector
    polysemanticity = einops.reduce(
        interference.pow(2),
        "instances feats_i feats_j -> instances feats_i",
        "sum",
    ).sqrt()

    # Get the norms (this is the bar height)
    norms = einops.reduce(
        W.pow(2),
        "instances hidden feats -> instances feats",
        "sum",
    ).sqrt()

    # We need W.T @ W for the heatmap (unless show_wtw=False, then we just use w)
    if show_wtw:
        WtW = einops.einsum(W, W, "instances hidden feats_i, instances hidden feats_j -> instances feats_i feats_j")
        data = WtW.numpy()
    else:
        data = einops.rearrange(W, "instances hidden feats -> instances feats hidden").numpy()

    x = t.arange(n_feats)

    fig = make_subplots(
        rows = n_instances,
        cols = 2,
        shared_xaxes = True,
        vertical_spacing = 0.02,
        horizontal_spacing = 0.1,
        column_widths = [0.75, 0.25],
    )
    for inst in range(n_instances):
        fig.add_trace(
            go.Bar(
                x=x, 
                y=norms[inst],
                marker=dict(
                    color=polysemanticity[inst],
                    cmin=0,
                    cmax=1
                ),
                width=0.9,
            ),
            row = 1+inst, 
            col = 1,
        )
        fig.add_trace(
            go.Image(
                z=plt.cm.coolwarm((1 + data[inst]) / 2, bytes=True),
                colormodel='rgba256',
                customdata=data[inst],
                hovertemplate='''\
    In: %{x}<br>
    Out: %{y}<br>
    Weight: %{customdata:0.2f}
    '''            
            ),
            row=1+inst, col=2
        )

    fig.add_vline(
      x=d_hidden-0.5, 
      line=dict(width=0.5),
      col=1,
    )
      
    # fig.update_traces(marker_size=1)
    fig.update_layout(
        showlegend=False, 
        width=width,
        height=height,
        margin=dict(t=40, b=40)
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.show()


def helper_get_viridis(v):
    r, g, b, a = plt.get_cmap('viridis')(v)
    return (r, g, b)



def parse_colors_for_superposition_plot(
    colors: Optional[Union[Tuple[int, int], Float[Tensor, "instances feats"]]],
    n_instances: int,
    n_feats: int,
) -> List[List[str]]:
    '''
    There are lots of different ways colors can be represented in the superposition plot.
    
    This function unifies them all by turning colors into a list of lists of strings, i.e. one color for each instance & feature.
    '''
    # If colors is a tensor, we assume it's the importances tensor, and we color according to a viridis color scheme
    if isinstance(colors, Tensor):
        colors = t.broadcast_to(colors, (n_instances, n_feats))
        colors = [
            [helper_get_viridis(v.item()) for v in colors_for_this_instance]
            for colors_for_this_instance in colors
        ]
    
    # If colors is a tuple of ints, it's interpreted as number of correlated / anticorrelated pairs
    elif isinstance(colors, tuple):
        n_corr, n_anti = colors
        n_indep = n_feats - 2 * (n_corr - n_anti)
        colors = [
            ["blue", "blue", "limegreen", "limegreen", "purple", "purple"][:n_corr*2] + ["red", "red", "orange", "orange", "brown", "brown"][:n_anti*2] + ["black"] * n_indep
            for _ in range(n_instances)
        ]
    
    # If colors is a string, make all datapoints that color
    elif isinstance(colors, str):
        colors = [[colors] * n_feats] * n_instances
    
    # Lastly, if colors is None, make all datapoints black
    elif colors is None:
        colors = [["black"] * n_feats] * n_instances
    
    return colors





def plot_features_in_2d(
    values: Float[Tensor, "timesteps instances d_hidden feats"],
    colors = None, # shape [timesteps instances feats]
    title: Optional[str] = None,
    subplot_titles: Optional[List[str]] = None,
    save: Optional[str] = None,
):
    '''
    Visualises superposition in 2D.

    If values is 4D, the first dimension is assumed to be timesteps, and an animation is created.
    '''
    # Convert values to 4D for consistency
    if values.ndim == 3:
        values = values.unsqueeze(0)
    values = values.transpose(-1, -2)
    
    # Get dimensions
    n_timesteps, n_instances, n_features, _ = values.shape

    # If we have a large number of features per plot (i.e. we're plotting projections of data) then use smaller lines
    linewidth, markersize = (1, 4) if (n_features >= 25) else (2, 10)
    
    # Convert colors to 3D, if it's 2D (i.e. same colors for all instances)
    if isinstance(colors, list) and isinstance(colors[0], str):
        colors = [colors for _ in range(n_instances)]
    # Convert colors to something which has 4D, if it is 3D (i.e. same colors for all timesteps)
    if any([
        colors is None,
        isinstance(colors, list) and isinstance(colors[0], list) and isinstance(colors[0][0], str),
        (isinstance(colors, Tensor) or isinstance(colors, Arr)) and colors.ndim == 3,
    ]):
        colors = [colors for _ in range(values.shape[0])]
    # Now that colors has length `timesteps` in some sense, we can convert it to lists of strings
    colors = [parse_colors_for_superposition_plot(c, n_instances, n_features) for c in colors]

    # Same for subplot titles & titles
    if subplot_titles is not None:
        if isinstance(subplot_titles, list) and isinstance(subplot_titles[0], str):
            subplot_titles = [subplot_titles for _ in range(values.shape[0])]
    if title is not None:
        if isinstance(title, str):
            title = [title for _ in range(values.shape[0])]

    # Create a figure and axes
    fig, axs = plt.subplots(1, n_instances, figsize=(5 * n_instances, 5))
    if n_instances == 1:
        axs = [axs]
    
    # If there are titles, add more spacing for them
    fig.subplots_adjust(bottom=0.2, top=0.9, left=0.1, right=0.9)
    if title:
        fig.subplots_adjust(top=0.8)
    
    # Initialize lines and markers
    lines = []
    markers = []
    for instance_idx, ax in enumerate(axs):
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal', adjustable='box')
        instance_lines = []
        instance_markers = []
        for feature_idx in range(n_features):
            line, = ax.plot([], [], color=colors[0][instance_idx][feature_idx], lw=linewidth)
            marker, = ax.plot([], [], color=colors[0][instance_idx][feature_idx], marker='o', markersize=markersize)
            instance_lines.append(line)
            instance_markers.append(marker)
        lines.append(instance_lines)
        markers.append(instance_markers)

    def update(val):
        # I think this doesn't work unless I at least reference the nonlocal slider object
        # It works if I use t = int(val), so long as I put something like X = slider.val first. Idk why!
        if n_timesteps > 1:
            _ = slider.val
        t = int(val) 
        for instance_idx in range(n_instances):
            for feature_idx in range(n_features):
                x, y = values[t, instance_idx, feature_idx].tolist()
                lines[instance_idx][feature_idx].set_data([0, x], [0, y])
                markers[instance_idx][feature_idx].set_data(x, y)
                lines[instance_idx][feature_idx].set_color(colors[t][instance_idx][feature_idx])
                markers[instance_idx][feature_idx].set_color(colors[t][instance_idx][feature_idx])
            if title:
                fig.suptitle(title[t], fontsize=15)
            if subplot_titles:
                axs[instance_idx].set_title(subplot_titles[t][instance_idx], fontsize=12)
        fig.canvas.draw_idle()
    
    def play(event):
        print("Play!")
        _ = slider.val
        for i in range(n_timesteps):
            update(i)
            slider.set_val(i)
            plt.pause(0.05)
        fig.canvas.draw_idle()

    if n_timesteps > 1:
        # Create the slider
        ax_slider = plt.axes([0.15, 0.05, 0.7, 0.05], facecolor='lightgray')
        slider = Slider(ax_slider, 'Time', 0, n_timesteps - 1, valinit=0, valfmt='%1.0f')

        # Create the play button
        # ax_button = plt.axes([0.8, 0.05, 0.08, 0.05], facecolor='lightgray')
        # button = Button(ax_button, 'Play')

        # Call the update function when the slider value is changed / button is clicked
        slider.on_changed(update)
        # button.on_clicked(play)

        # Initialize the plot
        play(0)
    else:
        update(0)

    # Save
    if isinstance(save, str):
        ani = FuncAnimation(fig, update, frames=n_timesteps, interval=0.04, repeat=False)
        ani.save(save, writer='pillow', fps=25)

    plt.show()




# def visualise_2d_superposition_over_time(
#     data_for_plotting: List[dict],
#     width: Optional[int] = None,
#     height: Optional[int] = None,
# ):
#     '''
#     Does the same as the function below, but has a different input type: a list of dictionaries containing "values", "colors", "steps" and "title".

#     Returns an animated plot showing progression over time.
#     '''
#     n_instances = len(data_for_plotting[0]["values"])
#     master_fig = make_subplots(rows=1, cols=n_instances)
#     steps = []
#     fig_count = len(data_for_plotting)

#     for i, data in enumerate(data_for_plotting):

#         # Create the figure for this timestep, and add all traces to the master figure
#         fig = visualise_2d_superposition(
#             values = data["values"],
#             colors = data["colors"],
#             subplot_titles = data.get("subplot_titles", None),
#             show = False,
#         )
#         # fig.show()
#         # Calculate the total number of traces we'll have by the end
#         if i == 0:
#             trace_total = len(fig.data) * fig_count
        
#         # Add the trace, also calculating the index of the first & last trace so we can set step visibility
#         pre_trace_idx = len(master_fig.data)
#         for trace in fig.data:
#             new_trace = deepcopy(trace)
#             new_trace.visible = i == 0
#             new_trace.hoverinfo = "none"
#             col = max(1, int("0" + new_trace["xaxis"].strip("x")))
#             master_fig.add_trace(new_trace, row=1, col=col)
#         post_trace_idx = len(master_fig.data)

#         # Add the slider step: visible iff the trace index is between pre and post
#         step = dict(
#             method = "update",
#             args = [
#                 {"visible": [pre_trace_idx <= trace_idx < post_trace_idx for trace_idx in range(trace_total)]},
#                 {"title": data["title"]},
#             ],
#             # label = data["title"],
#         )
#         steps.append(step)

#     # Add the slider
#     sliders = [dict(
#         active = 0,
#         currentvalue = {"prefix": "Step: "},
#         pad = {"t": 50},
#         steps = steps,
#     )]

#     # Do all the same layout updates as the function below, plus adding sliders
#     title = ""
#     master_fig.update_layout(
#         sliders=sliders,
#         width=width,
#         height=height,
#         title=title,
#         margin=dict(l=30, r=30, b=10, t=20 + (30 if title is not None else 0)),
#         plot_bgcolor='white',
#     )
#     master_fig.update_yaxes(
#         range=[-1.5, 1.5],
#         showticklabels=True,
#         scaleanchor="x",
#         scaleratio=1,
#         showline=True, 
#         linewidth=1, 
#         linecolor='black',
#         mirror=True,
#     )
#     master_fig.update_xaxes(
#         range=[-1.5, 1.5],
#         showticklabels=True,
#         showline=True, 
#         linewidth=1, 
#         linecolor='black',
#         mirror=True,
#     )
#     master_fig.show()

        

            



# def visualise_2d_superposition(
#     values: Float[Tensor, "instances d_hidden feats"],
#     colors: Optional[Union[Tuple[int, int], Float[Tensor, "instances feats"]]] = None,
#     width: Optional[int] = None,
#     height: Optional[int] = None,
#     title: Optional[str] = None,
#     subplot_titles: Optional[List[str]] = None,
#     small_lines: bool = False,
#     show: bool = True,
# ):
#     """Plot a grid of subplots, each of which is a scatter plot of the values of a feature for each instance.
#     The color of each point is determined by the value of the corresponding feature in the colors tensor.
#     """
#     n_instances, d_hidden, n_feats = values.shape

#     # Make subplot
#     fig = make_subplots(
#         rows = 1,
#         cols = n_instances,
#         subplot_titles = subplot_titles,
#         shared_yaxes = True,
#         shared_xaxes = True,
#     )
#     colors = parse_colors_for_superposition_plot(colors, n_instances, n_feats)

#     # Operations like transpose mean smth different in numpy, so change for consistency
#     if isinstance(values, np.ndarray):
#         values = t.from_numpy(values)

#     for instance_idx, (instance_values, instance_colors) in enumerate(zip(values.transpose(-1, -2), colors)):
#         for feat_values, feat_color in zip(instance_values, instance_colors):

#             x, y = feat_values.tolist()

#             fig.add_trace(
#                 go.Scatter(
#                     x = [0, x],
#                     y = [0, y],
#                     mode = "lines",
#                     line = dict(color = feat_color, width = 1 if small_lines else 2),
#                     showlegend = False,
#                 ),
#                 row = 1,
#                 col = instance_idx + 1,
#             )
#             fig.add_trace(
#                 go.Scatter(
#                     x = [x],
#                     y = [y],
#                     mode = "markers",
#                     marker = dict(color = feat_color, size = 5 if small_lines else 10),
#                     showlegend = False,
#                 ),
#                 row = 1,
#                 col = instance_idx + 1,
#             )

#     if not(show):
#         return fig

#     fig.update_layout(
#         width=width,
#         height=height,
#         title=title,
#         margin=dict(l=30, r=30, b=30, t=50 + (30 if title is not None else 0)),
#         plot_bgcolor='white',
#     )
#     fig.update_yaxes(
#         range=[-1.5, 1.5],
#         showticklabels=True,
#         scaleanchor="x",
#         scaleratio=1,
#         showline=True, 
#         linewidth=1, 
#         linecolor='black',
#         mirror=True,
#     )
#     fig.update_xaxes(
#         range=[-1.5, 1.5],
#         showticklabels=True,
#         showline=True, 
#         linewidth=1, 
#         linecolor='black',
#         mirror=True,
#     )
#     fig.show()





def frac_active_line_plot(
    frac_active: Float[Tensor, "n_steps n_instances n_hidden_ae"],
    feature_probability: float,
    plot_every_n_steps: int = 1,
    title: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    y_max: Optional[float] = None,
):
    frac_active = frac_active[::plot_every_n_steps]
    n_steps, n_instances, n_hidden_ae = frac_active.shape

    y_max = y_max if (y_max is not None) else (feature_probability * 3)
    
    fig = go.Figure(layout=dict(
        template = "simple_white",
        title = title,
        xaxis_title = "Training Step",
        yaxis_title = "Fraction of Active Neurons",
        width = width,
        height = height,
        yaxis_range = [0, y_max],
    ))

    for inst in range(n_instances):
        for neuron in range(n_hidden_ae):
            fig.add_trace(go.Scatter(
                x = list(range(0, plot_every_n_steps*n_steps, plot_every_n_steps)),
                y = frac_active[:, inst, neuron].tolist(),
                name = f"AE neuron #{neuron}",
                mode = "lines",
                opacity = 0.3,
                legendgroup = f"Instance #{inst}",
                legendgrouptitle_text = f"Instance #{inst}",
            ))
    fig.add_hline(
        y = feature_probability,
        opacity = 1,
        line = dict(color="black", width=2),
        annotation_text = "Feature prob",
        annotation_position = "bottom left",
        annotation_font_size = 14,
    )
    fig.show()


def plot_feature_geometry(model, dim_fracs = None):
    fig = px.line(
        x=1/model.feature_probability[:, 0].cpu(),
        y=(model.cfg.n_hidden/(t.linalg.matrix_norm(model.W.detach(), 'fro')**2)).cpu(),
        log_x=True,
        markers=True,
        template="ggplot2",
        height=600,
        width=1000,
        title=""
    )
    fig.update_xaxes(title="1/(1-S), <-- dense | sparse -->")
    fig.update_yaxes(title=f"m/||W||_F^2")
    if dim_fracs is not None:
        dim_fracs = dim_fracs.detach().cpu().numpy()
        density = model.feature_probability[:, 0].cpu()

        for a,b in [(1,2), (1,3), (1,4), (2,3), (2,5), (2,7)]:
            val = a/b
            fig.add_hline(val, line_color="purple", opacity=0.2, line_width=1, annotation=dict(text=f"{a}/{b}"))

        for a,b in [(5,6), (4,5), (3,4), (3,5), (4,9), (3,8), (3,20)]:
            val = a/b
            fig.add_hline(val, line_color="purple", opacity=0.2, line_width=1, annotation=dict(text=f"{a}/{b}", x=0.05))

        for i in range(len(dim_fracs)):
            fracs_ = dim_fracs[i]
            N = fracs_.shape[0]
            xs = 1/density
            if i!= len(dim_fracs)-1:
                dx = xs[i+1]-xs[i]
            fig.add_trace(
                go.Scatter(
                    x=1/density[i]*np.ones(N)+dx*np.random.uniform(-0.1,0.1,N),
                    y=fracs_,
                    marker=dict(
                        color='black',
                        size=1,
                        opacity=0.5,
                    ),
                    mode='markers',
                )
            )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)
        fig.update_layout(showlegend=False)
    fig.show()