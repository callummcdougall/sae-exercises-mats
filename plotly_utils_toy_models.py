import torch as t
from torch import Tensor
from typing import List, Union, Optional
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Tuple, List
from matplotlib import pyplot as plt
from jaxtyping import Float
import einops


def visualise_Nd_superposition(
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
    return f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})"


def visualise_2d_superposition(
    values: Float[Tensor, "instances d_hidden feats"],
    colors: Optional[Union[Tuple[int, int], Float[Tensor, "instances feats"]]] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    title: Optional[str] = None,
    subplot_titles: Optional[List[str]] = None,
    small_lines: bool = False,
):
    """Plot a grid of subplots, each of which is a scatter plot of the values of a feature for each instance.
    The color of each point is determined by the value of the corresponding feature in the colors tensor.
    """
    n_instances, d_hidden, n_feats = values.shape

    # Make subplot
    fig = make_subplots(
        rows = 1,
        cols = n_instances,
        subplot_titles = subplot_titles,
        shared_yaxes = True,
        shared_xaxes = True,
    )

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

    for instance_idx, (instance_values, instance_colors) in enumerate(zip(values.transpose(-1, -2), colors)):
        for feat_idx, (feat_values, feat_color) in enumerate(zip(instance_values, instance_colors)):
            x, y = feat_values.tolist()


            fig.add_trace(
                go.Scatter(
                    x = [0, x],
                    y = [0, y],
                    mode = "lines",
                    line = dict(color = feat_color, width = 1 if small_lines else 2),
                    showlegend = False,
                ),
                row = 1,
                col = instance_idx + 1,
            )
            fig.add_trace(
                go.Scatter(
                    x = [x],
                    y = [y],
                    mode = "markers",
                    marker = dict(color = feat_color, size = 5 if small_lines else 10),
                    showlegend = False,
                ),
                row = 1,
                col = instance_idx + 1,
            )

    fig.update_layout(
        width=width,
        height=height,
        title=title,
        margin=dict(l=30, r=30, b=30, t=50 + (30 if title is not None else 0)),
        plot_bgcolor='white',  # Set background color to white
    )
    fig.update_yaxes(
        range=[-1.5, 1.5],
        showticklabels=True,  # Enable y-axis tick labels
        scaleanchor="x",  # Ensure equal aspect ratio
        scaleratio=1,
        showline=True, 
        linewidth=1, 
        linecolor='black',
        mirror=True,
    )
    fig.update_xaxes(
        range=[-1.5, 1.5],
        showticklabels=True,  # Enable x-axis tick labels
        showline=True, 
        linewidth=1, 
        linecolor='black',
        mirror=True,
    )
    # fig.add_vline(x=0, line_width=1, line_color="black")
    # fig.add_hline(y=0, line_width=1, line_color="black")
    fig.show()

def plot_feature_geometry(model, dim_fracs = None):
    fig = px.line(
        x=1/model.feature_probability[:, 0].cpu(),
        y=(model.config.n_hidden/(t.linalg.matrix_norm(model.W.detach(), 'fro')**2)).cpu(),
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

        for a,b in [(1,2), (2,3), (2,5), (2,6), (2,7)]:
            val = a/b
            fig.add_hline(val, line_color="purple", opacity=0.2, annotation=dict(text=f"{a}/{b}"))

        for a,b in [(5,6), (4,5), (3,4), (3,8), (3,12), (3,20)]:
            val = a/b
            fig.add_hline(val, line_color="blue", opacity=0.2, annotation=dict(text=f"{a}/{b}", x=0.05))

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

def frac_active_line_plot(
    frac_active: Float[Tensor, "n_steps n_instances n_hidden_ae"],
    plot_every_n_steps: int,
    feature_probability: float,
    title: str,
    width: Optional[int] = None,
    height: Optional[int] = None,
):
    frac_active = frac_active[::plot_every_n_steps]
    n_steps, n_instances, n_hidden_ae = frac_active.shape
    
    fig = go.Figure(layout=dict(
        template = "simple_white",
        title = title,
        xaxis_title = "Training Step",
        yaxis_title = "Fraction of Active Neurons",
        width = width,
        height = height,
        yaxis_range = [0, feature_probability * 3],
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
        y=(model.config.n_hidden/(t.linalg.matrix_norm(model.W.detach(), 'fro')**2)).cpu(),
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

        for a,b in [(1,2), (2,3), (2,5), (2,6), (2,7)]:
            val = a/b
            fig.add_hline(val, line_color="purple", opacity=0.2, annotation=dict(text=f"{a}/{b}"))

        for a,b in [(5,6), (4,5), (3,4), (3,8), (3,12), (3,20)]:
            val = a/b
            fig.add_hline(val, line_color="blue", opacity=0.2, annotation=dict(text=f"{a}/{b}", x=0.05))

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