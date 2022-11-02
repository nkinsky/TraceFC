import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec

from neuropy.plotting.ca_events import plot_pe_traces, Raster
from neuropy.utils.plot_util import sparse_axes_labels


## NRK todo: make this work for neuron_inds as list or pd.DataFrame
def plot_paired_CS_PETH(
    caneurons: list,
    event_starts: list,
    event_ends: list,
    var_plot: str in ["YrA", "C", "S"],
    neuron_inds: list,
    cs_types: list,
    sesh_names: list,
    cs_color=np.array([1, 0.647, 0, 0.3]),
    plot_rast: bool = False,
    fig_or_gs_use: plt.Figure or plt.SubplotSpec = None,
    **kwargs,
):
    """**kwargs to seaborn.heatmap()"""

    if np.array(cs_color).ndim == 1:
        cs_color = np.array(cs_color).reshape(1, -1).repeat(len(caneurons), axis=0)

    # Check inputs
    ninds = [len(inds) for inds in neuron_inds]
    assert np.array(
        [n == ninds[0] for n in ninds]
    ).all(), "# inds in neuron_inds input must all be the same"
    neuron_inds = np.array(neuron_inds)

    # Set up plot
    ncols = len(caneurons)
    nrows = len(neuron_inds[0])
    if not plot_rast:
        assert (
            fig_or_gs_use is None
        ), "plot_rast=True only works with fig_use=None currently"
        fig, ax = plt.subplots(nrows, ncols, figsize=(8.5, 3 * nrows))
    else:
        fig, ax, axrast = create_peth_and_raster_axes(
            nrows, ncols, fig_or_gs=fig_or_gs_use
        )

    for ids, (caneuron, starts, ends, inds, color) in enumerate(
        zip(caneurons, event_starts, event_ends, neuron_inds, cs_color)
    ):
        traces_plot = getattr(caneuron, var_plot)[inds]
        cs_name = cs_types[ids]
        sesh_name = sesh_names[ids]
        for idc, trace in enumerate(traces_plot):
            _, _, rast, _, _ = plot_pe_traces(
                caneuron.t["Timestamps"],
                trace,
                event_starts=starts["Timestamp"],
                event_ends=ends["Timestamp"],
                event_color=color,
                raw_trace=None,
                end_buffer_sec=40,
                ax=ax[idc, ids],
            )
            # Plot raster if specified
            if plot_rast:
                sns.heatmap(rast, ax=axrast[idc, ids], **kwargs)
                sns.despine(ax=axrast[idc, ids])
                axrast[idc, 0].set_ylabel("Trials")
                axrast[idc, ids].set(xticks=[], xticklabels=[])
                sparse_axes_labels(axrast[idc, ids], "y")

            ax[idc, ids].set_title(f"{sesh_name}: {cs_name} Cell #{idc}")
        [a.axvline(30, color="r", linestyle="--") for a in ax[:, ids]]

    # Turn off redundant and overlapping ylabels
    [a.set_ylabel("") for a in ax[:, 1:].reshape(-1)]

    # Make font smaller so that you can read titles
    [a.set_title(a.get_title(), fontdict={"fontsize": 8}) for a in ax.reshape(-1)]

    # overwrite inds == -1 with a blank plot with an x through it.
    silent_bool = neuron_inds.T == -1
    for a in ax[silent_bool].reshape(-1):
        a.clear()
        sns.despine(ax=a, bottom=True, left=True)
        a.set(xticks=[], yticks=[])
    if plot_rast:
        for a in axrast[silent_bool].reshape(-1):
            a.clear()
            sns.despine(ax=a, bottom=True, left=True)
            a.set(xticks=[], yticks=[])

    return fig, ax


def create_peth_and_raster_axes(
    nrows, ncols, height_ratio=(3, 1), fig_or_gs: plt.Figure or plt.SubplotSpec = None
):
    """Create alternating rows of axes for PETH traces and corresponding rasters below"""
    if fig_or_gs is None:
        fig = plt.figure(figsize=(8.5, 4 * nrows))
    if isinstance(fig_or_gs, plt.Figure):
        gs = fig.add_gridspec(nrows * 2, ncols, height_ratios=height_ratio * nrows)
    else:
        gs = gridspec.GridSpecFromSubplotSpec(
            nrows * 2, ncols, subplot_spec=fig_or_gs, height_ratios=height_ratio * nrows
        )
        fig = fig_or_gs.get_gridspec().figure

    ax, axrast = [], []
    for idr in range(nrows):
        axcol, axcolrast = [], []
        for idc in range(ncols):
            axcol.append(fig.add_subplot(gs[idr * 2, idc]))
            axcolrast.append(fig.add_subplot(gs[idr * 2 + 1, idc]))
        ax.append(axcol)
        axrast.append(axcolrast)
    ax = np.array(ax)
    axrast = np.array(axrast)

    return fig, ax, axrast
