import numpy as np
import matplotlib.pyplot as plt
from neuropy.utils.signal_process import WaveletSg
from neuropy.core import Epoch
from neuropy.plotting.signals import plot_spectrogram

cs_color_dict = {"CS+": [1, 0.647, 0, 0.3], "CS-": [0, 1, 0, 0.3]}


def plot_cs_mean_wavelet(
    wv: WaveletSg,
    cs_start_times: np.ndarray,
    cs_type: str,
    buffer_sec: tuple = (5, 35),
    freq_lims: tuple = (5, 12),
    calc_mean: bool = True,
    ignore_epochs: Epoch = None,
    ax: plt.Axes = None,
    std_sxx=None,
):
    assert cs_type in ["CS+", "CS-"]
    if ax is None:
        _, ax = plt.subplots(
            2,
            1,
            figsize=(9, 4.5),
            height_ratios=[1, 5],
            sharex=True,
            layout="tight",
        )

    # Plot CS and US (predicted) times
    ax[0].axvspan(0, 10, color=cs_color_dict[cs_type])
    ax[0].axvline(30, color="r", linestyle="--")

    # Calculate and plot mean spectrogram
    if calc_mean:
        wv_mean = wv.get_pe_mean_spec(
            cs_start_times, buffer_sec, ignore_epochs=ignore_epochs
        )
    else:  # Use already calculated mean
        wv_mean = wv

    # Calculate standard deviation for scaling spectrogram before calculating mean if specified
    if std_sxx is None:
        std_sxx = np.nanstd(wv_mean.traces)
    plot_spectrogram(
        wv_mean,
        time_lims=np.multiply(buffer_sec, (-1, 1)),
        freq_lims=freq_lims,
        ax=ax[1],
        std_sxx=std_sxx,
    )
    ax[1].set_xlabel(f"Time from {cs_type} start (sec)")

    return ax, std_sxx


def plot_cs_outside_axes(
    ax: plt.Axes,
    buffer_sec: tuple,
    cs_type: str in ["CS-", "CS+"],
    plot_below: bool = True,
    box_width: float = 20,
):
    """Hack to plot cs duration above and below axes"""
    arrowprops = dict(
        arrowstyle="-",
        color=cs_color_dict[cs_type],
        linewidth=box_width,
        capstyle="butt",
    )
    xaxis_len = np.sum(buffer_sec)

    # plot above
    ax.annotate(
        "",
        xy=(buffer_sec[0] / xaxis_len, 1.05),
        xycoords="axes fraction",
        xytext=((buffer_sec[0] + 10) / xaxis_len, 1.05),
        arrowprops=arrowprops,
    )
    # plot below
    if plot_below:
        ax.annotate(
            "",
            xy=(buffer_sec[0] / xaxis_len, -0.05),
            xycoords="axes fraction",
            xytext=((buffer_sec[0] + 10) / xaxis_len, -0.05),
            arrowprops=arrowprops,
        )
