from typing import Optional
from typing import Tuple

import matplotlib.figure
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import gridspec
from matplotlib.ticker import MaxNLocator

from samba_mixer.utils.typing import Number


sns.set_theme(style="whitegrid", context="paper", font_scale=1.5, font="DejaVu Serif")


def get_figure_battery_prediction(
    battery_id: int,
    cycle_id: pd.Series,
    outputs: pd.Series,
    labels: pd.Series,
    error: pd.Series,
    eol_threshold: Optional[int] = None,
    eol_indicator_outputs: Optional[int] = None,
    eol_indicator_labels: Optional[int] = None,
    y_lim_pred: Tuple[Optional[Number], Optional[Number]] = (None, None),
    y_lim_error: Tuple[Optional[Number], Optional[Number]] = (None, None),
    y_label: str = "",
) -> matplotlib.figure.Figure:
    """Generates and returns figure to compare predictions with labels of a given battery.

    Args:
        battery_id (int): ID of the battery to be plotted
        cycle_id (pd.Series):
        outputs (pd.Series):
        labels (pd.Series):
        error (pd.Series):
        y_lim_pred (Tuple[Optional[Number], Optional[Number]]): Upper and lower limit of y-axis values for prediction plot.
        y_lim_error (Tuple[Optional[Number], Optional[Number]]): Upper and lower limit of y-axis values for error plot.
        y_label (str): label for y_axis.

    Returns:
        matplotlib.figure.Figure: Figure object.
    """
    # fig = plt.figure(figsize=(12, 5), dpi=200, constrained_layout=True)
    fig = plt.figure(figsize=(7, 7), dpi=200, constrained_layout=True)
    fig.suptitle(f"Battery: {battery_id}")

    spec = gridspec.GridSpec(ncols=1, nrows=2, height_ratios=[3, 1])

    # ax1 = fig.add_subplot(1, 2, 1)
    ax1 = fig.add_subplot(spec[0])
    ax1.title.set_text("Prediction vs. Ground Truth")
    # ax1.set_xlabel("Cycle")
    ax1.set_ylabel(y_label)
    ax1.set_ylim(bottom=y_lim_pred[0], top=y_lim_pred[1])
    if eol_threshold is not None:
        ax1.plot(cycle_id, eol_threshold, "--", color="g", label="EOL Threshold")
    if eol_indicator_outputs is not None:
        ax1.axvline(x=eol_indicator_outputs, color="r", label=f"EOL Prediction: {eol_indicator_outputs}")
    if eol_indicator_labels is not None:
        ax1.axvline(x=eol_indicator_labels, color="g", label=f"EOL GT: {eol_indicator_labels}")
    ax1.plot(cycle_id, labels, color="orange", label="Ground Truth")
    ax1.plot(cycle_id, outputs, color="b", label="Prediction")
    plt.tick_params("x", labelbottom=False)

    ax1.legend()
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    # ax2 = fig.add_subplot(1, 2, 2)
    ax2 = fig.add_subplot(spec[1], sharex=ax1)
    ax2.title.set_text("Prediction Error")
    ax2.set_xlabel("Cycle")
    ax2.set_ylabel(f"Error")
    ax2.set_ylim(bottom=y_lim_error[0], top=y_lim_error[1])
    ax2.hlines(10, 0, len(cycle_id), colors="red", linestyles="dashed", alpha=0.5)
    ax2.hlines(5, 0, len(cycle_id), colors="orange", linestyles="dashed", alpha=0.5)
    # ax2.hlines(0, 0, len(cycle_id), colors="green", linestyles="dashed",alpha=0.5)
    ax2.hlines(-5, 0, len(cycle_id), colors="orange", linestyles="dashed", alpha=0.5)
    ax2.hlines(-10, 0, len(cycle_id), colors="red", linestyles="dashed", alpha=0.5)
    ax2.plot(cycle_id, error)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

    sns.despine()

    return fig


def get_figure_confusion_matrix(confusion_matrix: pd.DataFrame) -> matplotlib.figure.Figure:
    # TODO Sascha: Add optional Battery ID as title
    fig = plt.figure(figsize=(8, 8), dpi=100)
    hmap = sns.heatmap(confusion_matrix, annot=True, fmt=".2f", cmap="Blues")
    hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha="right")
    hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha="right")
    plt.ylabel("True SOH bin")
    plt.xlabel("Predicted SOH bin")

    return fig


def get_figure_predicted_over_true(
    battery_data: pd.DataFrame,
    column_prediction: str,
    column_label: str,
) -> matplotlib.figure.Figure:
    fig = plt.figure(figsize=(8, 8), dpi=100)
    scatter = sns.scatterplot(
        data=battery_data,
        x=column_label,
        y=column_prediction,
        hue="battery_id",
        palette="tab10",
    )
    scatter.axline(xy1=(60, 60), xy2=(100, 100), color="r", linestyle="--")
    plt.ylabel("Predicted SOH[%]")
    plt.xlabel("Ground Truth SOH[%]")
    # plt.grid(visible=True)

    sns.despine()

    return fig
