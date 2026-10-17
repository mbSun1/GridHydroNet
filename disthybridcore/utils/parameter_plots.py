"""Compact figures for exported HBV and grid-routing parameters."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_figure(figure, directory, filename, purpose):
    path = Path(directory) / filename
    figure.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(figure)
    return {"file": filename, "purpose": purpose}


def plot_parameter_maps(dataset, names, directory, filename, title):
    columns = min(4, len(names))
    rows = int(np.ceil(len(names) / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(4.2 * columns, 3.5 * rows),
        squeeze=False,
    )
    mask = dataset.basin_mask.values.astype(bool)
    for axis, name in zip(axes.flat, names):
        values = np.where(mask, dataset[name].values, np.nan)
        image = axis.imshow(values, origin="upper", cmap="viridis")
        axis.set_title(name)
        axis.set_xticks([])
        axis.set_yticks([])
        figure.colorbar(image, ax=axis, shrink=.75)
    for axis in axes.flat[len(names):]:
        axis.set_visible(False)
    figure.suptitle(title)
    return save_figure(figure, directory, filename, title)


def plot_routing_parameter_distribution(dataset, names, directory):
    figure, axes = plt.subplots(
        1, len(names), figsize=(5 * len(names), 4), squeeze=False
    )
    mask = dataset.basin_mask.values.astype(bool)
    for axis, name in zip(axes.flat, names):
        variable = dataset[name]
        if variable.ndim == 2:
            axis.hist(variable.values[mask], bins=40, color="#35618d", alpha=.85)
        else:
            axis.bar([name], [float(variable)], color="#35618d")
        axis.set_title(name)
        axis.set_ylabel(variable.attrs.get("units", "1"))
    return save_figure(
        figure,
        directory,
        "routing_parameters.png",
        "Learned grid-routing parameters",
    )


def generate_parameter_figures(dataset, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    hbv = json.loads(dataset.attrs["hbv_parameters"])
    routing = json.loads(dataset.attrs["routing_parameters"])
    figures = [
        plot_parameter_maps(
            dataset,
            hbv,
            directory,
            "hbv_parameter_maps.png",
            "HBV parameter fields",
        ),
        plot_routing_parameter_distribution(dataset, routing, directory),
    ]
    grid_routing = [name for name in routing if dataset[name].ndim == 2]
    if grid_routing:
        figures.append(plot_parameter_maps(
            dataset, grid_routing, directory, "routing_parameter_maps.png",
            f"{dataset.attrs['routing']} parameter fields",
        ))
    correlation_path = directory / "parameter_correlations.csv"
    if correlation_path.is_file():
        table = pd.read_csv(correlation_path, index_col=0)
        figure, axis = plt.subplots(figsize=(9, 8))
        image = axis.imshow(table.values, vmin=-1, vmax=1, cmap="coolwarm")
        axis.set_xticks(range(len(table.columns)), table.columns, rotation=90)
        axis.set_yticks(range(len(table.index)), table.index)
        figure.colorbar(image, ax=axis, label="weighted Spearman")
        figures.append(save_figure(
            figure,
            directory,
            "parameter_correlations.png",
            "HBV spatial correlations",
        ))
    pd.DataFrame(figures).to_csv(directory / "figure_manifest.csv", index=False, encoding="utf-8-sig")
    return figures
