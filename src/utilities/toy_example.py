from collections.abc import Sequence, Callable
from typing import NamedTuple

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns


Neuron = NamedTuple(
    "Neuron",
    beta_color=float,
    beta_size=float,
    mean=float,
    std=float,
)


def create_stimuli(
    n: int,
    *,
    rng: np.random.Generator,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "color": rng.random(size=(n,)),
            "size": rng.random(size=(n,)),
        }
    ).set_index(1 + np.arange(n))


def simulate_neuron_responses(
    stimuli: pd.DataFrame,
    neuron: Neuron,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    def z_score(x: np.ndarray) -> np.ndarray:
        return (x - x.mean()) / x.std()

    return (
        neuron.beta_color * z_score(stimuli["color"])
        + neuron.beta_size * z_score(stimuli["size"])
        + neuron.std * rng.standard_normal(size=(len(stimuli),))
        + neuron.mean
    )


def simulate_multiple_neuron_responses(
    *,
    stimuli: pd.DataFrame,
    neurons: Sequence[Neuron],
    rng: np.random.Generator,
) -> xr.DataArray:
    data = []
    for i_neuron, neuron in enumerate(neurons):
        data.append(
            xr.DataArray(
                data=simulate_neuron_responses(
                    stimuli=stimuli,
                    neuron=neuron,
                    rng=rng,
                ),
                dims=("stimulus",),
                coords={
                    column: ("stimulus", values)
                    for column, values in stimuli.reset_index(names="stimulus").items()
                },
            )
            .expand_dims({"neuron": [i_neuron + 1]})
            .assign_coords(
                {
                    field: ("neuron", [float(value)])
                    for field, value in neuron._asdict().items()
                }
            )
        )

    return (
        xr.concat(data, dim="neuron")
        .rename("neuron responses")
        .transpose("stimulus", "neuron")
    )


def view_individual_scatter(
    data: xr.DataArray,
    *,
    coord: str,
    dim: str,
    template_func: Callable[[int], str],
) -> Figure:
    rng = np.random.default_rng()
    data_ = data.assign_coords(
        {"arbitrary": ("stimulus", rng.random(data.sizes["stimulus"]))}
    )
    min_, max_ = data_.min(), data_.max()

    n_features = data.sizes[dim]

    fig, axes = plt.subplots(nrows=n_features, figsize=(7, 2 * n_features))

    for index, ax in zip(data[coord].values, axes.flat):
        label = template_func(index)
        sns.scatterplot(
            ax=ax,
            data=(
                data_.isel({dim: data[coord].values == index})
                .rename(label)
                .to_dataframe()
            ),
            x=label,
            y="arbitrary",
            hue="color",
            size="size",
            palette="flare",
            legend=False,
        )
        sns.despine(ax=ax, left=True, offset=10)

        ax.set_xlim([min_, max_])
        ax.get_yaxis().set_visible(False)

    fig.tight_layout(h_pad=3)
    plt.close(fig)

    return fig
