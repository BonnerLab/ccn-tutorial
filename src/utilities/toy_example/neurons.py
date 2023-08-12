from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
import pandas as pd
import xarray as xr

Neuron = NamedTuple(
    "Neuron",
    beta_color=float,
    beta_size=float,
    mean=float,
    std=float,
)


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
