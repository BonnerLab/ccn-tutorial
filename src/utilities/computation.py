from collections.abc import Sequence

import numpy as np
import torch


def svd(
    x: torch.Tensor,
    *,
    truncated: bool,
    n_components: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if truncated:
        torch.manual_seed(seed)
        u, s, v = torch.pca_lowrank(x, center=False, q=n_components)
        v_h = v.transpose(-2, -1)
    else:
        u, s, v_h = torch.linalg.svd(x, full_matrices=False)
    u, v_h = svd_flip(u=u, v=v_h)
    return u, s, v_h.transpose(-2, -1)


def svd_flip(*, u: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    max_abs_cols = torch.argmax(torch.abs(u), dim=-2)
    match u.ndim:
        case 3:
            signs = torch.stack(
                [
                    torch.sign(u[i_batch, max_abs_cols[i_batch, :], range(u.shape[-1])])
                    for i_batch in range(u.shape[0])
                ],
                dim=0,
            )
        case 2:
            signs = torch.sign(u[..., max_abs_cols, range(u.shape[-1])])

    u *= signs.unsqueeze(-2)
    v *= signs.unsqueeze(-1)

    return u, v


def assign_logarithmic_bins(
    data: Sequence[int | float],
    *,
    base: int = 10,
    points_per_bin: int,
    min_: int | float,
    max_: int | float,
) -> Sequence[float]:
    n = points_per_bin * int(np.ceil((np.log(max_) - np.log(min_)) / (np.log(base))))
    bin_edges = np.geomspace(min_, max_, num=n)
    bin_centers = np.exp(np.log(bin_edges)[:-1] + np.diff(np.log(bin_edges)) / 2)

    bin_edges[-1] = np.inf
    labels = bin_centers[np.digitize(data, bin_edges) - 1]

    return labels
