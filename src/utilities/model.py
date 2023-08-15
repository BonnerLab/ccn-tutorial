from collections.abc import Callable, Collection, Sequence, Hashable
from functools import wraps
import gc
from pathlib import Path
from typing import ParamSpec, TypeVar

from tqdm.auto import tqdm
from loguru import logger
import numpy as np
import numpy.typing as npt
import xarray as xr
import netCDF4
import torch
from torchvision.models.feature_extraction import create_feature_extractor
from torchdata.datapipes.iter import IterDataPipe, IterableWrapper
from torchdata.datapipes.map import MapDataPipe
from PIL import Image


P = ParamSpec("P")
R = TypeVar("R")

DEFAULT_DEVICES: list[torch.device] = [
    torch.device(f"cuda:{gpu}") for gpu in range(torch.cuda.device_count())
] + [torch.device("cpu")]


def collate_fn(
    batch: Sequence[tuple[torch.Tensor, str]]
) -> tuple[torch.Tensor, npt.NDArray[np.str_]]:
    images = torch.stack([pair[0] for pair in batch])
    ids = np.array([pair[1] for pair in batch])
    return images, ids


def create_image_datapipe(
    datapipe: MapDataPipe,
    *,
    preprocess_fn: Callable[[Image.Image], torch.Tensor],
    batch_size: int,
    indices: list[Hashable] | None = None,
) -> IterDataPipe:
    """Creates a PyTorch datapipe for loading images and preprocessing them.

    Args:
        datapipe: source datapipe that maps a key to a PIL.Image.Image
        preprocess_fn: function used to preprocess each PIL.Image.Image
        batch_size: batch size

    Returns:
        torch datapipe that produces batches of data in the form (image_tensor, image_id)
    """
    return (
        datapipe.to_iter_datapipe(indices=indices)
        .map(fn=preprocess_fn)
        .zip(IterableWrapper(indices))
        .batch(batch_size=batch_size)
        .collate(collate_fn=collate_fn)
    )


def try_devices(
    func: Callable[P, R],
    devices: Collection[None | torch.device | str] = DEFAULT_DEVICES,
    *,
    current: bool = False,
) -> Callable[P, R]:
    """Tries to run a function on any of the provided `devices`, exiting on success.

    For each device provided, the tensor-valued arguments and keyword arguments to `func` are copied to the device before the function is run. This allows us to write device-agnostic code, since the function can be run on whichever device is available at runtime. This function can be used as a decorator.

    Example:

    ```
    import torch

    n = 10
    x = torch.zeros(n).to("cuda:1")
    y = torch.ones(n).to("cuda:0")

    # to try running the function on all devices

    @try_devices
    def add(x: torch.Tensor, *, y: torch.Tensor) -> torch.Tensor:
        return x + y

    # to try running the function only on "cuda:1" and "cpu"

    @try_devices(devices=("cuda:1", "cpu"))
    def add(x: torch.Tensor, *, y: torch.Tensor) -> torch.Tensor:
        return x + y

    # using the function without a decorator

    z = try_devices(add)(x, y=y)
    ```

    If you need more flexibility in how the function should be applied on different devices (for e.g., your function takes in numpy arrays and not tensors as inputs), consider using the function `try_environments`.

    Args:
        func: The function that should be wrapped.
        devices: GPUs/CPU that the function should be tried on, in the order specified. Defaults to all the GPUs available and then the CPU (i.e., ["cuda:0", ..., f"cuda:{torch.cuda.device_count()}", "cpu"])
        current: Whether to try running the function with all the tensors on their current devices, defaults to False.

    Returns:
        Wrapped function that can be called.
    """
    if not devices:  # handle case with empty Collection
        devices = DEFAULT_DEVICES
    else:
        devices = [
            None if device is None else torch.device(device) for device in devices
        ]

    if current:
        devices.insert(0, None)  # type:ignore  # devices is now a list

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        contains_tensor_arg = any(
            [isinstance(arg, torch.Tensor) for arg in args]
            + [isinstance(kwarg, torch.Tensor) for kwarg in kwargs.values()]
        )
        if not contains_tensor_arg:
            logger.warning(
                f"The function {func} does not have any tensor-valued arg/kwarg:"
                " `try_devices` is redundant"
            )

        for device in devices:
            try:
                args_device = [
                    arg.to(device) if isinstance(arg, torch.Tensor) else arg
                    for arg in args
                ]
                kwargs_device = {
                    key: kwarg.to(device) if isinstance(kwarg, torch.Tensor) else kwarg
                    for key, kwarg in kwargs.items()
                }

                return func(
                    *args_device,  # type:ignore  # args_device is guaranteed to have the same type as args
                    **kwargs_device,  # type:ignore  # kwargs_device is guaranteed to have the same type as kwargs
                )
            except Exception as e:
                logger.info(
                    f"Could not run the function {func} with device"
                    f" {device} (exception {e})"
                )
                try:
                    del args_device
                    del kwargs_device
                    gc.collect()
                    torch.cuda.empty_cache()
                except Exception as e2:
                    continue

                continue
        return None

    return wrapper


class Hook:
    def __init__(self, identifier: str) -> None:
        self.identifier = identifier

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        pass


class SparseRandomProjection(Hook):
    def __init__(
        self,
        *,
        n_components: int,
        density: float | None = None,
        seed: int = 0,
        allow_expansion: bool = False,
    ) -> None:
        self.n_components = n_components
        self.density = density
        self.seed = seed
        self.allow_expansion = allow_expansion

        super().__init__(
            identifier=(
                "sparse_random_projection"
                f".n_components={self.n_components}"
                f".density={self.density}"
                f".seed={self.seed}"
                f".allow_expansion={self.allow_expansion}"
            )
        )

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        features = features.flatten(start_dim=1)
        n_features = features.shape[-1]

        projection = create_sparse_projection_matrix(
            n_features=n_features,
            n_components=self.n_components,
            density=self.density,
            seed=self.seed,
        )

        if n_features <= projection.shape[-1]:
            if not self.allow_expansion:
                return features

        return try_devices(self._project)(features=features, projection=projection)

    def _project(
        self, *, features: torch.Tensor, projection: torch.Tensor
    ) -> torch.Tensor:
        return features @ projection


def extract_features(
    *,
    model: torch.nn.modules.module.Module,
    model_identifier: str,
    nodes: list[str],
    hooks: dict[str, Hook],
    datapipe: IterDataPipe,
    datapipe_identifier: str,
    cache_path: Path = Path("cache"),
    use_cached: bool = True,
    device: torch.device | None = None,
) -> dict[str, xr.DataArray]:
    """Extract features from the internal nodes of a PyTorch model.

    WARNING: this function assumes that

    * all 4-D features are from convolutional layers and have the shape ``(presentation, channel, spatial_x, spatial_y)``
    * all 2-D features are from linear layers and have the shape ``(presentation, channel)``
    * all 3-D features are from patch-based Vision Transformers and have the shape ``(presentation, patch, channel)``

    Args:
        model: a PyTorch model
        model_identifier: identifier for the model
        nodes: list of layer names to extract features from, in standard PyTorch format (e.g. 'classifier.0')
        hooks: dictionary mapping layer names to hooks to be applied to the features extracted from the layer (e.g. {"conv2": GlobalMaxpool()})
        datapipe: torch datapipe that provides batches of data of the form ``(data, stimulus_ids)``. ``data`` is a torch Tensor with shape (batch_size, *) and ``stimulus_ids`` is a Numpy array of string identifiers corresponding to each stimulus in ``data``.
        datapipe_identifier: identifier for the dataset
        use_cached: whether to use previously computed features, defaults to True
        device: torch device on which the feature extraction will occur, defaults to None

    Returns:
        dictionary where keys are node identifiers and values are xarray DataArrays containing the model's features. Each ``xarray.DataArray`` has a ``presentation`` dimension corresponding to the stimuli with a ``stimulus_id`` coordinate corresponding to the ``stimulus_ids`` from ``datapipe``, and other dimensions that depend on the layer type and the hook.
    """

    device = _get_device(device)
    cache_dir = _create_cache_directory(
        cache_path=cache_path,
        model_identifier=model_identifier,
        datapipe_identifier=datapipe_identifier,
    )
    filepaths = _get_filepaths(nodes=nodes, hooks=hooks, cache_dir=cache_dir)
    nodes_to_compute = _get_nodes_to_compute(
        nodes=nodes, filepaths=filepaths, use_cached=use_cached
    )

    if nodes_to_compute:
        _extract_features(
            model=model,
            nodes=nodes_to_compute,
            hooks=hooks,
            filepaths=filepaths,
            device=device,
            datapipe=datapipe,
        )

    assemblies = _open_with_xarray(
        model_identifier=model_identifier,
        filepaths=filepaths,
        hooks=hooks,
        datapipe_identifier=datapipe_identifier,
    )
    return assemblies


def _open_with_xarray(
    *,
    model_identifier: str,
    filepaths: dict[str, Path],
    hooks: dict[str, Hook],
    datapipe_identifier: str,
) -> dict[str, xr.DataArray]:
    assemblies = {}
    for node, filepath in filepaths.items():
        hook = hooks[node].identifier if node in hooks.keys() else None
        assembly = xr.open_dataset(filepath)
        assemblies[node] = (
            assembly[node]
            .assign_coords(
                {"stimulus_id": ("presentation", assembly["stimulus_id"].values)}
            )
            .rename(f"{model_identifier}.node={node}.hook={hook}.{datapipe_identifier}")
        )
    return assemblies


def _extract_features(
    *,
    model: torch.nn.modules.module.Module,
    nodes: list[str],
    hooks: dict[str, Hook],
    filepaths: dict[str, Path],
    device: torch.device,
    datapipe: IterDataPipe,
) -> None:
    netcdf4_files = {
        node: netCDF4.Dataset(filepaths[node], "w", format="NETCDF4") for node in nodes
    }

    with torch.no_grad():
        model = model.eval()
        model = model.to(device=device)

        extractor = create_feature_extractor(model, return_nodes=nodes)
        extractor = extractor.to(device=device)

        start = 0
        for batch, (input_data, stimulus_ids) in enumerate(
            tqdm(datapipe, desc="batch", leave=False)
        ):
            input_data = input_data.to(device)
            features = extractor(input_data)
            for node in features.keys():
                if node in hooks.keys():
                    features[node] = hooks[node](features[node])

            for node, netcdf4_file in netcdf4_files.items():
                features_node = features[node].detach().cpu().numpy()

                if batch == 0:
                    _create_netcdf4_file(
                        file=netcdf4_file,
                        node=node,
                        features=features_node,
                    )

                end = start + len(input_data)
                netcdf4_file.variables[node][start:end, ...] = features_node
                netcdf4_file.variables["presentation"][start:end] = (
                    np.arange(len(input_data)) + start
                )
                netcdf4_file.variables["stimulus_id"][start:end] = stimulus_ids

            start += len(input_data)

    for netcdf4_file in netcdf4_files.values():
        netcdf4_file.sync()
        netcdf4_file.close()


def _create_netcdf4_file(
    *,
    file: netCDF4.Dataset,
    node: str,
    features: torch.Tensor,
) -> None:
    match features.ndim:
        case 4:  # ASSUMPTION: convolutional layer
            dimensions = ["presentation", "channel", "spatial_x", "spatial_y"]
        case 2:  # ASSUMPTION: linear layer
            dimensions = ["presentation", "channel"]
        case 3:  # ASSUMPTION: patch-based ViT
            dimensions = ["presentation", "patch", "channel"]
        case _:
            raise ValueError("features do not have the appropriate shape")

    for dimension, length in zip(dimensions, (None, *features.shape[1:])):
        file.createDimension(dimension, length)
        if dimension == "presentation":
            file.createVariable(dimension, np.int64, (dimension,))
            file.createVariable("stimulus_id", str, (dimension,))
        else:
            variable = file.createVariable(dimension, np.int64, (dimension,))
            variable[:] = np.arange(length)

    dtype = np.dtype(getattr(np, str(features.dtype).replace("torch.", "")))
    file.createVariable(node, dtype, dimensions)


def _get_device(device: str | torch.device | None = None) -> torch.device:
    if device is None:
        if torch.cuda.is_available():
            device_ = torch.device("cuda")
        else:
            device_ = torch.device("cpu")
    else:
        device_ = torch.device(device)
    return device_


def _create_cache_directory(
    *, cache_path: Path, model_identifier: str, datapipe_identifier: str
) -> Path:
    cache_dir = (
        cache_path
        / "features"
        / f"{model_identifier}"
        / f"datapipe={datapipe_identifier}"
    )
    cache_dir.mkdir(exist_ok=True, parents=True)
    return cache_dir


def _get_filepaths(
    *,
    nodes: list[str],
    hooks: dict[str, Hook],
    cache_dir: Path,
) -> dict[str, Path]:
    filepaths: dict[str, Path] = {}
    for node in nodes:
        if node in hooks.keys():
            hook_identifier = hooks[node].identifier
        else:
            hook_identifier = "None"
        filepaths[node] = cache_dir / f"node={node}" / f"hook={hook_identifier}.nc"
        filepaths[node].parent.mkdir(exist_ok=True, parents=True)
    return filepaths


def _get_nodes_to_compute(
    *, nodes: list[str], filepaths: dict[str, Path], use_cached: bool
) -> list[str]:
    nodes_to_compute = nodes.copy()
    for node in nodes:
        if filepaths[node].exists():
            if use_cached:
                nodes_to_compute.remove(node)  # don't re-compute
            else:
                filepaths[node].unlink()  # delete pre-cached features
    return nodes_to_compute


def flatten_features(features: dict[str, xr.DataArray]) -> dict[str, xr.DataArray]:
    """Flattens features from each node into a ``neuroid`` dimension.

    Args:
        features: dictionary of features from each node

    Returns:
        dictionary of flattened features from each node
    """
    for node in features.keys():
        dims = list(set(features[node].dims) - {"presentation"})
        features[node] = features[node].stack({"neuroid": dims}).reset_index("neuroid")
    return features


def compute_johnson_lindenstrauss_limit(*, n_samples: int, epsilon: float) -> int:
    return int(
        np.ceil(4 * np.log(n_samples) / ((epsilon**2) / 2 - (epsilon**3) / 3))
    )


def create_sparse_projection_matrix(
    *,
    n_features: int,
    n_components: int,
    density: float | None = None,
    seed: int = 0,
) -> torch.Tensor:
    assert isinstance(n_features, int), "n_features must be an int"
    assert n_features > 1, "n_features must be > 1"

    if density is None:
        density = np.exp(-np.log(n_features) / 2)
    else:
        assert isinstance(density, float)
        assert density > 0, "density must be > 0"
        assert density <= 1, "density must be <= 1"

    assert isinstance(n_components, int), "n_components must be an int"
    assert n_components >= 1, "n_components must be >= 1"

    scale = np.exp(-(np.log(density) + np.log(n_components)) / 2)

    n_elements = n_features * n_components

    rng = np.random.default_rng(seed=seed)
    n_nonzero = rng.binomial(n=n_elements, p=density, size=1)[0]
    indices = rng.choice(a=n_elements, size=n_nonzero, replace=False).astype(np.int64)
    locations = np.stack(
        np.unravel_index(indices=indices, shape=(n_features, n_components))
    )

    projection = torch.sparse_coo_tensor(
        indices=torch.from_numpy(locations),
        values=scale
        * (2 * rng.binomial(n=1, p=0.5, size=n_nonzero) - 1).astype(np.float32),
        size=(n_features, n_components),
    )
    return projection

