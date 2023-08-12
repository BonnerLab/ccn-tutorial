from collections.abc import Hashable
from pathlib import Path
import requests
import typing
import uuid
import zipfile

from loguru import logger
import numpy as np
import xarray as xr
from scipy.ndimage import map_coordinates
import nibabel as nib
import nilearn.plotting
import matplotlib as mpl

MNI_ORIGIN = np.asarray([183 - 91, 127, 73]) - 1
MNI_RESOLUTION = 1
MNI_SHAPE = (182, 218, 182)

OSF_URL = "https://files.osf.io/v1/resources/zk265/providers/osfstorage/?zip="


def prepare_filepath(*, force: bool, filepath: Path | None, url: str) -> Path:
    if filepath is None:
        filepath = Path("/tmp") / f"{uuid.uuid4()}"
    elif filepath.exists():
        if not force:
            logger.debug(
                "Using previously downloaded file at"
                f" {filepath} instead of downloading from {url}"
            )
        else:
            filepath.unlink()
    filepath.parent.mkdir(exist_ok=True, parents=True)
    return filepath


def download(
    url: str,
    *,
    filepath: Path | None = None,
    stream: bool = True,
    allow_redirects: bool = True,
    chunk_size: int = 1024**2,
    force: bool = False,
) -> Path:
    filepath = prepare_filepath(
        filepath=filepath,
        url=url,
        force=force,
    )
    if filepath.exists():
        return filepath

    logger.debug(f"Downloading from {url} to {filepath}")
    r = requests.Session().get(url, stream=stream, allow_redirects=allow_redirects)
    with open(filepath, "wb") as f:
        for chunk in r.iter_content(chunk_size=chunk_size):
            f.write(chunk)

    return filepath


def unzip(
    filepath: Path,
    *,
    extract_dir: Path | None = None,
    remove_zip: bool = True,
    password: bytes | None = None,
) -> Path:
    if extract_dir is None:
        extract_dir = Path("/tmp")
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(filepath, "r") as f:
        if all([(extract_dir / filename).exists() for filename in f.namelist()]):
            logger.debug(
                f"Using previously extracted files from {extract_dir}"
                f" instead of extracting from {filepath}"
            )
        else:
            logger.debug(f"Extracting from {filepath} to {extract_dir}")
            f.extractall(extract_dir, pwd=password)

    if remove_zip:
        logger.debug(f"Deleting {filepath} after extraction")
        filepath.unlink()

    return extract_dir


def download_dataset(url: str = OSF_URL) -> Path:
    filepath = download(url)
    return unzip(filepath, extract_dir=Path("cache") / "downloads", remove_zip=False)


def load_dataset(*, subject: int, roi: str) -> xr.DataArray:
    filepath = download_dataset()
    return xr.open_dataarray(
        filepath / "data" / "activations" / f"roi={roi}" / f"subject={subject}.nc"
    )


def load_stimuli() -> xr.DataArray:
    filepath = download_dataset()
    return xr.open_dataarray(filepath / "data" / "stimuli.nc")


def load_transformation(*, subject: int) -> np.ndarray:
    filepath = download_dataset()
    return nib.load(
        filepath
        / "data"
        / "transformations"
        / f"subject={subject}"
        / "func1pt8-to-MNI.nii.gz"
    ).get_fdata()


def groupby_reset(
    x: xr.DataArray, *, groupby_coord: str, groupby_dim: Hashable
) -> xr.DataArray:
    return (
        x.reset_index(groupby_coord)
        .rename({groupby_coord: groupby_dim})
        .assign_coords({groupby_coord: (groupby_dim, x[groupby_coord].values)})
        .drop_vars(groupby_dim)
    )


def average_data_across_repetitions(data: xr.DataArray) -> xr.DataArray:
    return groupby_reset(
        data.load()
        .groupby("stimulus_id")
        .mean()
        .assign_attrs(
            data.attrs | {"postprocessing": "averaged across first two repetitions"}
        ),
        groupby_coord="stimulus_id",
        groupby_dim="presentation",
    ).transpose("presentation", "neuroid")


def reshape_dataarray_to_brain(data: xr.DataArray, *, mni: bool = False) -> np.ndarray:
    if mni:
        brain_shape = MNI_SHAPE
    else:
        brain_shape = data.attrs["brain shape"]

    output = np.full(brain_shape, fill_value=np.nan)
    output[..., data["x"].values, data["y"].values, data["z"].values] = data.values
    return output


def plot_brain_map(
    data: xr.DataArray,
    *,
    subject: int,
    **kwargs: typing.Any,
) -> mpl.figure.Figure:
    mni = reshape_dataarray_to_brain(
        convert_array_to_mni(data, subject=subject),
        mni=True,
    )
    volume = convert_ndarray_to_nifti1image(mni)
    kwargs = {
        "views": ["lateral", "medial", "ventral"],
        "hemispheres": ["left", "right"],
        "colorbar": True,
        "inflate": True,
        "threshold": np.finfo(np.float32).resolution,
    } | kwargs

    fig, _ = nilearn.plotting.plot_img_on_surf(
        volume,
        **kwargs,
    )
    return fig


def convert_array_to_mni(data: xr.DataArray, *, subject: int) -> xr.DataArray:
    data_ = reshape_dataarray_to_brain(data)

    transformation = load_transformation(subject=subject)
    transformation -= 1
    transformation = np.flip(transformation, axis=0)

    good_voxels = np.all(
        np.stack(
            [transformation[..., dim] < data_.shape[dim] for dim in (-1, -2, -3)]
            + [np.all(np.isfinite(transformation), axis=-1)],
            axis=-1,
        ),
        axis=-1,
    )
    neuroids = xr.DataArray(
        data=good_voxels,
        dims=("x", "y", "z"),
    ).stack({"neuroid": ("x", "y", "z")})
    neuroids = neuroids[neuroids]

    output = xr.DataArray(
        data=map_coordinates(
            np.nan_to_num(data_).astype(np.float64),
            transformation[good_voxels, :].transpose(),
            order=3,
            mode="nearest",
        ).astype(np.float32),
        dims=("neuroid",),
        coords={dim: ("neuroid", neuroids[dim].values) for dim in ("x", "y", "z")},
    )
    return output


def convert_ndarray_to_nifti1image(
    data: np.ndarray,
    *,
    resolution: float = MNI_RESOLUTION,
    origin: np.ndarray = MNI_ORIGIN,
) -> nib.Nifti1Image:
    header = nib.Nifti1Header()
    header.set_data_dtype(data.dtype)

    affine = np.diag([resolution] * 3 + [1])
    if origin is None:
        origin = (([1, 1, 1] + np.asarray(data.shape)) / 2) - 1
    affine[0, -1] = -origin[0] * resolution
    affine[1, -1] = -origin[1] * resolution
    affine[2, -1] = -origin[2] * resolution

    return nib.Nifti1Image(data, affine, header)
