"""
SmartSPIM Zarr writer. It takes an input path
where 3D fused chunked files are located,
reconstructs the volume as a dask array and
writes it in OME-Zarr format
"""

import logging
import multiprocessing
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Hashable, List, Optional, Sequence, Tuple, Union, cast

import dask
import dask.array as da

# import matplotlib.pyplot as plt
import numpy as np
import pims
import xarray_multiscale
import zarr
from dask import config as da_cfg
from dask.array.core import Array
from dask.base import tokenize
from dask.distributed import Client, LocalCluster, performance_report

# from distributed import wait
from numcodecs import blosc
from ome_zarr.format import CurrentFormat
from ome_zarr.io import parse_url
from ome_zarr.writer import write_multiscales_metadata
from skimage.io import imread as sk_imread

from aind_smartspim_data_transformation.compress.zarr_utilities import *
from aind_smartspim_data_transformation.compress.zarr_writer import (
    BlockedArrayWriter,
)
from aind_smartspim_data_transformation.io import PngReader


def set_dask_config(dask_folder: str):
    """
    Sets dask configuration

    Parameters
    ----------
    dask_folder: str
        Path to the temporary directory and local directory
        of workers in dask.
    """
    # Setting dask configuration
    da_cfg.set(
        {
            "temporary-directory": dask_folder,
            "local_directory": dask_folder,
            # "tcp-timeout": "300s",
            "array.chunk-size": "128MiB",
            "distributed.worker.memory.target": 0.90,  # 0.85,
            "distributed.worker.memory.spill": 0.92,  # False,#
            "distributed.worker.memory.pause": 0.95,  # False,#
            "distributed.worker.memory.terminate": 0.98,
        }
    )


def _build_ome(
    data_shape: Tuple[int, ...],
    image_name: str,
    channel_names: Optional[List[str]] = None,
    channel_colors: Optional[List[int]] = None,
    channel_minmax: Optional[List[Tuple[float, float]]] = None,
    channel_startend: Optional[List[Tuple[float, float]]] = None,
) -> Dict:
    """
    Create the necessary metadata for an OME tiff image

    Parameters
    ----------
    data_shape: A 5-d tuple, assumed to be TCZYX order
    image_name: The name of the image
    channel_names: The names for each channel
    channel_colors: List of all channel colors
    channel_minmax: List of all (min, max) pairs of channel pixel
    ranges (min value of darkest pixel, max value of brightest)
    channel_startend: List of all pairs for rendering where start is
    a pixel value of darkness and end where a pixel value is
    saturated

    Returns
    -------
    Dict: An "omero" metadata object suitable for writing to ome-zarr
    """
    if channel_names is None:
        channel_names = [
            f"Channel:{image_name}:{i}" for i in range(data_shape[1])
        ]
    if channel_colors is None:
        channel_colors = [i for i in range(data_shape[1])]
    if channel_minmax is None:
        channel_minmax = [(0.0, 1.0) for _ in range(data_shape[1])]
    if channel_startend is None:
        channel_startend = channel_minmax

    ch = []
    for i in range(data_shape[1]):
        ch.append(
            {
                "active": True,
                "coefficient": 1,
                "color": f"{channel_colors[i]:06x}",
                "family": "linear",
                "inverted": False,
                "label": channel_names[i],
                "window": {
                    "end": float(channel_startend[i][1]),
                    "max": float(channel_minmax[i][1]),
                    "min": float(channel_minmax[i][0]),
                    "start": float(channel_startend[i][0]),
                },
            }
        )

    omero = {
        "id": 1,  # ID in OMERO
        "name": image_name,  # Name as shown in the UI
        "version": "0.4",  # Current version
        "channels": ch,
        "rdefs": {
            "defaultT": 0,  # First timepoint to show the user
            "defaultZ": data_shape[2] // 2,  # First Z section to show the user
            "model": "color",  # "color" or "greyscale"
        },
    }
    return omero


def _compute_scales(
    scale_num_levels: int,
    scale_factor: Tuple[float, float, float],
    pixelsizes: Tuple[float, float, float],
    chunks: Tuple[int, int, int, int, int],
    data_shape: Tuple[int, int, int, int, int],
    translation: Optional[List[float]] = None,
) -> Tuple[List, List]:
    """Generate the list of coordinate transformations and associated chunk options.

    Parameters
    ----------
    scale_num_levels: the number of downsampling levels
    scale_factor: a tuple of scale factors in each spatial dimension (Z, Y, X)
    pixelsizes: a list of pixel sizes in each spatial dimension (Z, Y, X)
    chunks: a 5D tuple of integers with size of each chunk dimension (T, C, Z, Y, X)
    data_shape: a 5D tuple of the full resolution image's shape
    translation: a 5 element list specifying the offset in physical units in each dimension

    Returns
    -------
    A tuple of the coordinate transforms and chunk options
    """
    transforms = [
        [
            # the voxel size for the first scale level
            {
                "type": "scale",
                "scale": [
                    1.0,
                    1.0,
                    pixelsizes[0],
                    pixelsizes[1],
                    pixelsizes[2],
                ],
            }
        ]
    ]
    if translation is not None:
        transforms[0].append(
            {"type": "translation", "translation": translation}
        )
    chunk_sizes = []
    lastz = data_shape[2]
    lasty = data_shape[3]
    lastx = data_shape[4]
    opts = dict(
        chunks=(
            1,
            1,
            min(lastz, chunks[2]),
            min(lasty, chunks[3]),
            min(lastx, chunks[4]),
        )
    )
    chunk_sizes.append(opts)
    if scale_num_levels > 1:
        for i in range(scale_num_levels - 1):
            last_transform = transforms[-1][0]
            last_scale = cast(List, last_transform["scale"])
            transforms.append(
                [
                    {
                        "type": "scale",
                        "scale": [
                            1.0,
                            1.0,
                            last_scale[2] * scale_factor[0],
                            last_scale[3] * scale_factor[1],
                            last_scale[4] * scale_factor[2],
                        ],
                    }
                ]
            )
            if translation is not None:
                transforms[-1].append(
                    {"type": "translation", "translation": translation}
                )
            lastz = int(np.ceil(lastz / scale_factor[0]))
            lasty = int(np.ceil(lasty / scale_factor[1]))
            lastx = int(np.ceil(lastx / scale_factor[2]))
            opts = dict(
                chunks=(
                    1,
                    1,
                    min(lastz, chunks[2]),
                    min(lasty, chunks[3]),
                    min(lastx, chunks[4]),
                )
            )
            chunk_sizes.append(opts)

    return transforms, chunk_sizes


def _get_axes_5d(
    time_unit: str = "millisecond", space_unit: str = "micrometer"
) -> List[Dict]:
    """Generate the list of axes.

    Parameters
    ----------
    time_unit: the time unit string, e.g., "millisecond"
    space_unit: the space unit string, e.g., "micrometer"

    Returns
    -------
    A list of dictionaries for each axis
    """
    axes_5d = [
        {"name": "t", "type": "time", "unit": f"{time_unit}"},
        {"name": "c", "type": "channel"},
        {"name": "z", "type": "space", "unit": f"{space_unit}"},
        {"name": "y", "type": "space", "unit": f"{space_unit}"},
        {"name": "x", "type": "space", "unit": f"{space_unit}"},
    ]
    return axes_5d


def write_ome_ngff_metadata(
    group: zarr.Group,
    arr: da.Array,
    image_name: str,
    n_lvls: int,
    scale_factors: tuple,
    voxel_size: tuple,
    channel_names: List[str] = None,
    channel_colors: List[str] = None,
    channel_minmax: List[float] = None,
    channel_startend: List[float] = None,
    metadata: dict = None,
):
    """
    Write OME-NGFF metadata to a Zarr group.

    Parameters
    ----------
    group : zarr.Group
        The output Zarr group.
    arr : array-like
        The input array.
    image_name : str
        The name of the image.
    n_lvls : int
        The number of pyramid levels.
    scale_factors : tuple
        The scale factors for downsampling along each dimension.
    voxel_size : tuple
        The voxel size along each dimension.
    channel_names: List[str]
        List of channel names to add to the OMENGFF metadata
    channel_colors: List[str]
        List of channel colors to visualize the data
    chanel_minmax: List[float]
        List of channel min and max values based on the
        image dtype
    channel_startend: List[float]
        List of the channel start and end metadata. This is
        used for visualization. The start and end range might be
        different from the min max and it is usually inside the
        range
    metadata: dict
        Extra metadata to write in the OME-NGFF metadata
    """
    if metadata is None:
        metadata = {}
    fmt = CurrentFormat()

    # Building the OMERO metadata
    ome_json = _build_ome(
        arr.shape,
        image_name,
        channel_names=channel_names,
        channel_colors=channel_colors,
        channel_minmax=channel_minmax,
        channel_startend=channel_startend,
    )
    group.attrs["omero"] = ome_json
    axes_5d = _get_axes_5d()
    coordinate_transformations, chunk_opts = _compute_scales(
        n_lvls, scale_factors, voxel_size, arr.chunksize, arr.shape, None
    )
    fmt.validate_coordinate_transformations(
        arr.ndim, n_lvls, coordinate_transformations
    )
    # Setting coordinate transfomations
    datasets = [{"path": str(i)} for i in range(n_lvls)]
    if coordinate_transformations is not None:
        for dataset, transform in zip(datasets, coordinate_transformations):
            dataset["coordinateTransformations"] = transform

    # Writing the multiscale metadata
    write_multiscales_metadata(group, datasets, fmt, axes_5d, **metadata)


def create_smartspim_opts(codec: str, compression_level: int) -> dict:
    """
    Creates SmartSPIM options for writing
    the OMEZarr.

    Parameters
    ----------
    codec: str
        Image codec used to write the image

    compression_level: int
        Compression level for the image

    Returns
    -------
    dict
        Dictionary with the blosc compression
        to write the SmartSPIM image
    """
    return {
        "compressor": blosc.Blosc(
            cname=codec, clevel=compression_level, shuffle=blosc.SHUFFLE
        )
    }


def _get_pyramid_metadata():
    """
    Gets the image pyramid metadata
    using xarray_multiscale package
    """
    return {
        "metadata": {
            "description": "Downscaling implementation based on the windowed mean of the original array",
            "method": "xarray_multiscale.reducers.windowed_mean",
            "version": str(xarray_multiscale.__version__),
            "args": "[false]",
            # No extra parameters were used different
            # from the orig. array and scales
            "kwargs": {},
        }
    }


def compute_pyramid(
    data: dask.array.core.Array,
    n_lvls: int,
    scale_axis: Tuple[int],
    chunks: Union[str, Sequence[int], Dict[Hashable, int]] = "auto",
) -> List[dask.array.core.Array]:
    """
    Computes the pyramid levels given an input full resolution image data

    Parameters
    ------------------------

    data: dask.array.core.Array
        Dask array of the image data

    n_lvls: int
        Number of downsampling levels
        that will be applied to the original image

    scale_axis: Tuple[int]
        Scaling applied to each axis

    chunks: Union[str, Sequence[int], Dict[Hashable, int]]
        chunksize that will be applied to the multiscales
        Default: "auto"

    Returns
    ------------------------

    Tuple[List[dask.array.core.Array], Dict]:
        List with the downsampled image(s) and dictionary
        with image metadata
    """

    metadata = _get_pyramid_metadata()

    pyramid = xarray_multiscale.multiscale(
        array=data,
        reduction=xarray_multiscale.reducers.windowed_mean,  # func
        scale_factors=scale_axis,  # scale factors
        preserve_dtype=True,
        chunks=chunks,
    )[:n_lvls]

    return [pyramid_level.data for pyramid_level in pyramid], metadata


def wavelength_to_hex(wavelength: int) -> int:
    """
    Converts wavelength to corresponding color hex value.

    Parameters
    ------------------------
    wavelength: int
        Integer value representing wavelength.

    Returns
    ------------------------
    int:
        Hex value color.
    """
    # Each wavelength key is the upper bound to a wavelgnth band.
    # Wavelengths range from 380-750nm.
    # Color map wavelength/hex pairs are generated by sampling along a CIE diagram arc.
    color_map = {
        460: 0x690AFE,  # Purple
        470: 0x3F2EFE,  # Blue-Purple
        480: 0x4B90FE,  # Blue
        490: 0x59D5F8,  # Blue-Green
        500: 0x5DF8D6,  # Green
        520: 0x5AFEB8,  # Green
        540: 0x58FEA1,  # Green
        560: 0x51FF1E,  # Green
        565: 0xBBFB01,  # Green-Yellow
        575: 0xE9EC02,  # Yellow
        580: 0xF5C503,  # Yellow-Orange
        590: 0xF39107,  # Orange
        600: 0xF15211,  # Orange-Red
        620: 0xF0121E,  # Red
        750: 0xF00050,
    }  # Pink

    for ub, hex_val in color_map.items():
        if wavelength < ub:  # Exclusive
            return hex_val
    return hex_val


def add_leading_dim(data: ArrayLike):
    """
    Adds a leading dimension

    Parameters
    ------------------------

    data: ArrayLike
        Input array that will have the
        leading dimension

    Returns
    ------------------------

    ArrayLike:
        Array with the new dimension in front.
    """
    return data[None, ...]


def lazy_tiff_reader(
    filename: str,
    shape: Optional[Tuple[int]] = None,
    dtype: Optional[type] = None,
):
    """
    Creates a dask array to read an image located in a specific path.

    Parameters
    ------------------------

    filename: PathLike
        Path to the image

    shape: Optional[Tuple[int]]
        Optional shape provided to the
        reader to avoid accessing to the
        metadata

    dtype: Optional[type]
        Optional array type to the reader
        to avoid accessing to the metadata

    Returns
    ------------------------

    dask.array.core.Array
        Array representing the image data
    """
    name = "imread-%s" % tokenize(filename, map(os.path.getmtime, filename))

    if dtype is None or shape is None:
        with pims.open(filename) as imgs:
            dtype = np.dtype(imgs.pixel_type)
            shape = (1,) + (len(imgs),) + imgs.frame_shape

    key = [(name,) + (0,) * len(shape)]
    value = [(add_leading_dim, (sk_imread, filename))]
    dask_arr = dict(zip(key, value))
    chunks = tuple((d,) for d in shape)

    return Array(dask_arr, name, chunks, dtype)


def smartspim_channel_zarr_writer(
    image_data: ArrayLike,
    output_path: PathLike,
    voxel_size: List[float],
    final_chunksize: List[int],
    scale_factor: List[int],
    n_lvls: int,
    channel_name: str,
    logger: logging.Logger,
    stack_name: str,
    writing_options,
    client,
):
    """
    Writes a fused SmartSPIM channel in OMEZarr
    format. This channel was read as a lazy array.

    Parameters
    ----------
    image_data: ArrayLike
        Lazy readed SmartSPIM channel data

    output_path: PathLike
        Path where we want to write the OMEZarr
        channel

    voxel_size: List[float]
        Voxel size representing the dataset

    final_chunksize: List[int]
        Final chunksize we want to use to write
        the final dataset

    codec: str
        Image codec for writing the Zarr

    compression_level: int
        Compression level

    scale_factor: List[int]
        Scale factor per axis. The dimensionality
        is organized as ZYX.

    n_lvls: int
        Number of levels on the pyramid (multiresolution)
        for better visualization

    channel_name: str
        Channel name we are currently writing

    logger: logging.Logger
        Logger object

    """
    # Getting channel color
    tmp_channel_name = channel_name.replace(".zarr", "")
    em_wav = int(tmp_channel_name.split("_")[-1])
    channel_colors = [wavelength_to_hex(em_wav)]

    # Rechunking dask array
    image_data = image_data.rechunk(final_chunksize)
    image_data = pad_array_n_d(arr=image_data)

    print(f"About to write {image_data} in {output_path}")

    # Creating Zarr dataset
    store = parse_url(path=output_path, mode="w").store
    root_group = zarr.group(store=store)

    # Using 1 thread since is in single machine.
    # Avoiding the use of multithreaded due to GIL

    if np.issubdtype(image_data.dtype, np.integer):
        np_info_func = np.iinfo

    else:
        # Floating point
        np_info_func = np.finfo

    # Getting min max metadata for the dtype
    channel_minmax = [
        (
            np_info_func(image_data.dtype).min,
            np_info_func(image_data.dtype).max,
        )
        for _ in range(image_data.shape[1])
    ]

    # Setting values for SmartSPIM
    # Ideally we would use da.percentile(image_data, (0.1, 95))
    # However, it would take so much time and resources and it is
    # not used that much on neuroglancer
    channel_startend = [(0.0, 350.0) for _ in range(image_data.shape[1])]

    new_channel_group = root_group.create_group(
        name=stack_name, overwrite=True
    )

    # Writing OME-NGFF metadata
    write_ome_ngff_metadata(
        group=new_channel_group,
        arr=image_data,
        image_name=channel_name,
        n_lvls=n_lvls,
        scale_factors=scale_factor,
        voxel_size=voxel_size,
        channel_names=[channel_name],
        channel_colors=channel_colors,
        channel_minmax=channel_minmax,
        channel_startend=channel_startend,
        metadata=_get_pyramid_metadata(),
    )

    performance_report_path = f"{output_path}/report_{stack_name}.html"

    start_time = time.time()
    # Writing zarr and performance report
    with performance_report(filename=performance_report_path):
        logger.info(f"{'='*40}Writing channel {channel_name}{'='*40}")

        # Writing zarr
        block_shape = list(
            BlockedArrayWriter.get_block_shape(
                arr=image_data, target_size_mb=12800  # 51200,
            )
        )

        # Formatting to 5D block shape
        block_shape = ([1] * (5 - len(block_shape))) + block_shape
        written_pyramid = []

        # Writing multiple levels
        for level in range(n_lvls):
            if not level:
                array_to_write = image_data

            else:
                # It's faster to write the scale and then read it back
                # to compute the next scale
                previous_scale = da.from_zarr(
                    pyramid_group, pyramid_group.chunks
                )
                new_scale_factor = (
                    [1] * (len(previous_scale.shape) - len(scale_factor))
                ) + scale_factor

                previous_scale_pyramid, _ = compute_pyramid(
                    data=previous_scale,
                    scale_axis=new_scale_factor,
                    chunks=image_data.chunksize,
                    n_lvls=2,
                )
                array_to_write = previous_scale_pyramid[-1]

            logger.info(f"[level {level}]: pyramid level: {array_to_write}")

            # Create the scale dataset
            pyramid_group = new_channel_group.create_dataset(
                name=level,
                shape=array_to_write.shape,
                chunks=array_to_write.chunksize,
                dtype=array_to_write.dtype,
                compressor=writing_options,
                dimension_separator="/",
                overwrite=True,
            )

            # Block Zarr Writer
            BlockedArrayWriter.store(
                array_to_write, pyramid_group, block_shape
            )
            written_pyramid.append(array_to_write)

    end_time = time.time()
    logger.info(f"Time to write the dataset: {end_time - start_time}")
    logger.info(f"Written pyramid: {written_pyramid}")


def convert_stacks_to_ome_zarr(channel_path, logger, output_path):
    """
    Converts image stacks from PNG to OMEZarr.

    Parameters
    ----------
    channel_path: str
        Path where the stacks for the channel
        are located.

    logger: logging.Logger
        Logging object

    output_path: str
        Path where we want to write the converted
        stacks to OMEZarr.

    """
    # channel_path must end with Ex_{wav}_Em_{wav}

    # Setting up local cluster
    n_workers = multiprocessing.cpu_count()
    logger.info(f"Setting {n_workers} workers")
    threads_per_worker = 1

    # Instantiating local cluster for parallel writing
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        processes=True,
        memory_limit="auto",
    )

    client = Client(cluster)

    cols = os.listdir(channel_path)
    for col in cols:
        curr_col = channel_path.joinpath(col)
        for row in os.listdir(curr_col):
            curr_row = curr_col.joinpath(row)
            delayed_stack = PngReader(
                data_path=f"{curr_row}/*.png"
            ).as_dask_array()
            print(f"Writing curr stack {curr_row}: {delayed_stack}")

            smartspim_channel_zarr_writer(
                image_data=delayed_stack,
                output_path=Path(output_path).joinpath(f"{channel_path.stem}"),
                voxel_size=[2.0, 1.8, 1.8],
                final_chunksize=(128, 128, 128),
                scale_factor=[2, 2, 2],
                codec="zstd",
                compression_level=0,
                n_lvls=4,
                channel_name=channel_path.stem,
                logger=logger,
                stack_name=f"{col}_{row.split('_')[-1]}.zarr",
                client=client,
            )


def create_logger(output_log_path: PathLike) -> logging.Logger:
    """
    Creates a logger that generates
    output logs to a specific path.

    Parameters
    ------------
    output_log_path: PathLike
        Path where the log is going
        to be stored

    Returns
    -----------
    logging.Logger
        Created logger pointing to
        the file path.
    """
    CURR_DATE_TIME = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    LOGS_FILE = f"{output_log_path}/fusion_log_{CURR_DATE_TIME}.log"

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s : %(message)s",
        datefmt="%Y-%m-%d %H:%M",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOGS_FILE, "a"),
        ],
        force=True,
    )

    logging.disable("DEBUG")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    return logger


def main():
    """
    Main function in how to use these functions
    """

    data_folder = Path(os.path.abspath("../data"))
    results_folder = Path(os.path.abspath("../results"))
    scratch_folder = Path(os.path.abspath("../scratch"))

    set_dask_config(dask_folder=scratch_folder)

    logger = create_logger(output_log_path=results_folder)

    BASE_PATH = data_folder.joinpath(
        "SmartSPIM_692911_2023-10-23_11-27-30/SmartSPIM"
    )

    channels = os.listdir(str(BASE_PATH))

    for channel in channels:
        convert_stacks_to_ome_zarr(
            channel_path=BASE_PATH.joinpath(channel),
            logger=logger,
            output_path=results_folder,
        )


if __name__ == "__main__":
    main()
