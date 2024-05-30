"""
This module defines a class that takes
big chunks (compilation of chunks) from
a dask array and writes it on disk in
zarr format
"""

from typing import Generator, Tuple

import dask.array as da
import numpy as np
from numpy.typing import ArrayLike


def _get_size(shape: Tuple[int, ...], itemsize: int) -> int:
    """
    Return the size of an array with the given shape, in bytes
    Args:
        shape: the shape of the array
        itemsize: number of bytes per array element
    Returns:
        the size of the array, in bytes
    """
    if any(s <= 0 for s in shape):
        raise ValueError("shape must be > 0 in all dimensions")
    return np.product(shape) * itemsize


def _closer_to_target(
    shape1: Tuple[int, ...],
    shape2: Tuple[int, ...],
    target_bytes: int,
    itemsize: int,
) -> Tuple[int, ...]:
    """
    Given two shapes with the same number of dimensions,
    find which one is closer to target_bytes.
    Args:
        shape1: the first shape
        shape2: the second shape
        target_bytes: the target size for the returned shape
        itemsize: number of bytes per array element
    """
    size1 = _get_size(shape1, itemsize)
    size2 = _get_size(shape2, itemsize)
    if abs(size1 - target_bytes) < abs(size2 - target_bytes):
        return shape1
    return shape2


def expand_chunks(
    chunks: Tuple[int, int, int],
    data_shape: Tuple[int, int, int],
    target_size: int,
    itemsize: int,
    mode: str = "iso",
) -> Tuple[int, int, int]:
    """
    Given the shape and chunk size of a pre-chunked 3D array, determine the optimal chunk shape
    closest to target_size. Expanded chunk dimensions are an integer multiple of the base chunk dimension,
    to ensure optimal access patterns.
    Args:
        chunks: the shape of the input array chunks
        data_shape: the shape of the input array
        target_size: target chunk size in bytes
        itemsize: the number of bytes per array element
        mode: chunking strategy. Must be one of "cycle", or "iso"
    Returns:
        the optimal chunk shape
    """
    if any(c < 1 for c in chunks):
        raise ValueError("chunks must be >= 1 for all dimensions")
    if any(s < 1 for s in data_shape):
        raise ValueError("data_shape must be >= 1 for all dimensions")
    if any(c > s for c, s in zip(chunks, data_shape)):
        raise ValueError(
            "chunks cannot be larger than data_shape in any dimension"
        )
    if target_size <= 0:
        raise ValueError("target_size must be > 0")
    if itemsize <= 0:
        raise ValueError("itemsize must be > 0")
    if mode == "cycle":
        # get the spatial dimensions only
        current = np.array(chunks, dtype=np.uint64)
        prev = current.copy()
        idx = 0
        ndims = len(current)
        while _get_size(current, itemsize) < target_size:
            prev = current.copy()
            current[idx % ndims] = min(
                data_shape[idx % ndims], current[idx % ndims] * 2
            )
            idx += 1
            if all(c >= s for c, s in zip(current, data_shape)):
                break
        expanded = _closer_to_target(current, prev, target_size, itemsize)
    elif mode == "iso":
        initial = np.array(chunks, dtype=np.uint64)
        current = initial
        prev = current
        i = 2
        while _get_size(current, itemsize) < target_size:
            prev = current
            current = initial * i
            current = (
                min(data_shape[0], current[0]),
                min(data_shape[1], current[1]),
                min(data_shape[2], current[2]),
            )
            i += 1
            if all(c >= s for c, s in zip(current, data_shape)):
                break
        expanded = _closer_to_target(current, prev, target_size, itemsize)
    else:
        raise ValueError(f"Invalid mode {mode}")

    return tuple(int(d) for d in expanded)


class BlockedArrayWriter:
    """
    Static class to write a lazy array
    in big chunks to OMEZarr
    """

    @staticmethod
    def gen_slices(
        arr_shape: Tuple[int, ...], block_shape: Tuple[int, ...]
    ) -> Generator:
        """
        Generate a series of slices that can be used to traverse an array in blocks of a given shape.

        The method generates tuples of slices, each representing a block of the array. The blocks are generated by
        iterating over the array in steps of the block shape along each dimension.

        Parameters
        ----------
        arr_shape : tuple of int
            The shape of the array to be sliced.

        block_shape : tuple of int
            The desired shape of the blocks. This should be a tuple of integers representing the size of each
            dimension of the block. The length of `block_shape` should be equal to the length of
            `arr_shape`. If the array shape is not divisible by the block shape along a dimension, the last slice
            along that dimension is truncated.

        Returns
        -------
        generator of tuple of slice
            A generator yielding tuples of slices. Each tuple can be used to index the input array.
        """
        if len(arr_shape) != len(block_shape):
            raise Exception(
                "array shape and block shape have different lengths"
            )

        def _slice_along_dim(dim: int) -> Generator:
            """A helper generator function that slices along one dimension."""
            # Base case: if the dimension is beyond the last one, return an empty tuple
            if dim >= len(arr_shape):
                yield ()
            else:
                # Iterate over the current dimension in steps of the block size
                for i in range(0, arr_shape[dim], block_shape[dim]):
                    # Calculate the end index for this block
                    end_i = min(i + block_shape[dim], arr_shape[dim])
                    # Generate slices for the remaining dimensions
                    for rest in _slice_along_dim(dim + 1):
                        yield (slice(i, end_i),) + rest

        # Start slicing along the first dimension
        return _slice_along_dim(0)

    @staticmethod
    def store(
        in_array: da.Array, out_array: ArrayLike, block_shape: tuple
    ) -> None:
        """
        Partitions the last 3 dimensions of a Dask array into non-overlapping blocks and
        writes them sequentially to a Zarr array. This is meant to reduce the scheduling burden
        for massive (terabyte-scale) arrays.

        :param in_array: The input Dask array
        :param block_shape: Tuple of (block_depth, block_height, block_width)
        :param out_array: The output array
        """
        # Iterate through the input array in steps equal to the block shape dimensions
        for sl in BlockedArrayWriter.gen_slices(in_array.shape, block_shape):
            block = in_array[sl]
            da.store(
                block,
                out_array,
                regions=sl,
                lock=False,
                compute=True,
                return_stored=False,
            )

    @staticmethod
    def get_block_shape(arr, target_size_mb=409600, mode="cycle"):
        """
        Given the shape and chunk size of a pre-chunked array, determine the optimal block shape
        closest to target_size. Expanded block dimensions are an integer multiple of the chunk dimension
        to ensure optimal access patterns.
        Args:
            arr: the input array
            target_size_mb: target block size in megabytes, default is 409600
            mode: strategy. Must be one of "cycle", or "iso"
        Returns:
            the block shape
        """
        if isinstance(arr, da.Array):
            chunks = arr.chunksize[-3:]
        else:
            chunks = arr.chunks[-3:]

        return expand_chunks(
            chunks,
            arr.shape[-3:],
            target_size_mb * 1024**2,
            arr.itemsize,
            mode,
        )
