"""
Module that defines base Image Reader class
and the available metrics
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import dask.array as da
import pims
from dask_image.imread import imread as daimread

from aind_smartspim_data_transformation.models import PathLike


class ImageReader(ABC):
    """
    Abstract class to create image readers
    classes
    """

    def __init__(self, data_path: PathLike) -> None:
        """
        Class constructor of image reader.

        Parameters
        ------------------------
        data_path: PathLike
            Path where the image is located

        """

        self.__data_path = data_path
        super().__init__()

    @abstractmethod
    def as_dask_array(self, chunk_size: Optional[Any] = None) -> da.Array:
        """
        Abstract method to return the image as a dask array.

        Parameters
        ------------------------
        chunk_size: Optional[Any]
            If provided, the image will be rechunked to the desired
            chunksize

        Returns
        ------------------------
        da.Array
            Dask array with the image

        """

    @abstractmethod
    def close_handler(self) -> None:
        """
        Abstract method to close the image hander when it's necessary.

        """

    @abstractmethod
    def shape(self) -> Tuple:
        """
        Abstract method to return the shape of the image.

        Returns
        ------------------------
        Tuple
            Tuple with the shape of the image

        """

    @abstractmethod
    def chunks(self) -> Tuple:
        """
        Abstract method to return the chunks of the image if it's possible.

        Returns
        ------------------------
        Tuple
            Tuple with the chunks of the image

        """

    @property
    def data_path(self) -> PathLike:
        """
        Getter to return the path where the image is located.

        Returns
        ------------------------
        PathLike
            Path of the image

        """
        return self.__data_path

    @data_path.setter
    def data_path(self, new_data_path: PathLike) -> None:
        """
        Setter of the path attribute where the image is located.

        Parameters
        ------------------------
        new_data_path: PathLike
            New path of the image

        """
        self.__data_path = new_data_path


class PngTiffReader(ImageReader):
    """
    PngTiffReader class
    """

    def __init__(self, data_path: PathLike) -> None:
        """
        Class constructor of image PNG reader.

        Parameters
        ------------------------
        data_path: PathLike
            Path where the image is located

        """
        super().__init__(data_path)

    def as_dask_array(self, chunk_size: Optional[Any] = None) -> da.Array:
        """
        Method to return the image as a dask array.

        Parameters
        ------------------------
        chunk_size: Optional[Any]
            If provided, the image will be rechunked to the desired
            chunksize

        Returns
        ------------------------
        da.Array
            Dask array with the image

        """
        return daimread(self.data_path, arraytype="numpy")

    @property
    def shape(self) -> Tuple:
        """
        Abstract method to return the shape of the image.

        Returns
        ------------------------
        Tuple
            Tuple with the shape of the image

        """
        with pims.open(str(self.data_path)) as imgs:
            shape = (len(imgs),) + imgs.frame_shape

        return shape

    @property
    def chunks(self) -> Tuple:
        """
        Abstract method to return the chunks of the image if it's possible.

        Returns
        ------------------------
        Tuple
            Tuple with the chunks of the image

        """
        return self.as_dask_array().chunksize

    def close_handler(self) -> None:
        """
        Closes image handler
        """
        pass

    def __del__(self) -> None:
        """Overriding destructor to safely close image"""
        self.close_handler()
