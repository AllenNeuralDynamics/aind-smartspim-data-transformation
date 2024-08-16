"""
Tests for the zarr writer
"""

import unittest

import dask.array as da
import numpy as np

from aind_smartspim_data_transformation.compress import zarr_writer


class ZarrWriterTest(unittest.TestCase):
    """Class for testing the zarr writer"""

    def test_get_size(self):
        """
        Tests get size method
        """
        test_arr = da.zeros((2, 2), dtype=np.uint8)

        expected_result = np.prod(test_arr.shape) * test_arr.itemsize

        arr_size = zarr_writer._get_size(
            shape=test_arr.shape, itemsize=test_arr.itemsize
        )
        self.assertEqual(expected_result, arr_size)

    def test_get_size_fail(self):
        """
        Tests get size failure
        """
        test_arr = da.zeros((0, 2), dtype=np.uint8)

        with self.assertRaises(ValueError):
            zarr_writer._get_size(
                shape=test_arr.shape, itemsize=test_arr.itemsize
            )

    def test_closer_to_target(self):
        """Tests closer to target function"""
        test_arr_1 = da.zeros((10, 10), dtype=np.uint8)
        test_arr_2 = da.zeros((20, 20), dtype=np.uint8)
        target_bytes = 4

        shape_close_target = zarr_writer._closer_to_target(
            shape1=test_arr_1.shape,
            shape2=test_arr_2.shape,
            target_bytes=target_bytes,
            itemsize=test_arr_1.itemsize,
        )

        shape_close_target_2 = zarr_writer._closer_to_target(
            shape1=test_arr_2.shape,
            shape2=test_arr_1.shape,
            target_bytes=target_bytes,
            itemsize=test_arr_1.itemsize,
        )

        self.assertEqual(test_arr_1.shape, shape_close_target)
        self.assertEqual(test_arr_1.shape, shape_close_target_2)


if __name__ == "__main__":
    unittest.main()
