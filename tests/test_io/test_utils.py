"""
Unit tests of io utilities
"""

import os
import unittest
from pathlib import Path

import dask.array as da
import numpy as np

from aind_smartspim_data_transformation.io import utils

RESOURCES_DIR = (
    Path(os.path.dirname(os.path.realpath(__file__))) / ".." / "resources"
)

JSON_FILE_PATH = RESOURCES_DIR / "local_json.json"


class IoUtilitiesTest(unittest.TestCase):
    """Class for testing the io utilities"""

    def test_add_leading_dim(self):
        """
        Tests that a new dimension is added
        to the array.
        """
        test_arr = da.zeros((2, 2), dtype=np.uint8)
        transformed_arr = utils.add_leading_dim(data=test_arr)

        self.assertEqual(test_arr.ndim + 1, transformed_arr.ndim)

    def test_extract_data(self):
        """
        Tests the array data is extracted
        when there are expanded dimensions.
        """
        test_arr = da.zeros((1, 1, 1, 2, 2), dtype=np.uint8)
        transformed_arr_no_lead = utils.extract_data(arr=test_arr)
        transformed_arr_with_lead = utils.extract_data(
            arr=test_arr, last_dimensions=3
        )

        self.assertEqual(2, transformed_arr_no_lead.ndim)
        self.assertEqual(test_arr.shape[-2:], transformed_arr_no_lead.shape)

        self.assertEqual(3, transformed_arr_with_lead.ndim)
        self.assertEqual(test_arr.shape[-3:], transformed_arr_with_lead.shape)

    def test_extract_data_fail(self):
        """
        Tests failure of extract data
        """
        test_arr = da.zeros((2, 2), dtype=np.uint8)

        with self.assertRaises(ValueError):
            utils.extract_data(arr=test_arr, last_dimensions=3)

    def test_pad_array(self):
        """Tests padding an array"""
        test_arr = da.zeros((2, 2), dtype=np.uint8)
        padded_test_arr = utils.pad_array_n_d(arr=test_arr)

        self.assertEqual(5, padded_test_arr.ndim)

        padded_test_arr = utils.pad_array_n_d(arr=test_arr, dim=-1)
        self.assertEqual(test_arr.ndim, padded_test_arr.ndim)

    def test_pad_array_fail(self):
        """Tests padding an array"""
        test_arr = da.zeros((2, 2), dtype=np.uint8)

        with self.assertRaises(ValueError):
            utils.pad_array_n_d(arr=test_arr, dim=6)

    def test_read_json_as_dict(self):
        """
        Tests successful reading of a dictionary
        """
        expected_result = {"some_key": "some_value"}
        result = utils.read_json_as_dict(JSON_FILE_PATH)
        self.assertEqual(expected_result, result)


if __name__ == "__main__":
    unittest.main()
