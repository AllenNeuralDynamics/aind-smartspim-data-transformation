"""Tests for the SmartSPIM data transfer"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from numcodecs.blosc import Blosc

from aind_smartspim_data_transformation.smartspim_job import (
    SmartspimCompressionJob,
    SmartspimJobSettings,
)

TEST_DIR = Path(os.path.dirname(os.path.realpath(__file__))) / "resources"
DATA_DIR = TEST_DIR / "SmartSPIM_000000_2024-06-05_07-56-54"


class SmartspimCompressionTest(unittest.TestCase):
    """Class for testing the data transform"""

    @classmethod
    def setUpClass(cls) -> None:
        """Setup basic job settings and job that can be used across tests"""
        # Folder to test the zarr writing from PNGs
        cls.temp_folder = tempfile.mkdtemp(prefix="unittest_")

        basic_job_settings = SmartspimJobSettings(
            input_source=DATA_DIR,
            output_directory=Path(cls.temp_folder),
        )
        cls.basic_job_settings = basic_job_settings
        cls.basic_job = SmartspimCompressionJob(
            job_settings=basic_job_settings
        )

    def test_get_delayed_channel_stack(self):
        """Tests reading stacks of png images."""

        raw_path = self.basic_job.job_settings.input_source / "SmartSPIM"

        channel_paths = [
            Path(raw_path).joinpath(folder) for folder in os.listdir(raw_path)
        ]

        # Get channel stack iterators and delayed arrays
        read_delayed_channel_stacks = (
            self.basic_job._get_delayed_channel_stack(
                channel_paths=channel_paths,
                output_dir=self.basic_job.job_settings.output_directory,
            )
        )

        stacked_shape = (2, 1600, 2000)
        dtype = np.uint16
        extensions = [".ome", ".zarr"]

        for delayed_arr, _, stack_name in read_delayed_channel_stacks:
            self.assertEqual(stacked_shape, delayed_arr.shape)
            self.assertEqual(dtype, delayed_arr.dtype)
            self.assertEqual(extensions, Path(stack_name).suffixes)

    def test_compressor(self):
        """Test compression with Blosc"""
        compressor = self.basic_job._get_compressor()
        current_blosc = Blosc(**self.basic_job.job_settings.compressor_kwargs)
        self.assertEqual(compressor, current_blosc)

    def test_getting_compressor_fail(self):
        """Test failed compression with Blosc"""

        with self.assertRaises(Exception):
            # Failed blosc compressor
            failed_basic_job_settings = SmartspimJobSettings(
                input_source=DATA_DIR,
                output_directory=Path("output_dir"),
                compressor_name="not_blosc",
            )

            failed_basic_job_settings = failed_basic_job_settings
            SmartspimCompressionJob(job_settings=failed_basic_job_settings)

    def test_run_job(self):
        """Tests SmartSPIM compression and zarr writing"""
        self.basic_job.run_job()

    @classmethod
    def tearDownClass(cls) -> None:
        """Tear down class method to clean up"""
        if os.path.exists(cls.temp_folder):
            shutil.rmtree(cls.temp_folder, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
