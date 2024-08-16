"""Tests for the SmartSPIM data transfer"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

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
            num_of_partitions=4,
            partition_to_process=0,
        )
        cls.basic_job_settings = basic_job_settings
        cls.basic_job = SmartspimCompressionJob(
            job_settings=basic_job_settings
        )

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
