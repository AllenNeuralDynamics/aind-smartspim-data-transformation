"""Tests for the SmartSPIM data transfer"""

import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from numcodecs.blosc import Blosc

from aind_smartspim_data_transformation.models import SmartspimJobSettings
from aind_smartspim_data_transformation.smartspim_job import (
    SmartspimCompressionJob,
    job_entrypoint,
)

RESOURCES_DIR = Path(os.path.dirname(os.path.realpath(__file__))) / "resources"

DATA_DIR = RESOURCES_DIR / "SmartSPIM_000000_2024-06-05_07-56-54"


class SmartspimCompressionTest(unittest.TestCase):
    """Class for testing the data transform"""

    @classmethod
    def setUpClass(cls) -> None:
        """Setup basic job settings and job that can be used across tests"""

        basic_job_settings = SmartspimJobSettings(
            input_source=DATA_DIR,
            output_directory="fake_output_dir",
            num_of_partitions=4,
            partition_to_process=0,
        )
        cls.basic_job_settings = basic_job_settings
        cls.basic_job = SmartspimCompressionJob(
            job_settings=basic_job_settings
        )

    def test_partition_list(self):
        """Tests partition list method"""
        test_list = [f"ID: {x}" for x in range(75)]
        output_list1 = self.basic_job.partition_list(
            test_list, num_of_partitions=5
        )
        output_list2 = self.basic_job.partition_list(
            test_list, num_of_partitions=2
        )
        flat_output1 = [x for xs in output_list1 for x in xs]
        flat_output2 = [x for xs in output_list2 for x in xs]
        self.assertEqual(5, len(output_list1))
        self.assertEqual(2, len(output_list2))
        self.assertCountEqual(test_list, flat_output1)
        self.assertCountEqual(test_list, flat_output2)

    def test_get_partitioned_list_of_stack_paths(self):
        """Tests _get_partitioned_list_of_stack_paths"""
        stack_paths = self.basic_job._get_partitioned_list_of_stack_paths()
        flat_list_of_paths = [x for xs in stack_paths for x in xs]
        channel_dir1 = DATA_DIR / "SmartSPIM" / "Ex_445_Em_469"
        channel_dir2 = DATA_DIR / "SmartSPIM" / "Ex_561_Em_600"
        expected_flat_list = [
            channel_dir1 / "432380" / "432380_504340",
            channel_dir2 / "432380" / "432380_504340",
            channel_dir1 / "432380" / "432380_530260",
            channel_dir2 / "432380" / "432380_530260",
            channel_dir1 / "464780" / "464780_504340",
            channel_dir2 / "464780" / "464780_504340",
            channel_dir1 / "464780" / "464780_530260",
            channel_dir2 / "464780" / "464780_530260",
        ]
        self.assertEqual(4, len(stack_paths))
        self.assertEqual(expected_flat_list, flat_list_of_paths)

    def test_get_voxel_resolution(self):
        """Tests _get_voxel_resolution"""
        acq_path = DATA_DIR / "acquisition.json"
        voxel_res = self.basic_job._get_voxel_resolution(acq_path)
        expected_res = [2.0, 1.8, 1.8]
        self.assertEqual(expected_res, voxel_res)

    def test_get_voxel_resolution_error(self):
        """Tests _get_voxel_resolution when file not found"""
        acq_path = DATA_DIR / "acq.json"  # No acquisition.json file here
        with self.assertRaises(FileNotFoundError):
            self.basic_job._get_voxel_resolution(acq_path)

    def test_get_compressor(self):
        """Tests _get_compressor method"""

        compressor = self.basic_job._get_compressor()
        expected_compressor = Blosc(
            cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE, blocksize=0
        )
        self.assertEqual(expected_compressor, compressor)

    def test_get_compressor_none(self):
        """Tests _get_compressor method returns None if no config set"""

        job_settings = SmartspimJobSettings.model_construct(
            compressor_name="foo"
        )
        job = SmartspimCompressionJob(job_settings=job_settings)
        compressor = job._get_compressor()
        self.assertIsNone(compressor)

    @patch(
        "aind_smartspim_data_transformation.smartspim_job"
        ".smartspim_channel_zarr_writer"
    )
    def test_run_job(self, mock_zarr_write: MagicMock):
        """Test run_job method"""
        response = self.basic_job.run_job()
        mock_zarr_write.assert_called()
        self.assertIsNotNone(response)

    @patch(
        "aind_smartspim_data_transformation.smartspim_job"
        ".SmartspimCompressionJob.run_job"
    )
    @patch("aind_smartspim_data_transformation.smartspim_job.get_parser")
    def test_job_entrypoint_job_settings(
        self, mock_get_parser: MagicMock, mock_run_job: MagicMock
    ):
        """Tests job_entrypoint with json settings"""

        json_settings = json.dumps(
            {
                "input_source": "input",
                "output_directory": "output_dir",
                "num_of_partitions": 4,
                "partition_to_process": 1,
            }
        )
        mock_parser = MagicMock()
        mock_parser.parse_args.return_value.job_settings = json_settings
        mock_get_parser.return_value = mock_parser
        sys_args = ["", "-j", f"' {json_settings}' "]
        job_entrypoint(sys_args)
        mock_run_job.return_value = {"message": "ran job"}
        # job_main(sys_args)
        # mock_run_job.assert_called()

    @patch(
        "aind_smartspim_data_transformation.smartspim_job"
        ".SmartspimCompressionJob.run_job"
    )
    @patch("aind_smartspim_data_transformation.smartspim_job.get_parser")
    def test_job_entrypoint_config_file(
        self, mock_get_parser: MagicMock, mock_run_job: MagicMock
    ):
        """Tests job_entrypoint with config file"""
        config_file = RESOURCES_DIR / "test_configs.json"
        mock_parser = MagicMock()
        mock_parser.parse_args.return_value.job_settings = None
        mock_parser.parse_args.return_value.config_file = config_file
        mock_get_parser.return_value = mock_parser
        sys_args = ["", "--config-file", f"{config_file}"]
        job_entrypoint(sys_args)
        mock_run_job.return_value = {"message": "ran job"}
        # job_main(sys_args)
        # mock_run_job.assert_called()

    @patch.dict(
        os.environ,
        {
            "TRANSFORMATION_JOB_INPUT_SOURCE": "input",
            "TRANSFORMATION_JOB_OUTPUT_DIRECTORY": "output_dir",
            "TRANSFORMATION_JOB_NUM_OF_PARTITIONS": "4",
            "TRANSFORMATION_JOB_PARTITION_TO_PROCESS": "1",
        },
        clear=True,
    )
    @patch(
        "aind_smartspim_data_transformation.smartspim_job"
        ".SmartspimCompressionJob.run_job"
    )
    @patch("aind_smartspim_data_transformation.smartspim_job.get_parser")
    def test_job_entrypoint_env_vars(
        self, mock_get_parser: MagicMock, mock_run_job: MagicMock
    ):

        mock_parser = MagicMock()
        mock_parser.parse_args.return_value.job_settings = None
        mock_parser.parse_args.return_value.config_file = None
        mock_get_parser.return_value = mock_parser
        sys_args = [""]
        job_entrypoint(sys_args)
        mock_run_job.return_value = {"message": "ran job"}

    # aind_smartspim_data_transformation.compress.png_to_zarr.smartspim_channel_zarr_writer

    # def test_get_delayed_channel_stack(self):
    #     """Tests reading stacks of png images."""
    #
    #     raw_path = self.basic_job.job_settings.input_source / "SmartSPIM"
    #
    #     channel_paths = [
    #         Path(raw_path).joinpath(folder) for folder in os.listdir(raw_path)
    #     ]
    #
    #     # Get channel stack iterators and delayed arrays
    #     read_delayed_channel_stacks = (
    #         self.basic_job._get_delayed_channel_stack(
    #             channel_paths=channel_paths,
    #             output_dir=self.basic_job.job_settings.output_directory,
    #         )
    #     )
    #
    #     stacked_shape = (2, 1600, 2000)
    #     dtype = np.uint16
    #     extensions = [".ome", ".zarr"]
    #
    #     for delayed_arr, _, stack_name in read_delayed_channel_stacks:
    #         self.assertEqual(stacked_shape, delayed_arr.shape)
    #         self.assertEqual(dtype, delayed_arr.dtype)
    #         self.assertEqual(extensions, Path(stack_name).suffixes)
    #
    # def test_compressor(self):
    #     """Test compression with Blosc"""
    #     compressor = self.basic_job._get_compressor()
    #     current_blosc = Blosc(**self.basic_job.job_settings.compressor_kwargs)
    #     self.assertEqual(compressor, current_blosc)
    #
    # def test_getting_compressor_fail(self):
    #     """Test failed compression with Blosc"""
    #
    #     with self.assertRaises(Exception):
    #         # Failed blosc compressor
    #         failed_basic_job_settings = SmartspimJobSettings(
    #             input_source=DATA_DIR,
    #             output_directory=Path("output_dir"),
    #             compressor_name="not_blosc",
    #         )
    #
    #         failed_basic_job_settings = failed_basic_job_settings
    #         SmartspimCompressionJob(job_settings=failed_basic_job_settings)
    #
    # def test_run_job(self):
    #     """Tests SmartSPIM compression and zarr writing"""
    #     self.basic_job.run_job()
    #
    # @classmethod
    # def tearDownClass(cls) -> None:
    #     """Tear down class method to clean up"""
    #     if os.path.exists(cls.temp_folder):
    #         shutil.rmtree(cls.temp_folder, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
