"""Module to handle smartspim data compression"""

import logging
import os
import shutil
import sys
from pathlib import Path
from time import time
from typing import Any, List, Optional

from aind_data_transformation.core import GenericEtl, JobResponse, get_parser
from numcodecs.blosc import Blosc

from aind_smartspim_data_transformation.compress.png_to_zarr import (
    smartspim_channel_zarr_writer,
)
from aind_smartspim_data_transformation.io import utils
from aind_smartspim_data_transformation.io.readers import PngReader
from aind_smartspim_data_transformation.models import (
    CompressorName,
    SmartspimJobSettings,
)

logging.basicConfig(level=os.getenv("LOG_LEVEL", "WARNING"))


class SmartspimCompressionJob(GenericEtl[SmartspimJobSettings]):
    """Job to handle compressing and uploading SmartSPIM data."""

    @staticmethod
    def partition_list(
        lst: List[Any], num_of_partitions: int
    ) -> List[List[Any]]:
        """Partitions a list"""
        accumulated_list = []
        for _ in range(num_of_partitions):
            accumulated_list.append([])
        for list_item_index, list_item in enumerate(lst):
            a_index = list_item_index % num_of_partitions
            accumulated_list[a_index].append(list_item)
        return accumulated_list

    def _get_partitioned_list_of_stack_paths(self) -> List[List[Path]]:
        """Scans through the input source and partitions a list of stack
        paths that it finds there."""
        all_stack_paths = []
        total_counter = 0
        for channel in [
            p
            for p in Path(self.job_settings.input_source)
            .joinpath("SmartSPIM")
            .iterdir()
            if p.is_dir()
        ]:
            for col in [p for p in channel.iterdir() if p.is_dir()]:
                for col_and_row in [p for p in col.iterdir() if p.is_dir()]:
                    total_counter += 1
                    all_stack_paths.append(col_and_row)
        # Important to sort paths so every node computes the same list
        all_stack_paths.sort(key=lambda x: str(x))
        return self.partition_list(
            all_stack_paths, self.job_settings.num_of_partitions
        )

    @staticmethod
    def _get_voxel_resolution(acquisition_path: Path) -> List[float]:
        """Get the voxel resolution from an acquisition.json file."""

        if not acquisition_path.is_file():
            raise FileNotFoundError(
                f"acquisition.json file not found at: {acquisition_path}"
            )

        acquisition_config = utils.read_json_as_dict(acquisition_path)

        # Grabbing a tile with metadata from acquisition - we assume all
        # dataset was acquired with the same resolution
        tile_coord_transforms = acquisition_config["tiles"][0][
            "coordinate_transformations"
        ]

        scale_transform = [
            x["scale"] for x in tile_coord_transforms if x["type"] == "scale"
        ][0]

        x = float(scale_transform[0])
        y = float(scale_transform[1])
        z = float(scale_transform[2])

        return [z, y, x]

    def _get_compressor(self) -> Optional[Blosc]:
        """
        Utility method to construct a compressor class.
        Returns
        -------
        Blosc | None
          An instantiated Blosc compressor. Return None if not set in configs.

        """
        if self.job_settings.compressor_name == CompressorName.BLOSC:
            return Blosc(**self.job_settings.compressor_kwargs)
        else:
            return None

    def _write_stacks(self, stacks_to_process: List) -> None:
        """
        Write a list of stacks.
        Parameters
        ----------
        stacks_to_process : List

        Returns
        -------
        None

        """
        compressor = self._get_compressor()
        acquisition_path = Path(self.job_settings.input_source).joinpath(
            "acquisition.json"
        )
        voxel_size_zyx = self._get_voxel_resolution(
            acquisition_path=acquisition_path
        )
        logging.info(
            f"Stacks to process: {stacks_to_process}, - Voxel res: "
            f"{voxel_size_zyx}"
        )
        for stack in stacks_to_process:
            logging.info(f"Converting {stack}")
            channel_name = stack.parent.parent.name
            stack_name = stack.name

            output_path = Path(self.job_settings.output_directory).joinpath(
                channel_name
            )

            delayed_stack = PngReader(
                data_path=f"{stack}/*.png"
            ).as_dask_array()

            smartspim_channel_zarr_writer(
                image_data=delayed_stack,
                output_path=output_path,
                voxel_size=voxel_size_zyx,
                final_chunksize=self.job_settings.chunk_size,
                scale_factor=self.job_settings.scale_factor,
                n_lvls=self.job_settings.downsample_levels,
                channel_name=channel_name,
                stack_name=f"{stack_name}.ome.zarr",
                logger=logging,
                writing_options=compressor,
            )

            if self.job_settings.s3_location is not None:
                channel_zgroup_file = output_path / ".zgroup"
                s3_channel_zgroup_file = (
                    f"{self.job_settings.s3_location}/{channel_name}/.zgroup"
                )
                logging.info(
                    f"Uploading {channel_zgroup_file} to "
                    f"{s3_channel_zgroup_file}"
                )
                utils.copy_file_to_s3(
                    channel_zgroup_file, s3_channel_zgroup_file
                )
                ome_zarr_stack_name = f"{stack_name}.ome.zarr"
                ome_zarr_stack_path = output_path.joinpath(ome_zarr_stack_name)
                s3_stack_dir = (
                    f"{self.job_settings.s3_location}/{channel_name}/"
                    f"{ome_zarr_stack_name}"
                )
                logging.info(
                    f"Uploading {ome_zarr_stack_path} to {s3_stack_dir}"
                )
                utils.sync_dir_to_s3(ome_zarr_stack_path, s3_stack_dir)
                logging.info(f"Removing: {ome_zarr_stack_path}")
                # Remove stack if uploaded to s3. We can potentially do all
                # the stacks in the partition in parallel using dask to speed
                # this up
                shutil.rmtree(ome_zarr_stack_path)

    def _upload_derivatives_folder(self):
        """
        Uploads the derivatives folder inside of
        the SPIM folder in the cloud.
        """
        s3_derivatives_dir = f"{self.job_settings.s3_location}/derivatives"
        derivatives_path = Path(self.job_settings.input_source).joinpath(
            "derivatives"
        )

        if not derivatives_path.exists():
            raise FileNotFoundError(f"{derivatives_path} does not exist.")

        if self.job_settings.s3_location is not None:
            logging.info(
                f"Uploading {derivatives_path} to {s3_derivatives_dir}"
            )
            utils.sync_dir_to_s3(derivatives_path, s3_derivatives_dir)
            logging.info(f"{derivatives_path} uploaded to s3.")

    def run_job(self):
        """Main entrypoint to run the job."""
        job_start_time = time()

        partitioned_list = self._get_partitioned_list_of_stack_paths()
        # Upload derivatives folder
        if self.job_settings.partition_to_process == 0:
            self._upload_derivatives_folder()

        stacks_to_process = partitioned_list[
            self.job_settings.partition_to_process
        ]

        self._write_stacks(stacks_to_process=stacks_to_process)
        total_job_duration = time() - job_start_time
        return JobResponse(
            status_code=200, message=f"Job finished in {total_job_duration}"
        )


# TODO: Add this to core aind_data_transformation class
def job_entrypoint(sys_args: list):
    """Main function"""
    parser = get_parser()
    cli_args = parser.parse_args(sys_args)
    if cli_args.job_settings is not None:
        job_settings = SmartspimJobSettings.model_validate_json(
            cli_args.job_settings
        )
    elif cli_args.config_file is not None:
        job_settings = SmartspimJobSettings.from_config_file(
            cli_args.config_file
        )
    else:
        # Construct settings from env vars
        job_settings = SmartspimJobSettings()
    job = SmartspimCompressionJob(job_settings=job_settings)
    job_response = job.run_job()
    logging.info(job_response.model_dump_json())


if __name__ == "__main__":
    job_entrypoint(sys.argv[1:])
