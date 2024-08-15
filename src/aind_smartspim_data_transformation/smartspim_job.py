"""Module to handle smartspim data compression"""

import logging
import sys
from pathlib import Path
from typing import List, Optional, Any

from aind_data_transformation.core import (
    BasicJobSettings,
    GenericEtl,
    JobResponse,
    get_parser,
)
from numcodecs.blosc import Blosc
from time import time
from pydantic import Field
from aind_smartspim_data_transformation.compress.png_to_zarr import (
    smartspim_channel_zarr_writer,
)
from aind_smartspim_data_transformation.io import PngReader, utils
from aind_smartspim_data_transformation.models import CompressorName


class SmartspimJobSettings(BasicJobSettings):
    """SmartspimCompressionJob settings."""

    input_source: str = Field(
        ...,
        description=(
            "Source of the SmartSPIM channel data. For example, "
            "/scratch/SmartSPIM_695464_2023-10-18_20-30-30/SmartSPIM"
        ),
    )
    output_directory: str = Field(
        ...,
        description=("Where to write the data to locally."),
    )
    s3_location: Optional[str] = None
    num_of_partitions: int = Field(
        ...,
        description=(
            "This script will generate a list of individual stacks, "
            "and then partition the list into this number of partitions."
        ),
    )
    partition_to_process: int = Field(
        ...,
        description=("Which partition of stacks to process. "),
    )
    compressor_name: CompressorName = Field(
        default=CompressorName.BLOSC,
        description="Type of compressor to use.",
        title="Compressor Name.",
    )
    # It will be safer if these kwargs fields were objects with known schemas
    compressor_kwargs: dict = Field(
        default={"cname": "zstd", "clevel": 3, "shuffle": Blosc.SHUFFLE},
        description="Arguments to be used for the compressor.",
        title="Compressor Kwargs",
    )
    compress_job_save_kwargs: dict = Field(
        default={"n_jobs": -1},  # -1 to use all available cpu cores.
        description="Arguments for recording save method.",
        title="Compress Job Save Kwargs",
    )
    chunk_size: List[int] = Field(
        default=[128, 128, 128],  # Default list with three integers
        description="Chunk size in axis, a list of three integers",
        title="Chunk Size",
    )
    scale_factor: List[int] = Field(
        default=[2, 2, 2],  # Default list with three integers
        description="Scale factors in axis, a list of three integers",
        title="Scale Factors",
    )
    downsample_levels: int = Field(
        default=4,
        description="The number of levels of the image pyramid",
        title="Downsample Levels",
    )


class SmartspimCompressionJob(GenericEtl[SmartspimJobSettings]):

    @staticmethod
    def partition_list(lst: List[Any], num_of_partitions: int) -> List[List[Any]]:
        """Partitions a list"""
        accumulated_list = []
        for _ in range(num_of_partitions):
            accumulated_list.append([])
        for list_item_index, list_item in enumerate(lst):
            a_index = list_item_index % num_of_partitions
            accumulated_list[a_index].append(list_item)
        return accumulated_list

    def _get_partitioned_list_of_stack_paths(self) -> List[List[Path]]:
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
        return self.partition_list(all_stack_paths, self.job_settings.num_of_partitions)

    def _get_voxel_resolution(self, acquisition_path: str) -> List[int]:

        if not acquisition_path.exists():
            raise ValueError(
                f"Acquisition path {acquisition_path} does not exist."
            )

        acquisition_config = utils.read_json_as_dict(acquisition_path)

        # Grabbing a tile with metadata from acquisition - we assume all dataset
        # was acquired with the same resolution
        tile_coord_transforms = acquisition_config["tiles"][0][
            "coordinate_transformations"
        ]

        scale_transform = [
            x["scale"] for x in tile_coord_transforms if x["type"] == "scale"
        ][0]

        x = float(scale_transform[0])
        y = float(scale_transform[1])
        z = float(scale_transform[2])

        return z, y, x

    def _get_compressor(self) -> Blosc:
        """
        Utility method to construct a compressor class.
        Returns
        -------
        Blosc
          An instantiated Blosc compressor.

        """
        if self.job_settings.compressor_name == CompressorName.BLOSC:
            return Blosc(**self.job_settings.compressor_kwargs)

        return None

    def run_job(self):
        job_start_time = time()
        acquisition_path = Path(self.job_settings.input_source).joinpath(
            "acquisition.json"
        )
        voxel_size_zyx = self._get_voxel_resolution(
            acquisition_path=acquisition_path
        )

        partitioned_list = self._get_partitioned_list_of_stack_paths()
        stacks_to_process = partitioned_list[
            self.job_settings.partition_to_process
        ]

        # Getting compressors
        compressor = self._get_compressor()

        print(
            f"Stacks to process: {stacks_to_process}, - Voxel res: {voxel_size_zyx}"
        )

        for stack in stacks_to_process:
            logging.info(f"Converting {stack}")
            channel_name = stack.parent.parent.name
            stack_name = stack.name

            if self.job_settings.s3_location is not None:
                output_path = (
                    f"{self.job_settings.s3_location.rstrip('/')}/"
                    f"{channel_name.rstrip}"
                )
            else:
                output_path = Path(self.job_settings.output_directory).joinpath(channel_name)

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
            # Method to process single stack using dask
            # It'd be nice if there was an option to upload the stacks
            # and remove local files
        total_job_duration = job_start_time - time()
        return JobResponse(
            status_code=200,
            message=f"Job finished in {total_job_duration}"
        )


def main():
    """Main function"""
    sys_args = sys.argv[1:]
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
    main()

