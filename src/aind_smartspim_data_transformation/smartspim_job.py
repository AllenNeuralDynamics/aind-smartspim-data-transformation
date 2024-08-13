"""Module to handle smartspim data compression"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterator, List, Optional

from aind_data_transformation.core import (
    BasicJobSettings,
    GenericEtl,
    JobResponse,
    get_parser,
)
from numcodecs.blosc import Blosc
from pydantic import Field

from aind_smartspim_data_transformation.compress.dask_utils import (
    _cleanup,
    get_client,
    get_deployment,
)
from aind_smartspim_data_transformation.compress.png_to_zarr import (
    smartspim_channel_zarr_writer,
)
from aind_smartspim_data_transformation.io import PngReader, utils
from aind_smartspim_data_transformation.models import CompressorName


class SmartspimJobSettings2(BasicJobSettings):
    """SmartspimCompressionJob settings."""

    # Compress settings
    random_seed: Optional[int] = 0
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
    chunk_size: int = Field(
        default=128,
        description="Image chunk size",
        title="Image Chunk Size",
    )


class SmartspimCompressionJob2(GenericEtl[SmartspimJobSettings2]):
    """Main class to handle smartspim data compression"""

    def _get_delayed_channel_stack(
        self, channel_paths: List[str], output_dir: str
    ) -> Iterator[tuple]:
        """
        Reads a stack of PNG images into a delayed zarr dataset.

        Returns:
        Iterator[tuple]
            A generator that returns delayed PNG stacks.

        """
        for channel_path in channel_paths:

            cols = [
                col_f
                for col_f in os.listdir(channel_path)
                if Path(channel_path).joinpath(col_f).is_dir()
            ]

            for col in cols:
                curr_col = channel_path.joinpath(col)
                rows_in_cols = [
                    row_f
                    for row_f in os.listdir(curr_col)
                    if Path(curr_col).joinpath(row_f).is_dir()
                ]
                for row in rows_in_cols:
                    curr_row = curr_col.joinpath(row)
                    delayed_stack = PngReader(
                        data_path=f"{curr_row}/*.png"
                    ).as_dask_array()
                    stack_name = f"{col}_{row.split('_')[-1]}.ome.zarr"
                    stack_output_path = Path(
                        f"{output_dir}/{channel_path.stem}"
                    )

                    yield (delayed_stack, stack_output_path, stack_name)

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

    @staticmethod
    def _compress_and_write_channels(
        read_channel_stacks: Iterator[tuple],
        compressor: Blosc,
        job_kwargs: dict,
    ):
        """
        Compresses SmartSPIM image data.

        Parameters
        ----------
        read_channel_stacks: Iterator[tuple]
            Iterator that returns the delayed image stack,
            image path and stack name.
        """

        if job_kwargs["n_jobs"] == -1:
            job_kwargs["n_jobs"] = os.cpu_count()

        n_workers = job_kwargs["n_jobs"]

        # Instantiating local cluster for parallel writing
        deployment = get_deployment()
        client, _ = get_client(
            deployment,
            worker_options=None,  # worker_options,
            n_workers=n_workers,
            processes=True,
        )
        print(f"Instantiated client: {client}")

        try:
            for delayed_arr, output_path, stack_name in read_channel_stacks:
                print(
                    f"Converting {delayed_arr} from {stack_name} to {output_path}"
                )
                smartspim_channel_zarr_writer(
                    image_data=delayed_arr,
                    output_path=output_path,
                    voxel_size=[2.0, 1.8, 1.8],
                    final_chunksize=(128, 128, 128),
                    scale_factor=[2, 2, 2],
                    n_lvls=4,
                    channel_name=output_path.stem,
                    stack_name=stack_name,
                    logger=logging,
                    writing_options=compressor,
                )

        except Exception as e:
            print(f"Error converting array: {e}")

        try:
            _cleanup(deployment)
        except Exception as e:
            print(f"Error shutting down client: {e}")

    def _compress_raw_data(self) -> None:
        """Compresses smartspim data"""

        # Clip the data
        logging.info("Converting PNG to OMEZarr. This may take some minutes.")
        output_compressed_data = self.job_settings.output_directory

        raw_path = self.job_settings.input_source / "SmartSPIM"
        print(f"Raw path: {raw_path} " f"OS: {raw_path}")

        channel_paths = [
            Path(raw_path).joinpath(folder)
            for folder in os.listdir(raw_path)
            if Path(raw_path).joinpath(folder).is_dir()
        ]

        # Get channel stack iterators and delayed arrays
        read_delayed_channel_stacks = self._get_delayed_channel_stack(
            channel_paths=channel_paths,
            output_dir=output_compressed_data,
        )

        # Getting compressors
        compressor = self._get_compressor()

        # Writing compressed stacks
        self._compress_and_write_channels(
            read_channel_stacks=read_delayed_channel_stacks,
            compressor=compressor,
            job_kwargs=self.job_settings.compress_job_save_kwargs,
        )
        logging.info("Finished compressing source data.")

    def run_job(self) -> JobResponse:
        """
        Main public method to run the compression job
        Returns
        -------
        JobResponse
          Information about the job that can be used for metadata downstream.

        """
        job_start_time = datetime.now()
        self._compress_raw_data()
        job_end_time = datetime.now()
        return JobResponse(
            status_code=200,
            message=f"Job finished in: {job_end_time-job_start_time}",
            data=None,
        )


class SmartspimJobSettings(BasicJobSettings):
    """SmartspimCompressionJob settings."""

    input_source: str = Field(
        ...,
        description=(
            "Source of the SmartSPIM channel data. For example, "
            "/scratch/SmartSPIM_695464_2023-10-18_20-30-30/SmartSPIM"
        ),
    )
    staging_directory: str = Field(
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
    def partition_list(lst: List, size: int):
        """Partitions a list"""
        for i in range(0, len(lst), size):
            yield lst[i : i + size]

    def _get_partitioned_list_of_stack_paths(self) -> List[Path]:
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
        partition_size = int(
            len(all_stack_paths) / self.job_settings.num_of_partitions
        )

        return list(self.partition_list(all_stack_paths, partition_size))

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
            output_path = self.job_settings.output_directory
            channel_name = stack.parent.parent.name
            stack_name = stack.name

            if self.job_settings.s3_location is not None:
                output_path = self.job_settings.s3_location

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


def get_col_row_len(input_source):
    # Remove this function later?
    from pathlib import Path

    input_source = Path(input_source)

    # Get any channel
    first_channel = list(input_source.glob("Ex_*_Em_*"))[0]

    # Get cols
    cols = [folder for folder in first_channel.glob("*/") if folder.is_dir()]
    n_cols = len(cols)

    if not n_cols:
        raise ValueError("No columns found!")

    rows = [folder for folder in cols[0].glob("*/") if folder.is_dir()]
    print("Rows: ", rows)
    n_rows = len(rows)

    if not n_rows:
        raise ValueError("No rows found")

    return n_cols, n_rows


def main2():
    input_source = "/allen/aind/scratch/svc_aind_upload/test_data_sets/smartspim/SmartSPIM_738819_2024-06-21_13-48-58"
    staging_directory = "/allen/aind/scratch/svc_aind_upload/test_data_sets/smartspim/test_transform_outputs"
    s3_location = "s3://aind-msma-morphology-data/test_data/SmartSPIM/test_data_transform/"
    n_cols, n_rows = get_col_row_len(
        input_source=Path(input_source).joinpath("SmartSPIM")
    )
    num_of_partitions = n_cols * n_rows

    print("Number of partitions: ", num_of_partitions)

    job_settings = SmartspimJobSettings(
        input_source=input_source,
        staging_directory=staging_directory,
        s3_location=s3_location,
        num_of_partitions=num_of_partitions,
        partition_to_process=0,
        output_directory="/allen/aind/scratch/svc_aind_upload/test_data_sets/smartspim/test_transform_outputs/",
    )

    job = SmartspimCompressionJob(job_settings=job_settings)
    job_response = job.run_job()
    # logging.info(job_response.model_dump_json())


if __name__ == "__main__":
    main2()
