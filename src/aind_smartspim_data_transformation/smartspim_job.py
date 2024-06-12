"""Module to handle ephys data compression"""

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
from aind_smartspim_data_transformation.io import PngReader
from aind_smartspim_data_transformation.models import CompressorName


class SmartspimJobSettings(BasicJobSettings):
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


class SmartspimCompressionJob(GenericEtl[SmartspimJobSettings]):
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
