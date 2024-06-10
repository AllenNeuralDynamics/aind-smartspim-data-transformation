"""
Module for dask utilities
"""

import logging
import os
import socket
from enum import Enum
from typing import Optional, Tuple

import distributed
import requests
from distributed import Client, LocalCluster

try:
    from dask_mpi import initialize

    DASK_MPI_INSTALLED = True
except ImportError:
    DASK_MPI_INSTALLED = False

LOGGER = logging.getLogger(__name__)


class Deployment(Enum):
    """Deployment enums"""

    LOCAL = "local"
    SLURM = "slurm"


def log_dashboard_address(
    client: distributed.Client, login_node_address: str = "hpc-login"
) -> None:
    """
    Logs the terminal command required to access the Dask dashboard

    Args:
        client: the Client instance
        login_node_address: the address of the cluster login node
    """
    host = client.run_on_scheduler(socket.gethostname)  # noqa: F841
    port = client.scheduler_info()["services"]["dashboard"]  # noqa: F841
    user = os.getenv("USER")  # noqa: F841
    LOGGER.info(
        f"To access the dashboard, run the following in "
        f"a terminal: ssh -L {port}:{host}:{port} {user}@"
        f"{login_node_address} "
    )


def get_deployment() -> str:
    """
    Gets the SLURM deployment if this
    exists

    Returns
    -------
    str
        SLURM_JOBID
    """
    if os.getenv("SLURM_JOBID") is None:
        deployment = Deployment.LOCAL.value
    else:
        # we're running on the Allen HPC
        deployment = Deployment.SLURM.value
    return deployment


def get_client(
    deployment: str = Deployment.LOCAL.value,
    worker_options: Optional[dict] = None,
    n_workers: int = 1,
    processes=True,
) -> Tuple[distributed.Client, int]:
    """
    Create a distributed Client

    Args:
        deployment: the type of deployment. Either "local" or "slurm"
        worker_options: a dictionary of options to pass to the worker class
        n_workers: the number of workers (only applies to "local" deployment)

    Returns:
        the distributed Client and number of workers
    """
    if deployment == Deployment.SLURM.value:
        if not DASK_MPI_INSTALLED:
            raise ImportError(
                "dask-mpi must be installed to use the SLURM deployment"
            )
        if worker_options is None:
            worker_options = {}
        slurm_job_id = os.getenv("SLURM_JOBID")
        if slurm_job_id is None:
            raise Exception(
                "SLURM_JOBID environment variable is not set."
                "Are you running under SLURM?"
            )
        initialize(
            nthreads=int(os.getenv("SLURM_CPUS_PER_TASK", 1)),
            local_directory=f"/scratch/fast/{slurm_job_id}",
            worker_class="distributed.nanny.Nanny",
            worker_options=worker_options,
        )
        client = Client()
        log_dashboard_address(client)
        n_workers = int(os.getenv("SLURM_NTASKS"))
    elif deployment == Deployment.LOCAL.value:
        client = Client(
            LocalCluster(
                n_workers=n_workers, processes=processes, threads_per_worker=1
            )
        )
    else:
        raise NotImplementedError
    return client, n_workers


def cancel_slurm_job(
    job_id: str, api_url: str, headers: dict
) -> requests.Response:
    """
    Attempt to release resources and cancel the job

    Args:
        job_id: the SLURM job ID
        api_url: the URL of the SLURM REST API.
        E.g., "http://myhost:80/api/slurm/v0.0.36"

    Raises:
        HTTPError: if the request to cancel the job fails
    """
    # Attempt to release resources and cancel the job
    # Workaround for https://github.com/dask/dask-mpi/issues/87
    endpoint = f"{api_url}/job/{job_id}"
    response = requests.delete(endpoint, headers=headers)

    return response


def _cleanup(deployment: str) -> None:
    """
    Clean up any resources that were created during the job.

    Parameters
    ----------
    deployment : str
      The type of deployment. Either "local" or "slurm"
    """
    if deployment == Deployment.SLURM.value:
        job_id = os.getenv("SLURM_JOBID")
        if job_id is not None:
            try:
                api_url = f"http://{os.environ['HPC_HOST']}"
                api_url += f":{os.environ['HPC_PORT']}"
                api_url += f"/{os.environ['HPC_API_ENDPOINT']}"
                headers = {
                    "X-SLURM-USER-NAME": os.environ["HPC_USERNAME"],
                    "X-SLURM-USER-PASSWORD": os.environ["HPC_PASSWORD"],
                    "X-SLURM-USER-TOKEN": os.environ["HPC_TOKEN"],
                }
            except KeyError as ke:
                logging.error(f"Failed to get SLURM env vars to cleanup: {ke}")
                return
            logging.info(f"Cancelling SLURM job {job_id}")
            response = cancel_slurm_job(job_id, api_url, headers)
            if response.status_code != 200:
                logging.error(
                    f"Failed to cancel SLURM job {job_id}: {response.text}"
                )
            else:
                # This might not run if the job is cancelled
                logging.info(f"Cancelled SLURM job {job_id}")
