"""Tests dask utils"""

import unittest
from unittest.mock import MagicMock, patch

from distributed import Client

from aind_smartspim_data_transformation.compress import dask_utils


class DaskUtilsTest(unittest.TestCase):
    """Class for testing the zarr writer"""

    def test_get_local_deployment(self):
        """Tests getting a deployment"""
        deployment = dask_utils.get_deployment()

        self.assertEqual(dask_utils.Deployment.LOCAL.value, deployment)

    @patch.dict("os.environ", {"SLURM_JOBID": "000"})
    def test_get_allen_deploymet(self):
        """Tests getting a deployment on the Allen HPC"""
        deployment = dask_utils.get_deployment()

        self.assertEqual(dask_utils.Deployment.SLURM.value, deployment)

    @patch("aind_smartspim_data_transformation.compress.dask_utils.get_client")
    def test_get_local_client(self, mock_client: MagicMock):
        """Tests getting a local client"""
        mock_client.return_value = (Client, 0)

        deployment = dask_utils.get_deployment()
        client, _ = dask_utils.get_client(
            deployment=deployment,
            worker_options=0,
            n_workers=1,
            processes=True,
        )

        self.assertEqual(client, Client)

    def test_get_client_fail(self):
        """Tests fail getting a local client"""

        with self.assertRaises(NotImplementedError):
            dask_utils.get_client(
                deployment="UnknownDeployment",
                worker_options=0,
                n_workers=1,
                processes=True,
            )

    @patch.dict("os.environ", {"SLURM_JOBID": "000"})
    @patch("distributed.Client")
    def test_get_slurm_client_mpi_failure(self, mock_client: MagicMock):
        """Tests getting a slurm client"""
        mock_client.return_value = (Client, 0)

        deployment = dask_utils.get_deployment()

        with self.assertRaises(ImportError):
            dask_utils.get_client(
                deployment=deployment,
                worker_options=0,
                n_workers=1,
                processes=True,
            )

    @patch.dict("os.environ", {"SLURM_JOBID": "000"})
    @patch(
        "aind_smartspim_data_transformation.compress.dask_utils.DASK_MPI_INSTALLED",
        new=True,
    )
    @patch.dict("sys.modules", {"dask_mpi": None})
    @patch("aind_smartspim_data_transformation.compress.dask_utils.get_client")
    @patch("distributed.Client")
    def test_get_slurm_client_mpi(
        self, mock_client: MagicMock, mock_get_client: MagicMock
    ):
        """Tests getting a slurm client"""
        mock_client.return_value = (Client, 0)
        mock_get_client.return_value = Client

        deployment = dask_utils.get_deployment()
        slurm_client = dask_utils.get_client(
            deployment=deployment,
            worker_options=None,
            n_workers=1,
            processes=True,
        )

        self.assertEqual(slurm_client, Client)
        mock_get_client.assert_called_once_with(
            deployment=deployment,
            worker_options=None,
            n_workers=1,
            processes=True,
        )

    @patch("requests.delete")
    def test_cancel_slurm_job_success(self, mock_requests_delete: MagicMock):
        """
        Tests cancelling a slurm job successfully
        """
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_requests_delete.return_value = mock_response

        job_id = "123"
        api_url = "http://myhost:80/api/slurm/v0.0.36"
        headers = {"Authorization": "Bearer token"}

        response = dask_utils.cancel_slurm_job(job_id, api_url, headers)

        self.assertEqual(response.status_code, mock_response.status_code)
        mock_requests_delete.assert_called_once_with(
            f"{api_url}/job/{job_id}", headers=headers
        )

    @patch("requests.delete")
    def test_cancel_slurm_job_failure(self, mock_requests_delete: MagicMock):
        """
        Tests cancelling slurm job with
        mock job failure
        """
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_requests_delete.return_value = mock_response

        job_id = "123"
        api_url = "http://myhost:80/api/slurm/v0.0.36"
        headers = {"Authorization": "Bearer token"}

        response = dask_utils.cancel_slurm_job(job_id, api_url, headers)
        self.assertEqual(response.status_code, mock_response.status_code)

        mock_requests_delete.assert_called_once_with(
            f"{api_url}/job/{job_id}", headers=headers
        )

    @patch.dict(
        "os.environ",
        {
            "SLURM_JOBID": "000",
            "HPC_HOST": "example.com",
            "HPC_PORT": "80",
            "HPC_API_ENDPOINT": "api",
            "HPC_USERNAME": "username",
            "HPC_PASSWORD": "password",
            "HPC_TOKEN": "token",
        },
    )
    @patch("os.getenv")
    @patch(
        "aind_smartspim_data_transformation.compress.dask_utils.cancel_slurm_job"
    )
    def test_cleanup_slurm_with_env_vars(
        self, mock_cancel_slurm_job: MagicMock, mock_getenv: MagicMock
    ):
        """
        Cleaning up slurm job with
        environment variables
        """
        mock_getenv.side_effect = lambda x: {
            "SLURM_JOBID": "123",
            "HPC_HOST": "example.com",
            "HPC_PORT": "80",
            "HPC_API_ENDPOINT": "api",
            "HPC_USERNAME": "username",
            "HPC_PASSWORD": "password",
            "HPC_TOKEN": "token",
        }.get(x)

        # Set up mock response for cancel_slurm_job
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_cancel_slurm_job.return_value = mock_response

        dask_utils._cleanup(deployment=dask_utils.Deployment.SLURM.value)

        mock_cancel_slurm_job.assert_called_once_with(
            "123",
            "http://example.com:80/api",
            {
                "X-SLURM-USER-NAME": "username",
                "X-SLURM-USER-PASSWORD": "password",
                "X-SLURM-USER-TOKEN": "token",
            },
        )

    @patch.dict(
        "os.environ",
        {
            "SLURM_JOBID": "000",
            "HPC_HOST": "example.com",
            "HPC_PORT": "80",
            "HPC_API_ENDPOINT": "api",
            "HPC_USERNAME": "username",
            "HPC_PASSWORD": "password",
            "HPC_TOKEN": "token",
        },
    )
    @patch("os.getenv")
    @patch(
        "aind_smartspim_data_transformation.compress.dask_utils.cancel_slurm_job"
    )
    @patch("aind_smartspim_data_transformation.compress.dask_utils.logging")
    def test_cleanup_slurm_with_env_vars_failed(
        self,
        mock_logging: MagicMock,
        mock_cancel_slurm_job: MagicMock,
        mock_getenv: MagicMock,
    ):
        """
        Tests failure cleaning up slurm job with
        environment variables
        """
        mock_getenv.side_effect = lambda x: {
            "SLURM_JOBID": "123",
            "HPC_HOST": "example.com",
            "HPC_PORT": "80",
            "HPC_API_ENDPOINT": "api",
            "HPC_USERNAME": "username",
            "HPC_PASSWORD": "password",
            "HPC_TOKEN": "token",
        }.get(x)

        # Set up mock response for cancel_slurm_job
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.text = "test"
        mock_cancel_slurm_job.return_value = mock_response

        dask_utils._cleanup(deployment=dask_utils.Deployment.SLURM.value)

        mock_cancel_slurm_job.assert_called_once_with(
            "123",
            "http://example.com:80/api",
            {
                "X-SLURM-USER-NAME": "username",
                "X-SLURM-USER-PASSWORD": "password",
                "X-SLURM-USER-TOKEN": "token",
            },
        )
        mock_logging.error.assert_called_once_with(
            "Failed to cancel SLURM job 123: test"
        )

    @patch.dict("os.environ", {"SLURM_JOBID": "000"})
    @patch("aind_smartspim_data_transformation.compress.dask_utils.logging")
    @patch("os.getenv")
    def test_cleanup_slurm_without_env_vars(
        self, mock_getenv: MagicMock, mock_logging: MagicMock
    ):
        """
        Tests cleaning up slurm without
        environment variables
        """
        mock_getenv.side_effect = lambda x: {
            "SLURM_JOBID": "123",
            "HPC_HOST": "example.com",
            "HPC_PORT": "80",
            "HPC_API_ENDPOINT": "api",
            "HPC_USERNAME": "username",
            "HPC_PASSWORD": "password",
            "HPC_TOKEN": "token",
        }.get(x)

        dask_utils._cleanup(deployment=dask_utils.Deployment.SLURM.value)
        mock_logging.error.assert_called_once_with(
            "Failed to get SLURM env vars to cleanup: 'HPC_HOST'"
        )

    @patch("os.getenv")
    @patch("aind_smartspim_data_transformation.compress.dask_utils.logging")
    def test_cleanup_local(
        self, mock_logging: MagicMock, mock_getenv: MagicMock
    ):
        """
        Tests cleaning up a local cluster
        """
        mock_getenv.return_value = None

        dask_utils._cleanup(deployment=dask_utils.Deployment.LOCAL.value)

        mock_logging.info.assert_not_called()

    @patch("os.getenv")
    @patch(
        "aind_smartspim_data_transformation.compress.dask_utils.LOGGER.info"
    )
    @patch("distributed.Client")
    def test_log_dashboard_address(
        self,
        mock_Client: MagicMock,
        mock_logger_info: MagicMock,
        mock_getenv: MagicMock,
    ):
        """
        Tests log dashboard address
        """
        mock_getenv.return_value = "testuser"

        mock_client = MagicMock()
        mock_Client.return_value = mock_client

        mock_client.scheduler_info.return_value = {
            "services": {"dashboard": 8787}
        }

        mock_client.run_on_scheduler.return_value = "scheduler-host"

        dask_utils.log_dashboard_address(client=mock_client)

        mock_logger_info.assert_called_once_with(
            "To access the dashboard, run the following in "
            "a terminal: ssh -L 8787:scheduler-host:8787 testuser@hpc-login "
        )


if __name__ == "__main__":
    unittest.main()
