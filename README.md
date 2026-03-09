# aind-smartspim-data-transformation

[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
![Code Style](https://img.shields.io/badge/code%20style-black-black)
[![semantic-release: angular](https://img.shields.io/badge/semantic--release-angular-e10079?logo=semantic-release)](https://github.com/semantic-release/semantic-release)
![Interrogate](https://img.shields.io/badge/interrogate-100.0%25-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-87%25-yellow?logo=codecov)
![Python](https://img.shields.io/badge/python->=3.9.2-blue?logo=python)



## Overview
This package converts SmartSPIM tile stacks (PNG or TIFF) into OME-Zarr
pyramids, with optional upload to S3. It partitions the work across
independent stack folders so you can parallelize by running multiple
partitions at once.

Key capabilities:
- Read PNG or TIFF stacks and write OME-Zarr pyramids.
- Derive voxel size from the input acquisition.json metadata.
- Optional Blosc compression and S3 sync.

## Expected input layout
Input folders are expected to look like this:

```text
<input_source>/
	acquisition.json
	derivatives/
		metadata.json
	SmartSPIM/
		Ex_445_Em_469/
			432380/
				432380_504340/
					*.png | *.tif | *.tiff
				432380_530260/
					*.png | *.tif | *.tiff
		Ex_561_Em_600/
			...
```

Each stack folder (for example `432380_504340`) is processed into an
OME-Zarr named `<stack_name>.ome.zarr` under the output channel folder.

## Installation
To use the software, in the root directory, run:
```bash
pip install -e .
```

To develop the code, run:
```bash
pip install -e .[dev]
```

## Quick start
Run a single partition locally:

```bash
python -m aind_smartspim_data_transformation.smartspim_job \
	--config-file path/to/config.json
```

Example config:

```json
{
	"input_source": "/data/SmartSPIM_000000_2024-06-05_07-56-54",
	"output_directory": "/data/ome_zarr_output",
	"num_of_partitions": 4,
	"partition_to_process": 0,
	"chunk_size": [128, 128, 128],
	"scale_factor": [2, 2, 2],
	"downsample_levels": 4,
	"compressor_name": "blosc",
	"compressor_kwargs": {"cname": "zstd", "clevel": 3, "shuffle": 1},
	"s3_location": null
}
```

If `s3_location` is set (for example `s3://bucket/prefix`), the job will
sync each OME-Zarr to S3 and remove the local copy after upload. The
`derivatives/` folder is uploaded once by partition 0.

## Environment variables
You can also configure the job via env vars. These are the keys used in
the tests and the default `BasicJobSettings` behavior:

```bash
export TRANSFORMATION_JOB_INPUT_SOURCE=/data/SmartSPIM_000000_2024-06-05_07-56-54
export TRANSFORMATION_JOB_OUTPUT_DIRECTORY=/data/ome_zarr_output
export TRANSFORMATION_JOB_NUM_OF_PARTITIONS=4
export TRANSFORMATION_JOB_PARTITION_TO_PROCESS=0
```

Then run:

```bash
python -m aind_smartspim_data_transformation.smartspim_job
```

## Output layout
The output directory is organized by channel name, each containing one
OME-Zarr per stack:

```text
<output_directory>/
	Ex_445_Em_469/
		432380_504340.ome.zarr/
		432380_530260.ome.zarr/
	Ex_561_Em_600/
		432380_504340.ome.zarr/
		432380_530260.ome.zarr/
```

## Requirements and notes
- PNG or TIFF stacks only. Mixed extensions within a stack are not
	supported.
- AWS CLI must be installed and configured if using S3 upload.
- The voxel size is read from `acquisition.json` and assumed to be
	consistent across the dataset.

## Contributing

### Linters and testing

There are several libraries used to run linters, check documentation, and run tests.

- Please test your changes using the **coverage** library, which will run the tests and log a coverage report:

```bash
coverage run -m unittest discover && coverage report
```

- Use **interrogate** to check that modules, methods, etc. have been documented thoroughly:

```bash
interrogate .
```

- Use **flake8** to check that code is up to standards (no unused imports, etc.):
```bash
flake8 .
```

- Use **black** to automatically format the code into PEP standards:
```bash
black .
```

- Use **isort** to automatically sort import statements:
```bash
isort .
```

### Pull requests

For internal members, please create a branch. For external members, please fork the repository and open a pull request from the fork. We'll primarily use [Angular](https://github.com/angular/angular/blob/main/CONTRIBUTING.md#commit) style for commit messages. Roughly, they should follow the pattern:
```text
<type>(<scope>): <short summary>
```

where scope (optional) describes the packages affected by the code changes and type (mandatory) is one of:

- **build**: Changes that affect build tools or external dependencies (example scopes: pyproject.toml, setup.py)
- **ci**: Changes to our CI configuration files and scripts (examples: .github/workflows/ci.yml)
- **docs**: Documentation only changes
- **feat**: A new feature
- **fix**: A bugfix
- **perf**: A code change that improves performance
- **refactor**: A code change that neither fixes a bug nor adds a feature
- **test**: Adding missing tests or correcting existing tests

### Semantic Release

The table below, from [semantic release](https://github.com/semantic-release/semantic-release), shows which commit message gets you which release type when `semantic-release` runs (using the default configuration):

| Commit message                                                                                                                                                                                   | Release type                                                                                                    |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------- |
| `fix(pencil): stop graphite breaking when too much pressure applied`                                                                                                                             | ~~Patch~~ Fix Release, Default release                                                                          |
| `feat(pencil): add 'graphiteWidth' option`                                                                                                                                                       | ~~Minor~~ Feature Release                                                                                       |
| `perf(pencil): remove graphiteWidth option`<br><br>`BREAKING CHANGE: The graphiteWidth option has been removed.`<br>`The default graphite width of 10mm is always used for performance reasons.` | ~~Major~~ Breaking Release <br /> (Note that the `BREAKING CHANGE: ` token must be in the footer of the commit) |

### Documentation
To generate the rst files source files for documentation, run
```bash
sphinx-apidoc -o doc_template/source/ src 
```
Then to create the documentation HTML files, run
```bash
sphinx-build -b html doc_template/source/ doc_template/build/html
```
More info on sphinx installation can be found [here](https://www.sphinx-doc.org/en/master/usage/installation.html).
