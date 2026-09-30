"""
Utility functions for the radial correction step
"""

import json
import os
from typing import List


def read_json_as_dict(filepath: str) -> dict:
    """
    Reads a json as dictionary.

    Parameters
    ------------------------

    filepath: PathLike
        Path where the json is located.

    Returns
    ------------------------

    dict:
        Dictionary with the data the json has.

    """

    dictionary = {}

    if os.path.exists(filepath):
        with open(filepath) as json_file:
            dictionary = json.load(json_file)

    return dictionary


def _get_voxel_resolution_v1(acquisition_config: dict) -> List[float]:
    """
    Get the voxel resolution from an acquisition.json file.

    Parameters
    ----------
    acquisition_config: Dict
        Dictionary with the acquisition.json data.
    Returns
    -------
    List[float]
        Voxel resolution in the format [z, y, x].
    """

    if not acquisition_config:
        raise ValueError("acquisition.json file is empty or invalid.")

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


def _get_voxel_resolution_v2(acquisition_config: dict) -> List[float]:
    """
    Get the voxel resolution from an acquisition.json in
    aind-data-schema v2 format.

    Parameters
    ----------
    acquisition_config: Dict
        Dictionary with the acquisition.json data.

    Returns
    -------
    List[float]
        Voxel resolution in the format [z, y, x].
    """
    try:
        data_stream = acquisition_config.get("data_streams", [])[0]
        configuration = data_stream.get("configurations", [])[0]
        image = configuration.get("images", [])[0]
        image_to_acquisition_transform = image[
            "image_to_acquisition_transform"
        ]
    except (IndexError, AttributeError, KeyError) as e:
        raise ValueError(
            "acquisition_config structure is invalid or missing "
            "required fields"
        ) from e

    scale_transform = [
        x["scale"]
        for x in image_to_acquisition_transform
        if x["object_type"] == "Scale"
    ][0]

    # v2 acquisitions store the Scale in data-array (Z, Y, X) order to match
    # the coordinate system axes, so it is already in the returned order.
    z = float(scale_transform[0])
    y = float(scale_transform[1])
    x = float(scale_transform[2])

    return [z, y, x]
