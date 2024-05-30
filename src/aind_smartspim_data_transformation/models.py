"""Helpful models used in the ephys compression job"""

from enum import Enum

from numcodecs import Blosc


class CompressorName(str, Enum):
    """Enum for compression algorithms a user can select"""

    BLOSC = Blosc.codec_id
