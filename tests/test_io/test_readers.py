"""Tests methods in io.readers module"""

from aind_smartspim_data_transformation.io.readers import PngReader
import unittest

from pathlib import Path
import os

RESOURCES_DIR = (
    Path(os.path.dirname(os.path.realpath(__file__))) / ".." / "resources"
)

STACK_DIR = (
    RESOURCES_DIR
    / "SmartSPIM_000000_2024-06-05_07-56-54"
    / "SmartSPIM"
    / "Ex_445_Em_469"
    / "432380"
    / "432380_504340"
    / "*.png"
)


class TestPngReader(unittest.TestCase):
    """Tests methods in PngReader class"""

    @classmethod
    def setUpClass(cls):
        """Sets up class with common reader"""
        cls.reader = PngReader(STACK_DIR)

    def test_shape(self):
        """Tests shape method"""
        expected_shape = (2, 1600, 2000)
        self.assertEqual(expected_shape, self.reader.shape)

    def test_chunks(self):
        """Tests chunks method"""
        expected_chunks = (1, 1600, 2000)
        self.assertEqual(expected_chunks, self.reader.chunks)

    def test_data_path(self):
        """Tests data_path method"""
        self.assertEqual(STACK_DIR, self.reader.data_path)

    def test_data_path_setter(self):
        """Tests data_path_setter"""
        reader = PngReader(STACK_DIR)
        reader.data_path = "tests"
        self.assertEqual("tests", reader.data_path)


if __name__ == "__main__":
    unittest.main()
