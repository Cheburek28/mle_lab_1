import configparser
import os
import unittest
import pandas as pd
import sys

sys.path.insert(1, os.path.join(os.getcwd(), "src"))

from train import SVMModel

config = configparser.ConfigParser()
config.read("config.ini")


class TestMultiModel(unittest.TestCase):

    def setUp(self) -> None:
        self.multi_model = SVMModel()

    def test_SVM(self):
        self.assertEqual(self.multi_model.svm(), True)


if __name__ == "__main__":
    unittest.main()
