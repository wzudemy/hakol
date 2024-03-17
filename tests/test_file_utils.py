import unittest

from src.utils.file_utils import group_file_by_column


class MyTestCase(unittest.TestCase):

    def test_something(self):
        csv_file = '/home/avrash/ai/hakol-1/data/subset/hackathon_train_subset.csv'
        src_folder = '/home/avrash/ai/hakol-1/data/subset/wav_files_subset'
        dest_folder = '/home/avrash/ai/hakol-1/output'
        column = 'noise_type'
        group_file_by_column(csv_file, src_folder, dest_folder, column)




