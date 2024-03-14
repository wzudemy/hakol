from glob import glob
import os
from config import DATA_DIR
from utils.file_utils import group_file_by_speaker
from utils.speaker_tasks import filelist_to_manifest


def preprocess_hackathon_files(mode, csv_filelist, wav_source_folder, output_dir):
    csv_file = DATA_DIR / mode / csv_filelist
    src_folder = DATA_DIR / mode / wav_source_folder
    dest_folder = DATA_DIR / mode / output_dir

    group_file_by_speaker(csv_file, src_folder, dest_folder)

    # wav_files_list = glob(os.path.join(dest_folder, "**", "*.wav"), recursive=True)

    