import logging
import os
import shutil
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import config as C
from utils.audio_utils import convert_to_nemo_format_using_pydub
from utils.file_utils import group_file_by_speaker, create_stub_folder, get_base_filenames
from glob import glob

logger = logging.getLogger(__name__)

def preprocess_data():

    filelist_csv = C.DATA_DIR / 'subset' / 'hackathon_train_subset.csv'
    filtered_filelist_csv = C.DATA_DIR / 'subset' / 'filtered_filelist.csv'
    src_folder = C.DATA_DIR / 'subset' / 'wav_files_subset'
    dest_folder = C.DATA_DIR / 'subset' / 'nemo'

    if C.STUB:
        generate_stub_dataset(src_folder, 100)

        filelist_csv = C.DATA_DIR / 'stub' / 'stub.csv'
        filtered_filelist_csv = C.DATA_DIR / 'stub' / 'filtered_filelist.csv'
        src_folder = C.DATA_DIR / 'stub' / 'wav_files_subset'
        dest_folder = C.DATA_DIR / 'stub' / 'nemo'

    if os.path.exists(dest_folder):
       shutil.rmtree(dest_folder)

    # filter the data for
    df = pd.read_csv(filelist_csv)
    filtered_df = df.groupby(['speaker']).head(C.MAX_SPEAKERS_COUNT)
    filtered_df.to_csv(filtered_filelist_csv)

    group_file_by_speaker(filtered_filelist_csv, src_folder, dest_folder)

    files_list = glob(os.path.join(dest_folder, "**", "*.wav"), recursive=True)

    logger.info('convert_to_nemo_format_using_pydub start')
    with ProcessPoolExecutor() as pool:
        nemo_wav_files = pool.map(convert_to_nemo_format_using_pydub, files_list)
    logger.info('convert_to_nemo_format_using_pydub end')

    print(len(list(nemo_wav_files)))

    return dest_folder

def generate_stub_dataset(src_folder, num_duplicates):
    logger.info('start')
    stub_folder = C.DATA_DIR / 'stub' / 'wav_files_subset'
    create_stub_folder(src_folder, stub_folder, num_duplicates)
    filenames = os.listdir(stub_folder)

    start_value = 1000
    num_items = 1000

    # Create a list starting from 1000 and ending at 1999
    speakers = list(range(start_value, start_value + num_items))
    # Create a list repeating the given numbers
    speakers_ids = [speakers[i % len(speakers)] for i in range(len(filenames))]
    stub_df = pd.DataFrame()
    stub_df['file'] = filenames
    stub_df['speaker'] = speakers_ids
    stub_df['language'] = 'russian'
    stub_df['noise_type'] = "clean"
    stub_df.to_csv('/home/eyalshw/github/wzudemy/hakol/data/stub/stub.csv', index=False)
    logger.info('end')



    