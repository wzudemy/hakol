import pandas as pd
import shutil
import os
import librosa
from pydub import AudioSegment
from src import config as C
from pathlib import Path
import torch
from tqdm import tqdm

import glob
from collections import Counter

import pandas as pd
# import shutil
from tqdm import tqdm
# import shutil
from datetime import datetime
import logging
# import colorlog

logger = logging.getLogger(__name__)

def init_logging(logging_level=logging.INFO):
    # Set up logging configuration
    log_folder = "logs"  # Choose your desired log folder
    os.makedirs(log_folder, exist_ok=True)  # Create the folder if it doesn't exist

    # Get the current date and time
    now = datetime.now()

    # Format the timestamp as "HH_MM"
    timestamp = now.strftime("%H_%M_%S")

    # Define the log filename
    log_filename = os.path.join(log_folder, f"log_{timestamp}.txt")
    logging_format = "[%(asctime)s]|[%(levelname)-8s]|(%(filename)-20s:%(lineno)-3s)|%(funcName)s|%(message)s"

    # Configure logging
    logging.basicConfig(
        filename=log_filename,  # Specify the log file
        level=logging_level,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format=logging_format,
        datefmt='%H:%M:%S'
    )

    # Create a logger
    logger = colorlog.getLogger()
    # logger.setLevel(logging.DEBUG)

    # Create a console handler with colors
    console = colorlog.StreamHandler()
    console.setFormatter(colorlog.ColoredFormatter(
        f'%(log_color)s{logging_format}',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    ))
    logger.addHandler(console)

    logging.debug("This is a debug message")
    logging.info("This is an info message")
    logging.warning("This is a warning message")
    logging.error("This is an error message")
    logging.critical("This is a critical message")

def get_base_filenames(directory_path):
    # Get list of all files in the directory
    files = os.listdir(directory_path)

    # Extract base name of each file
    base_names = [os.path.splitext(file)[0] for file in files]
    return base_names


# Function to duplicate files
def create_stub_folder(original_folder, output_folder, num_duplicates):
    logger.info('start')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    original_files = os.listdir(original_folder)
    file_count = len(original_files)

    if file_count == 0:
        print("No files found in the original folder.")
        return

    for i in range(num_duplicates):
        for original_file in original_files:
            original_path = os.path.join(original_folder, original_file)
            # output_file = f"{i * file_count + int(original_file.split('.')[0]):06d}.wav"
            output_file = f"{original_file.split('.')[0]}_{i:03d}.wav"
            output_path = os.path.join(output_folder, output_file)
            shutil.copyfile(original_path, output_path)

    logger.info('end')

def check_extension(file_path, extension):
    _, file_extension = os.path.splitext(file_path)
    return file_extension == extension

def write_list_to_file(lst, file_path):
    with open(file_path, "w") as f:
        for item in lst:
            f.write(str(item) + "\n")

def get_files_in_folder(folder_path):
    files = glob.glob(folder_path + '/*')
    return files


def group_file_by_column(csv_file, src_folder, dest_folder, column='speaker', max_per_val=100):
    val_counter = Counter()
    logger.info("start")
    speakers = pd.read_csv(csv_file, chunksize=1000)
    for chunk in tqdm(speakers):
        records = chunk.to_dict('records')
        for record in records:
            column_val = record.get(column)
            val_counter.update([column_val])
            column_val_count = val_counter.get(column_val)
            if column_val_count > max_per_val:
                continue
            speaker_folder = f'{dest_folder}/{column_val}'
            if not os.path.exists(speaker_folder):
                os.makedirs(speaker_folder)
            file_name = record.get("file")
            src_file = f'{src_folder}/{file_name}'
            dest_file = f'{speaker_folder}/{file_name}'

            shutil.copyfile(src_file, dest_file)
    logger.info("end")
    
    # for record in records:

def pre_process_files(src_folder, dest_folder):
    sampling_rate = 16_000
    skipped = 0
    # sampling_rate_ms = sampling_rate / 1000
    file_list = Path(src_folder).glob('*.wav')
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=True)
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
    for file in tqdm(file_list):
        wav = read_audio(file, sampling_rate=sampling_rate)
        speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=sampling_rate)
        try:
            wav = collect_chunks(speech_timestamps, wav)

        except:
            print('vad failed')
        save_audio(Path(dest_folder) / file.name, wav, sampling_rate=sampling_rate)



def main():
    # pre_process_files(src_folder, dest_folder)
    # csv_file = './data/subset/hackathon_train_subset.csv'
    src_folder = C.CLN_DATA_DIR 
    dest_folder = C.BY_SPEAKER_DIR
    csv_file = C.CSV_DATA_FILE
    group_file_by_column(csv_file, src_folder, dest_folder, max_per_val=20)


        
    print('done')




if __name__ == '__main__':
    main()


