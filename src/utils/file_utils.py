import glob

import pandas as pd
import shutil
import os
from tqdm import tqdm

import os
import shutil
from datetime import datetime
import logging
import colorlog

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
        '%(log_color)s%(levelname)s:%(message)s',
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

def group_file_by_speaker(csv_file, src_folder, dest_folder):
    logger.info("start")
    speakers = pd.read_csv(csv_file, chunksize=1000)
    for chunk in tqdm(speakers):
        records = chunk.to_dict('records')
        for record in records:
            speaker_folder = f'{dest_folder}/{record.get("speaker")}'
            if not os.path.exists(speaker_folder):
                os.makedirs(speaker_folder)
            file_name = record.get("file")
            src_file = f'{src_folder}/{file_name}'
            dest_file = f'{speaker_folder}/{file_name}'

            shutil.copyfile(src_file, dest_file)
    logger.info("end")

def download_from_gdrive(url, output):
    gdown.download(url, output, quiet=False)

    # for record in records:

if __name__ == '__main__':
    import gdown
    # Google Drive URL of the file
    url = 'https://drive.google.com/file/d/10mr9tLip_QhV0cEfp2FsdWoAvuPoscc9/view?usp=drive_link'
    YOUR_FILE_ID = '10mr9tLip_QhV0cEfp2FsdWoAvuPoscc9'
    YOUR_FILE_ID = '1EsBdxcycc7LjcIHCb1LzFnJxX5lGgIc4'
    url = f'https://drive.google.com/uc?id={YOUR_FILE_ID}'
    # Path to save the downloaded file
    output = 'Speakathon.ipynb'
    # Download the file
    gdown.download(url, output, quiet=False)

# wav_folder = '/home/avrash/ai/data/wav_files_subset'
# dest_folder = '/home/avrash/ai/data/by_speaker'
# csv_file_1 = '/home/avrash/ai/data/hackathon_train_subset.csv'

# group_file_by_speaker(csv_file_1, wav_folder, dest_folder)

