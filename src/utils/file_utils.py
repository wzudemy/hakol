import pandas as pd
import shutil
import os

def write_list_to_file(lst, file_path):
    with open(file_path, "w") as f:
        for item in lst:
            f.write(str(item) + "\n")

def group_file_by_speaker(csv_file, src_folder, dest_folder):
    speakers = pd.read_csv(csv_file, chunksize=10)
    for chunk in speakers:
        records = chunk.to_dict('records')
        for record in records:
            speaker_folder = f'{dest_folder}/{record.get("speaker")}'
            if not os.path.exists(speaker_folder):
                os.makedirs(speaker_folder)
            file_name = record.get("file")
            src_file = f'{src_folder}/{file_name}'
            dest_file = f'{speaker_folder}/{file_name}'
            shutil.copyfile(src_file, dest_file)

    # for record in records:



# wav_folder = '/home/avrash/ai/data/wav_files_subset'
# dest_folder = '/home/avrash/ai/data/by_speaker'
# csv_file_1 = '/home/avrash/ai/data/hackathon_train_subset.csv'

# group_file_by_speaker(csv_file_1, wav_folder, dest_folder)

