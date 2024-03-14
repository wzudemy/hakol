import pandas as pd
import shutil
import os
import librosa
from pydub import AudioSegment
from src import config as C
from pathlib import Path
import torch
from tqdm import tqdm




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
    src_folder = C.RAW_DATA_DIR
    dest_folder = C.CLN_DATA_DIR 
    pre_process_files(src_folder, dest_folder)
    csv_file = './data/subset/hackathon_train_subset.csv'
    src_folder = dest_folder
    dest_folder = './data/subset/by_speaker'
    group_file_by_speaker(csv_file, src_folder, dest_folder)


        
    print('done')




if __name__ == '__main__':

    main()




