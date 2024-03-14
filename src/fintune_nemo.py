from config import *
from utils.data_utils import preprocess_hackathon_files
from utils.file_utils import group_file_by_speaker
from utils.speaker_tasks import filelist_to_manifest


def main():
    print("start finetuning ...")
    
    # preprocess the files
    dest_folder = preprocess_hackathon_files(
         'subset',
         'hackathon_train_subset.csv',
         'wav_files_subset',
        'nemo'
    )

    # TODO: write preprocess cv files

    filelist_to_manifest(dest_folder, 'manifest', -2, 'out')

    # create config

    # update config with manifest

    # load pretraind model

    # fine tune

    # evalute


if __name__ == "__main__":
     main()