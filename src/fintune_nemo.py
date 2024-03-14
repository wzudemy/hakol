from config import *
from utils.file_utils import group_file_by_speaker


def main():
    print("start finetuning ...")

    # create mainfest

    # copy the challenge (subset) files to the diffrent speakers folders
    # create a manifset using the script
    mode = "subset" # remember to chnage for 'challenge'
    csv_file = DATA_DIR / mode / 'hackathon_train_subset.csv'
    src_folder = DATA_DIR / mode / 'wav_files_subset'
    dest_folder = DATA_DIR / mode / 'nemo'

    group_file_by_speaker(csv_file, src_folder, dest_folder)


    # [2] copy the cv files to the diffrent speakers folders
    # [2] create a manifset using the script


    # create config

    # update config with manifest

    # load pretraind model

    # fine tune

    # evalute


if __name__ == "__main__":
     main()