import os
from config import *
from utils.data_utils import preprocess_hackathon_files
from utils.file_utils import group_file_by_speaker
from utils.speaker_tasks import filelist_to_manifest
import wget


def main():
    print("start finetuning ...")
    
      
    # preprocess the files
    dest_folder = preprocess_hackathon_files(
         DATASET_MODE,
         'hackathon_train_subset.csv',
         'wav_files_subset',
        'nemo'
    )

    # TODO: write preprocess cv files
    
    # based on
    # !python {NEMO_ROOT}/scripts/speaker_tasks/filelist_to_manifest.py --filelist {data_dir}/an4/wav/an4test_clstk/test_all.txt --id -2 --out {data_dir}/an4/wav/an4test_clstk/test.json


    filelist_to_manifest(dest_folder, 'manifest', -2, 'out')

    train_manifest = os.path.join(DATA_DIR,'an4/wav/an4_clstk/train.json')
    validation_manifest = os.path.join(DATA_DIR,'an4/wav/an4_clstk/dev.json')
    test_manifest = os.path.join(DATA_DIR,'an4/wav/an4_clstk/dev.json')
    
    # create config
    config_dir = DATA_DIR / DATASET_MODE / 'conf'
    os.makedirs(config_dir, exist_ok=True)
    url = f"https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/recognition/conf/{NEMO_MODEL_NAME}.yaml"
    wget.download(url, out=DATA_DIR / DATASET_MODE / 'conf' / f"{NEMO_MODEL_NAME}.yaml")

    # update config with manifest

    # load pretraind model

    # fine tune

    # evalute


if __name__ == "__main__":
     main()