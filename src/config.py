from pathlib import Path

ROOT = Path.cwd()
DATA_DIR = Path('/workdir/data')
RAW_DATA_DIR = DATA_DIR / 'wav_files'
CLN_DATA_DIR = DATA_DIR / 'wav_files_cln'
BY_SPEAKER_DIR = DATA_DIR / 'by_speaker'
CSV_DATA_FILE = DATA_DIR / "hackathon_train.csv"
Path(CLN_DATA_DIR).mkdir(parents=True, exist_ok=True)
Path(BY_SPEAKER_DIR).mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 16000
    
DATA_DIR = ROOT / 'data'

SPEAKATHON_DATA_SUBSET = ROOT / 'speakathon_data_subset'
SPEAKATHON_MIN_SPEAKER_COUNT = 5
SPEAKATHON_MAX_SPEAKER_COUNT = 10
SPEAKATHON_FILTER_PREDICTED_SAME_GENDER = False
    
# NEMO_MODEL_NAME = 'ecapa_tdnn'
NEMO_MODEL_NAME = 'titanet-small'

# TODO: change this to challenge
DATASET_TYPE = 'subset'  # challenge, cv-subset, cv-full

TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 4

MAX_SPEAKERS_COUNT_FILTER = 20

LOGGING_LEVEL = "DEBUG"

STUB = False

GENDER_CHUNK_SIZE = 4



