from pathlib import Path

ROOT = Path.cwd()
    
DATA_DIR = Path('/workdir') / 'data'

SPEAKATHON_DATA_SUBSET = DATA_DIR
SPEAKATHON_MIN_SPEAKER_COUNT = 5
SPEAKATHON_MAX_SPEAKER_COUNT = 10

    
# NEMO_MODEL_NAME = 'ecapa_tdnn'
NEMO_MODEL_NAME = 'titanet-large'
HP_SEGMENTS = False
HP_MAX_EPOCS = 5
HP_FILTER_PREDICTED_SAME_GENDER = True
HP_TEST_SIZE = 0.1

# TODO: change this to challenge
DATASET_TYPE = 'subset'  # challenge, cv-subset, cv-full

TRAIN_BATCH_SIZE = 36
VALID_BATCH_SIZE = 36

# TODO: increase

MAX_SPEAKERS_COUNT_FILTER = 20

LOGGING_LEVEL = "DEBUG"

STUB = False

GENDER_CHUNK_SIZE = 4



