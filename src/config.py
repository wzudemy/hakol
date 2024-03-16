from pathlib import Path

ROOT = Path.cwd()
    
DATA_DIR = ROOT / 'data'

SPEAKATHON_DATA_SUBSET = ROOT / 'speakathon_data_subset'

# NEMO_MODEL_NAME = 'ecapa_tdnn'
NEMO_MODEL_NAME = 'titanet-small'

# TODO: change this to challenge
DATASET_TYPE = 'subset'  # challenge, cv-subset, cv-full

TRAIN_BATCH_SIZE = 4

VALID_BATCH_SIZE = 4

MAX_SPEAKERS_COUNT = 1000

LOGGING_LEVEL = "INFO"

STUB = False
