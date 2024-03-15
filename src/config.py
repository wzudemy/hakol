from pathlib import Path

ROOT = Path.cwd()
    
DATA_DIR = ROOT / 'data'

# NEMO_MODEL_NAME = 'ecapa_tdnn'
NEMO_MODEL_NAME = 'titanet-small'

# TODO: change this to challenge
DATASET_TYPE = 'subset'  # challenge, cv-subset, cv-full

TRAIN_BATCH_SIZE = 2

VALID_BATCH_SIZE = 2