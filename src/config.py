from pathlib import Path


ROOT = Path.cwd()

DATA_DIR = ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'subset' / 'wav_files_subset'
CLN_DATA_DIR = DATA_DIR / 'subset' / 'wav_files_subset_cln'

Path(CLN_DATA_DIR).mkdir(parents=True, exist_ok=True)