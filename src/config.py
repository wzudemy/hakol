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