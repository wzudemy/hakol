import pandas as pd
import wandb
import os
import colorlog
import logging

from datetime import datetime

from utils.file_utils import init_logging

from omegaconf import OmegaConf
import torch
import pytorch_lightning as pl
import config as C
from src.utils.nemo_inference import verify_speakers
from utils.data_utils import preprocess_data, generate_results
from utils.speaker_tasks import filelist_to_manifest
import wget
import nemo.collections.asr as nemo_asr

init_logging(C.LOGGING_LEVEL)

logger = logging.getLogger(__name__)


def main():
    logger.info('start')

    logger.info('preprocess_data')
    dest_folder = preprocess_data()

    logger.info('filelist_to_manifest')
    # Convert the dest folder to manifest
    # based on
    # !python {NEMO_ROOT}/scripts/speaker_tasks/filelist_to_manifest.py --filelist {data_dir}/an4/wav/an4test_clstk/test_all.txt --id -2 --out {data_dir}/an4/wav/an4test_clstk/test.json
    manifest_filename, speakers = filelist_to_manifest(dest_folder, 'manifest', -2, 'out',
                                             min_count=C.SPEAKATHON_MIN_SPEAKER_COUNT, max_count=C.SPEAKATHON_MAX_SPEAKER_COUNT)

    logger.info('create_nemo_config')
    decoder_num_classes = len(set(speakers))
    # download model config
    finetune_config = create_nemo_config(manifest_filename, C.TRAIN_BATCH_SIZE, manifest_filename, C.VALID_BATCH_SIZE,
                                         decoder_num_classes)

    # TODO: add wandb
    logger.info('Create trainer config')
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    trainer_config = OmegaConf.create(dict(
        devices=1,
        accelerator=accelerator,
        max_epochs=5,
        max_steps=-1,  # computed at runtime if not set
        num_nodes=1,
        accumulate_grad_batches=1,
        enable_checkpointing=False,  # Provided by exp_manager
        logger=False,  # Provided by exp_manager
        log_every_n_steps=1,  # Interval of logging.
        val_check_interval=1.0,  # Set to 0.25 to check 4 times per epoch, or an int for number of iterations
    ))
    print(OmegaConf.to_yaml(trainer_config))
    trainer_finetune = pl.Trainer(**trainer_config)

    from nemo.utils.exp_manager import exp_manager
    log_dir = exp_manager(trainer_finetune, finetune_config.get("exp_manager", None))
    logger.info(f"exp_manager: logged to {log_dir}")

    logger.info(f'Load Nemo mode: {C.NEMO_MODEL_NAME}')
    speaker_model = nemo_asr.models.EncDecSpeakerLabelModel(cfg=finetune_config.model, trainer=trainer_finetune)
    speaker_model.maybe_init_from_pretrained_checkpoint(finetune_config)

    logger.info(f'trainer_finetune.fit()')
    trainer_finetune.fit(speaker_model)

    pretrained_model_path =C.DATA_DIR / f"{C.NEMO_MODEL_NAME}_ft.nemo"
    logger.info(f'save_to {pretrained_model_path}')
    speaker_model.save_to(pretrained_model_path)

    # generate results
    encoder = nemo_asr.models.EncDecSpeakerLabelModel.restore_from(pretrained_model_path)
    # valid_audio_path = TBD
    # generate_results(encoder, 'speakathon_data_subset/groups_validation.csv', valid_audio_path)

    logger.info('end')


def create_nemo_config(train_manifest, train_natch_size, valid_manifest, valid_batch_size,
                       decoder_num_classes=10, augmentor=None, strategy='auto'):
    logger.info('start')
    config_dir = C.DATA_DIR / C.DATASET_TYPE / 'conf'
    model_config = os.path.join(config_dir, f"{C.NEMO_MODEL_NAME}.yaml")
    url = f"https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/recognition/conf/{C.NEMO_MODEL_NAME}.yaml"
    wget.download(url, out=model_config)
    finetune_config = OmegaConf.load(model_config)
    finetune_config.model.train_ds.manifest_filepath = train_manifest
    finetune_config.model.train_ds.augmentor = augmentor
    finetune_config.model.train_ds.batch_size = train_natch_size
    finetune_config.model.validation_ds.manifest_filepath = valid_manifest
    finetune_config.model.validation_ds.batch_size = valid_batch_size
    finetune_config.trainer.strategy = strategy
    finetune_config.model.decoder.num_classes = decoder_num_classes

    finetune_config.exp_manager.create_tensorboard_logger = False
    finetune_config.exp_manager.create_checkpoint_callback = True
    finetune_config.exp_manager.create_wandb_logger = True
    finetune_config.exp_manager.wandb_logger_kwargs = {
        "name": "NeSpeak_name",
        "project": "NeSpeak_project"
    }
    logger.info('end')
    return finetune_config


if __name__ == "__main__":
    main()
