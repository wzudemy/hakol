import os

from omegaconf import OmegaConf
import torch
import pytorch_lightning as pl
import config as C
from src.utils.nemo_inference import verify_speakers
from utils.data_utils import preprocess_hackathon_files
from utils.file_utils import group_file_by_speaker
from utils.speaker_tasks import filelist_to_manifest
import wget
from nemo.utils.exp_manager import exp_manager
import wandb

from pytorch_lightning.loggers import WandbLogger

wandb.login()

# wandb_logger = WandbLogger(project="Nemo Project", log_model='all')


import nemo.collections.asr as nemo_asr


def main():
    wandb.init(
        # set the wandb project where this run will be logged
        project="NeSpeak-project",
    )
    print("start finetuning ...")

    # preprocess the files
    if 'cv-' in C.DATASET_TYPE:
        dest_folder = preprocess_cv_files()
    else:
        dest_folder = preprocess_hackathon_files(
            C.DATASET_TYPE,
            'hackathon_train_subset.csv',
            'wav_files_subset',
            'nemo'
        )

    # Convert the dest folder to manifest
    # based on
    # !python {NEMO_ROOT}/scripts/speaker_tasks/filelist_to_manifest.py --filelist {data_dir}/an4/wav/an4test_clstk/test_all.txt --id -2 --out {data_dir}/an4/wav/an4test_clstk/test.json

    manifest_filename = filelist_to_manifest(dest_folder, 'manifest', -2, 'out')

    # TODO: change with real number
    decoder_num_classes = 10

    # download model config
    finetune_config = create_nemo_config(manifest_filename, C.TRAIN_BATCH_SIZE, manifest_filename, C.VALID_BATCH_SIZE,
                                         decoder_num_classes)

    # TODO: replace with wandb
    # finetune_config.exp_manager.create_tensorboard_logger = False
    # finetune_config.exp_manager.create_

    # print(OmegaConf.to_yaml(finetune_config))

    # Setup the new trainer object
    # Let us modify some trainer configs for this demo
    # Checks if we have GPU available and uses it
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'

    trainer_config = OmegaConf.create(dict(
        devices=1,
        accelerator=accelerator,
        max_epochs=5,
        max_steps=-1,  # computed at runtime if not set
        num_nodes=1,
        accumulate_grad_batches=1,
        enable_checkpointing=False,  # Provided by exp_manager
        logger=None,  # Provided by exp_manager
        log_every_n_steps=1,  # Interval of logging.
        val_check_interval=1.0,  # Set to 0.25 to check 4 times per epoch, or an int for number of iterations
    ))
    print(OmegaConf.to_yaml(trainer_config))

    trainer_finetune = pl.Trainer(**trainer_config)

    speaker_model = nemo_asr.models.EncDecSpeakerLabelModel(cfg=finetune_config.model, trainer=trainer_finetune)
    speaker_model.maybe_init_from_pretrained_checkpoint(finetune_config)
    trainer_finetune.fit(speaker_model)

    result = verify_speakers(speaker_model,
                             '/home/eyalshw/github/wzudemy/hakol/data/subset/nemo/928/1884556.wav',
                             [
                                             '/home/eyalshw/github/wzudemy/hakol/data/subset/nemo/928/8173692.wav',
                                             '/home/eyalshw/github/wzudemy/hakol/data/subset/nemo/2533/0731480.wav',
                                             '/home/eyalshw/github/wzudemy/hakol/data/subset/nemo/3191/1298927.wav',
                                             '/home/eyalshw/github/wzudemy/hakol/data/subset/nemo/4177/1461122.wav',
                                        ]
                             )

    print(result)

    saved_model = 'titanet-small-finetune.nemo'

    # wandb.finish()

    return

    # create config

    os.makedirs(config_dir, exist_ok=True)
    # url = f"https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/recognition/conf/{NEMO_MODEL_NAME}.yaml"
    # wget.download(url, out=os.path.join(config_dir, f"{NEMO_MODEL_NAME}.yaml"))
    # !wget -P conf https://raw.githubusercontent.com/NVIDIA/NeMo/$BRANCH/examples/speaker_tasks/recognition/conf/titanet-finetune.yaml

    wget.download(url, out=model_config)
    finetune_config = OmegaConf.load(model_config)
    print(finetune_config)


def create_nemo_config(train_manifest, train_natch_size, valid_manifest, valid_batch_size,
                       decoder_num_classes=10, augmentor=None, strategy='auto'):
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

    finetune_config.exp_manager.create_tensorboard_logger = True
    finetune_config.exp_manager.create_checkpoint_callback = True
    # finetune_config.exp_manager.create_wandb_logger = True
    # finetune_config.exp_manager.wandb_logger_kwargs = {
    #     "name": "NeSpeak_name",
    #     "project": f"{C.DATASET_TYPE}_{C.NEMO_MODEL_NAME}"
    # }
    return finetune_config


if __name__ == "__main__":
    main()
