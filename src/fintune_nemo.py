import os

from omegaconf import OmegaConf
import torch
import pytorch_lightning as pl
from config import *
from utils.data_utils import preprocess_hackathon_files
from utils.file_utils import group_file_by_speaker
from utils.speaker_tasks import filelist_to_manifest
import wget
from nemo.utils.exp_manager import exp_manager

import nemo.collections.asr as nemo_asr


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

    manifset_filename = filelist_to_manifest(dest_folder, 'manifest', -2, 'out')


    # download model config
    config_dir = DATA_DIR / DATASET_MODE / 'conf'
    model_config = os.path.join(config_dir, f"{NEMO_MODEL_NAME}.yaml")
    url = f"https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/recognition/conf/{NEMO_MODEL_NAME}.yaml"
    wget.download(url, out=model_config)

    finetune_config = OmegaConf.load(model_config)
    finetune_config.model.train_ds.manifest_filepath = '/home/eyalshw/github/wzudemy/hakol/data/subset/nemo/train_manifest.json'
    finetune_config.model.train_ds.augmentor = None
    finetune_config.model.train_ds.batch_size = 2
    finetune_config.model.validation_ds.manifest_filepath = '/home/eyalshw/github/wzudemy/hakol/data/subset/nemo/valid_manifest.json'
    finetune_config.model.validation_ds.batch_size = 2
    finetune_config.trainer.strategy = 'auto'

    # TODO: change with real number
    # finetune_config.model.decoder.num_classes = 10

    # TODO: replace with wandb
    finetune_config.exp_manager.create_tensorboard_logger = False


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
        logger=False,  # Provided by exp_manager
        log_every_n_steps=1,  # Interval of logging.
        val_check_interval=1.0,  # Set to 0.25 to check 4 times per epoch, or an int for number of iterations
    ))
    print(OmegaConf.to_yaml(trainer_config))

    trainer_finetune = pl.Trainer(**trainer_config)

    speaker_model = nemo_asr.models.EncDecSpeakerLabelModel(cfg=finetune_config.model, trainer=trainer_finetune)
    speaker_model.maybe_init_from_pretrained_checkpoint(finetune_config)
    trainer_finetune.fit(speaker_model)

    return


    # create config

    os.makedirs(config_dir, exist_ok=True)
    # url = f"https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/recognition/conf/{NEMO_MODEL_NAME}.yaml"
    # wget.download(url, out=os.path.join(config_dir, f"{NEMO_MODEL_NAME}.yaml"))
    # !wget -P conf https://raw.githubusercontent.com/NVIDIA/NeMo/$BRANCH/examples/speaker_tasks/recognition/conf/titanet-finetune.yaml

    wget.download(url, out=model_config)
    finetune_config = OmegaConf.load(model_config)
    print(finetune_config)




    return




    # log_dir_finetune = exp_manager(trainer_finetune)
    # speaker_model = nemo_asr.models.EncDecSpeakerLabelModel(cfg=finetune_config.model, trainer=trainer_finetune)
    # speaker_model.maybe_init_from_pretrained_checkpoint(finetune_config)

    # trainer_finetune.fit(speaker_model)




    # notebook finetuning
    # !wget -P conf https://raw.githubusercontent.com/NVIDIA/NeMo/$BRANCH/examples/speaker_tasks/recognition/conf/titanet-finetune.yaml
    # MODEL_CONFIG = os.path.join(NEMO_ROOT,'conf/titanet-finetune.yaml')
    # finetune_config = OmegaConf.load(MODEL_CONFIG)
    # print(OmegaConf.to_yaml(finetune_config))
    MODEL_CONFIG = '/home/eyalshw/github/wzudemy/hakol/data/subset/conf/titanet-finetune.yaml'
    finetune_config = OmegaConf.load(MODEL_CONFIG)

    # test_manifest = os.path.join(data_dir,'an4/wav/an4test_clstk/test.json')
    # finetune_config.model.train_ds.manifest_filepath = test_manifest
    # finetune_config.model.validation_ds.manifest_filepath = test_manifest
    # finetune_config.model.decoder.num_classes = 10

    test_manifest = train_manifest
    finetune_config.model.train_ds.manifest_filepath = test_manifest
    finetune_config.model.validation_ds.manifest_filepath = test_manifest
    finetune_config.model.decoder.num_classes = 10

    # Setup the new trainer object
    # # Let us modify some trainer configs for this demo
    # # Checks if we have GPU available and uses it
    # accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'

    # trainer_config = OmegaConf.create(dict(
    #     devices=1,
    #     accelerator=accelerator,
    #     max_epochs=5,
    #     max_steps=-1,  # computed at runtime if not set
    #     num_nodes=1,
    #     accumulate_grad_batches=1,
    #     enable_checkpointing=False,  # Provided by exp_manager
    #     logger=False,  # Provided by exp_manager
    #     log_every_n_steps=1,  # Interval of logging.
    #     val_check_interval=1.0,  # Set to 0.25 to check 4 times per epoch, or an int for number of iterations
    # ))
    # print(OmegaConf.to_yaml(trainer_config))

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
        logger=False,  # Provided by exp_manager
        log_every_n_steps=1,  # Interval of logging.
        val_check_interval=1.0,  # Set to 0.25 to check 4 times per epoch, or an int for number of iterations
    ))
    print(OmegaConf.to_yaml(trainer_config))


    # from the start:
    MODEL_CONFIG = '/home/eyalshw/github/wzudemy/hakol/data/subset/conf/titanet-small.yaml'
    config = OmegaConf.load(MODEL_CONFIG)

    trainer_finetune = pl.Trainer(**trainer_config)
    log_dir_finetune = exp_manager(trainer_finetune, config.get("exp_manager", None))
    print(log_dir_finetune)

    test_manifest = train_manifest
    trainer_finetune.model.train_ds.manifest_filepath = test_manifest
    trainer_finetune.model.validation_ds.manifest_filepath = test_manifest
    trainer_finetune.model.decoder.num_classes = 10

    speaker_model = nemo_asr.models.EncDecSpeakerLabelModel(cfg=finetune_config.model, trainer=trainer_finetune)
    speaker_model.maybe_init_from_pretrained_checkpoint(finetune_config)

    ## Fine-tuning for 5 epochs¶
    trainer_finetune.fit(speaker_model)




    # load pretraind model

    # fine tune

    # evalute


if __name__ == "__main__":
     main()