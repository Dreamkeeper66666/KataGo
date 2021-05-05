import json
import time
from argparse import ArgumentParser

import pytorch_lightning as pl

from dataparse import KataDataModule
from model_training import KataGoTrainingModelv8


def main(hparams):
    with open(hparams.model_config) as f:
        model_config = json.load(f)
    assert model_config["version"] == 8
    assert model_config["support_japanese_rules"] == True
    assert model_config["use_fixup"] == True
    assert model_config["use_scoremean_as_lead"] == False

    pl.seed_everything(hparams.seed)
    dm = KataDataModule(hparams)
    model = KataGoTrainingModelv8(model_config, hparams)
    trainer = pl.Trainer(
        benchmark=True,
        default_root_dir=hparams.run_name,
        distributed_backend="ddp",
        gpus=hparams.gpus,
        max_epochs=hparams.max_epochs,
        num_nodes=hparams.num_nodes,
        precision=16,
        resume_from_checkpoint=hparams.resume
        )
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    description = """
    Train a KataGo network.
    """
    parser = ArgumentParser(description)

    parser.add_argument(
        "--run-name",
        type=str,
        default=time.strftime("%Y%m%d-%H%M%S"),
        help="Experiment label",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        required=True,
        help="KataGo model.config.json file",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for randomization",
    )
    parser.add_argument(
        "--resume", type=str, default="None", help="Checkpoint to resume training"
    )

    parser = KataDataModule.add_datamodule_specific_args(parser)
    parser = KataGoTrainingModelv8.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)

    hparams = parser.parse_args()

    if hparams.resume == "None":
        hparams.resume = None

    main(hparams)


