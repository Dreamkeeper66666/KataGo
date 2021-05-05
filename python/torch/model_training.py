from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch_optimizer as optim

from dataparse import parse_data_batchwise
from datatransform import apply_random_symmetry_to_dataset
from metric import loss_dict_batchwise, loss_keys
from modelbasis import *
from operations import NormMask, act


class KataGoTrainingModelv8(pl.LightningModule):
    def __init__(self, conf, args):
        super().__init__()

        self.conf = conf
        self.support_jp_rules = conf["support_japanese_rules"]
        self.scoremean_as_lead = conf["use_scoremean_as_lead"]
        self.use_fixup = conf["use_fixup"]
        self.block_kind = conf["block_kind"]
        self.C = conf["trunk_num_channels"]
        self.C_mid = conf["mid_num_channels"]
        self.C_regular = conf["regular_num_channels"]
        self.C_dilated = conf["dilated_num_channels"]
        self.C_gpool = conf["gpool_num_channels"]
        self.C_p = conf["p1_num_channels"]
        self.C_pg = conf["g1_num_channels"]
        self.C_v1 = conf["v1_num_channels"]
        self.C_v2 = conf["v2_size"]
        self.activation = "ReLU"
        if self.use_fixup:
            self.normalization = "FixUp"
        else:
            self.normalization = "BN"
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay

        self.linear_ginput = nn.Linear(19, self.C, bias=False)
        self.conv1 = nn.Conv2d(22, self.C, 5, 1, 2, bias=False)

        self.blocks = nn.ModuleList()
        for block_conf in self.block_kind:
            if block_conf[1] == "regular":
                self.blocks += [
                    ResBlock(self.C, self.activation, self.normalization, masking=True)
                ]
            elif block_conf[1] == "gpool":
                self.blocks += [
                    GpoolResBlock(
                        self.C,
                        self.C_gpool,
                        self.C_regular,
                        self.activation,
                        self.normalization,
                        masking=True,
                    )
                ]
            else:
                assert False

        self.norm1 = NormMask(self.C, self.normalization, masking=True)
        self.act1 = act(self.activation)
        self.policy_head = TrainingPolicyHead(
            self.C,
            self.C_p,
            self.C_pg,
            self.activation,
            self.normalization,
            masking=True,
        )
        self.value_head = TrainingValueHead(
            self.C,
            self.C_v1,
            self.C_v2,
            self.activation,
            self.normalization,
            self.support_jp_rules,
            masking=True,
        )

    def forward(self, input_binary, input_global):
        mask = input_binary[:, 0:1, :, :]

        x_bin = self.conv1(input_binary)
        x_global = self.linear_ginput(input_global).unsqueeze(-1).unsqueeze(-1)
        out = x_bin + x_global

        for block in self.blocks:
            out = block(out, mask)

        out = self.norm1(out, mask)
        out = self.act1(out)

        out_policy = self.policy_head(out, mask)
        (
            out_value,
            out_miscvalue,
            out_ownership,
            out_scoring,
            out_futurepos,
            out_seki,
        ) = self.value_head(out, mask)

        return (
            out_policy,
            out_value,
            out_miscvalue,
            out_ownership,
            out_scoring,
            out_futurepos,
            out_seki,
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--learning-rate", type=float, default=1e-2, help="Learning rate"
        )
        parser.add_argument(
            "--weight-decay", type=float, default=3e-5, help="Weight decay"
        )
        return parser

    def configure_optimizers(self):
        base_optimizer = optim.AdaBelief(
            self.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=self.weight_decay,
        )

        optimizer = optim.Lookahead(base_optimizer, k=5, alpha=0.5)

        return optimizer

    def training_step(self, batch, batch_idx):
        batch = parse_data_batchwise(*batch)
        batch = apply_random_symmetry_to_dataset(batch)

        loss_dict = loss_dict_batchwise(
            *self(batch["input_binary"], batch["input_global"]), batch
        )
        logs = {
            ("train/" + key.replace("loss_", "loss/")): value
            for key, value in loss_dict.items()
        }

        return {"loss": loss_dict["loss"], "log": logs}

    def test_step(self, batch, batch_idx):
        batch = parse_data_batchwise(*batch)

        loss_dict = loss_dict_batchwise(
            *self(batch["input_binary"], batch["input_global"]), batch
        )
        logs = {
            ("test/" + key.replace("loss_", "loss/")): value
            for key, value in loss_dict.items()
        }

        return {"test_loss": loss_dict["loss"], "log": logs}
