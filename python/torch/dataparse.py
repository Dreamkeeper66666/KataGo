import os
from argparse import ArgumentParser

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import webdataset as wds

from torch.utils.data import DataLoader

def parse_data_batchwise(binchwp, ginc, ptncm, gtnc, sdn, vtnchw):
    input_binary_nchw = binchwp
    input_global_nc = ginc
    target_policy_ncmove = ptncm
    target_global_nc = gtnc
    score_distribution_ns = sdn
    target_value_nchw = vtnchw

    # policy targets
    target_policy_player = target_policy_ncmove[:, 0, :]
    target_policy_player /= target_policy_player.sum(1, keepdim=True)
    target_policy_opponent = target_policy_ncmove[:, 1, :]
    target_policy_opponent /= target_policy_opponent.sum(1, keepdim=True)

    # global targets
    target_weight_policy_player = target_global_nc[:, 26]
    target_weight_policy_opponent = target_global_nc[:, 28]

    target_value = target_global_nc[:, 0:3]
    target_scoremean = target_global_nc[:, 3]
    target_td_value = torch.stack(
        (target_global_nc[:, 4:7], target_global_nc[:, 8:11]), 2
    )
    target_lead = target_global_nc[:, 21]
    target_variance_time = target_global_nc[:, 22]
    target_weight_from_data = target_global_nc[:, 25]
    target_weight_ownership = target_global_nc[:, 27]
    target_weight_lead = target_global_nc[:, 29]
    target_weight_futurepos = target_global_nc[:, 33]
    target_weight_scoring = target_global_nc[:, 34]
    include_history = target_global_nc[:, 36:41]
    # selfkomi = target_global_nc[:, 47]

    # score distribution
    score_distribution = score_distribution_ns / 100.0

    # value targets
    target_ownership = target_value_nchw[:, 0, :, :]
    target_seki = target_value_nchw[:, 1, :, :]
    target_futurepos = target_value_nchw[:, 2:4, :, :]
    target_scoring = target_value_nchw[:, 4, :, :] / 120.0

    # input_binary
    input_binary = input_binary_nchw[:, :, :361].clamp(max=1.0)
    input_binary = input_binary.reshape((-1, 22, 19, 19)).float()
    hmat = history_matrix(input_binary.shape[1], include_history, batchwise=True)
    # input_binary: (n_bin, 19, 19)
    # h_mat: (n_bin, n_bin) - (in, out), sum over in
    # Result: (n_bin, 19, 19)
    input_binary = torch.einsum("bijk,bil->bljk", input_binary, hmat)

    mask = input_binary[:, 0:1, :, :]

    # input_global
    input_global = input_global_nc * F.pad(
        include_history, ((0, 19 - include_history.shape[1])), value=1.0
    )

    data_dict = {
        # binary inputs and mask
        "input_binary": input_binary,
        # global inputs
        "input_global": input_global,
        # policy targets
        "target_policy_player": target_policy_player,
        "target_policy_opponent": target_policy_opponent,
        "target_weight_policy_player": target_weight_policy_player,
        "target_weight_policy_opponent": target_weight_policy_opponent,
        # value target
        "target_value": target_value,
        # td_value target
        "target_td_value": target_td_value,
        # score distribution belif
        "score_distribution": score_distribution,
        # ownership target
        "target_ownership": target_ownership,
        "target_weight_ownership": target_weight_ownership,
        # scoring target
        "target_scoring": target_scoring,
        "target_weight_scoring": target_weight_scoring,
        # futurepos target
        "target_futurepos": target_futurepos,
        "target_weight_futurepos": target_weight_futurepos,
        # seki target
        "target_seki": target_seki,
        # scoremean target
        "target_scoremean": target_scoremean,
        # lead target
        "target_lead": target_lead,
        "target_weight_lead": target_weight_lead,
        # variance time
        "target_variance_time": target_variance_time,
        # target weight multiplied to row
        "target_weight_used": target_weight_from_data,
        # misc
        "mask": mask,
        # These are not really used for training
        # "include_history": include_history,
        # "history_matrix": history_matrix
        # "selfkomi": selfkomi
    }

    return data_dict


def history_matrix(num_bin_features, include_history, batchwise=True):
    assert num_bin_features > 16

    # Not sure what to do when number of binary features changes...
    # But it's not my responsibility :)
    h_mat = torch.diag(
        torch.tensor(
            [
                1.0,  # 0
                1.0,  # 1
                1.0,  # 2
                1.0,  # 3
                1.0,  # 4
                1.0,  # 5
                1.0,  # 6
                1.0,  # 7
                1.0,  # 8
                0.0,  # 9   Location of move 1 turn ago
                0.0,  # 10  Location of move 2 turns ago
                0.0,  # 11  Location of move 3 turns ago
                0.0,  # 12  Location of move 4 turns ago
                0.0,  # 13  Location of move 5 turns ago
                1.0,  # 14  Ladder-threatened stone
                0.0,  # 15  Ladder-threatened stone, 1 turn ago
                0.0,  # 16  Ladder-threatened stone, 2 turns ago
                1.0,  # 17
                1.0,  # 18
                1.0,  # 19
                1.0,  # 20
                1.0,  # 21
            ],
            device=include_history.device,
        )
    )
    # Comments copy-pasted from the original code:
    #
    # Because we have ladder features that express past states rather than past diffs,
    # the most natural encoding when we have no history is that they were always the
    # same, rather than that they were all zero. So rather than zeroingthem we have no
    # history, we add entries in the matrix to copy them over.
    # By default, without history, the ladder features 15 and 16 just copy over from 14.
    h_mat[14, 15] = 1.0
    h_mat[14, 16] = 1.0

    h0 = torch.zeros(num_bin_features, num_bin_features, device=include_history.device)
    # When have the prev move, we enable feature 9 and 15
    h0[9, 9] = 1.0  # Enable 9 -> 9
    h0[14, 15] = -1.0  # Stop copying 14 -> 15
    h0[14, 16] = -1.0  # Stop copying 14 -> 16
    h0[15, 15] = 1.0  # Enable 15 -> 15
    h0[15, 16] = 1.0  # Start copying 15 -> 16

    h1 = torch.zeros(num_bin_features, num_bin_features, device=include_history.device)
    # When have the prevprev move, we enable feature 10 and 16
    h1[10, 10] = 1.0  # Enable 10 -> 10
    h1[15, 16] = -1.0  # Stop copying 15 -> 16
    h1[16, 16] = 1.0  # Enable 16 -> 16

    h2 = torch.zeros(num_bin_features, num_bin_features, device=include_history.device)
    h2[11, 11] = 1.0

    h3 = torch.zeros(num_bin_features, num_bin_features, device=include_history.device)
    h3[12, 12] = 1.0

    h4 = torch.zeros(num_bin_features, num_bin_features, device=include_history.device)
    h4[13, 13] = 1.0

    if batchwise:
        h_mat = h_mat.reshape((1, num_bin_features, num_bin_features))
    else:
        h_mat = h_mat.reshape((num_bin_features, num_bin_features))
    # (5, n_bin, n_bin)
    h_builder = torch.stack((h0, h1, h2, h3, h4))

    # include_history: (N, 5)
    # bi * ijk -> bjk, (N, 5) * (5, n_bin, n_bin) -> (N, n_bin, n_bin)
    if batchwise:
        h_mat = h_mat + torch.einsum("bi,ijk->bjk", include_history, h_builder)
    else:
        h_mat = h_mat + torch.einsum("i,ijk->jk", include_history, h_builder)

    return h_mat


class KataDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.data_dir = args.data_dir
        self.validation_ratio = args.validation_ratio
        self.test_data_dir = args.test_data_dir
        self.batch_size = args.batch_size
        self.samples_per_epoch = args.samples_per_epoch
        self.num_workers = args.dataset_num_workers

    @staticmethod
    def add_datamodule_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--data-dir",
            type=str,
            default="data/data",
            help="Location of training and validation data",
        )
        parser.add_argument(
            "--test-data-dir",
            type=str,
            default="data/test",
            help="Location of test data",
        )
        parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
        parser.add_argument(
            "--samples-per-epoch",
            type=int,
            default=1000000,
            help="Number of samples per epoch",
        )
        parser.add_argument(
            "--dataset-num-workers",
            type=int,
            default=4,
            help="Number of workers per node assigned to test loader",
        )
        parser.add_argument(
            "--validation-ratio",
            type=float,
            default=0,
            help="Validation ratio",
        )
        return parser

    def setup(self, stage: None):
        if stage == "fit" or stage is None:
            data_files = [
                os.path.join(self.data_dir, f)
                for f in os.listdir(self.data_dir)
                if f.endswith(".tar")
            ]
            train_set = (
                wds.Dataset(data_files)
                .shuffle(3072)
                .decode()
                .to_tuple("ten")
                .map(lambda x: x[0])
                .map(
                    lambda x: [
                        np.array(np.reshape(np.unpackbits(x[0]), (22, 368))),
                        np.array(x[1]),
                        np.array(x[2]).astype(np.float32),
                        np.array(x[3]),
                        np.array(x[4]).astype(np.float32),
                        np.array(x[5]).astype(np.float32),
                    ]
                )
                .map(lambda x: [torch.from_numpy(elem) for elem in x])
            )
            num_batches_per_epoch = int(round(self.samples_per_epoch / self.batch_size))
            print(num_batches_per_epoch)
            self.train_set = wds.ResizedDataset(train_set, length=num_batches_per_epoch)

        if stage == "test" or stage is None:
            data_files = [
                os.path.join(self.test_data_dir, f)
                for f in os.listdir(self.test_data_dir)
                if f.endswith(".tar")
            ]
            self.test_set = (
                wds.Dataset(data_files)
                .decode()
                .to_tuple("ten")
                .map(lambda x: x[0])
                .map(
                    lambda x: [
                        np.array(np.reshape(np.unpackbits(x[0]), (22, 368))),
                        np.array(x[1]),
                        np.array(x[2]).astype(np.float32),
                        np.array(x[3]),
                        np.array(x[4]).astype(np.float32),
                        np.array(x[5]).astype(np.float32),
                    ]
                )
                .map(lambda x: [torch.from_numpy(elem) for elem in x])
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )
