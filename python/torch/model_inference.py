import torch
import torch.nn as nn

from modelbasis import *
from operations import NormMask, act


class KataGoInferenceModelv8(nn.Module):
    def __init__(self, conf):
        super(KataGoInferenceModelv8, self).__init__()

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

        self.linear_ginput = nn.Linear(19, self.C, bias=False)
        self.conv1 = nn.Conv2d(22, self.C, 5, 1, 2, bias=False)

        self.blocks = nn.ModuleList()
        for block_conf in self.block_kind:
            if block_conf[1] == "regular":
                self.blocks += [
                    ResBlock(self.C, self.activation, self.normalization, masking=False)
                ]
            elif block_conf[1] == "gpool":
                self.blocks += [
                    GpoolResBlock(
                        self.C,
                        self.C_gpool,
                        self.C_regular,
                        self.activation,
                        self.normalization,
                        masking=False,
                    )
                ]
            else:
                assert False

        self.norm1 = NormMask(self.C, self.normalization, masking=False)
        self.act1 = act(self.activation)
        self.policy_head = InferencePolicyHead(
            self.C,
            self.C_p,
            self.C_pg,
            self.activation,
            self.normalization,
            masking=False,
        )
        self.value_head = InferenceValueHead(
            self.C,
            self.C_v1,
            self.C_v2,
            self.activation,
            self.normalization,
            self.support_jp_rules,
            masking=False,
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
        (out_value, out_miscvalue, out_ownership) = self.value_head(out, mask)

        return (
            out_policy,
            out_value,
            out_miscvalue,
            out_ownership,
        )