import torch
import torch.nn as nn

loss_keys = [
    "loss_policy_player",
    "loss_policy_opponent",
    "loss_value",
    "loss_td_value",
    "loss_ownership",
    "loss_scoring",
    "loss_futurepos",
    "loss_seki",
    "loss_scoremean",
    "loss_lead",
    "loss",
]


# We don't have a soft version in PyTorch
def cross_entropy(pred, target):
    logsoftmax = nn.LogSoftmax(dim=1)
    return torch.sum(-target * logsoftmax(pred), dim=1)


def huber_loss(x, y, delta):
    abs_diff = torch.abs(x - y)
    return torch.where(
        abs_diff > delta,
        (0.5 * delta * delta) + delta * (abs_diff - delta),
        0.5 * abs_diff * abs_diff,
    )


def loss_policy_player_samplewise(pred, target, weight, global_weight):
    loss = weight * cross_entropy(pred, target)
    # print(f"loss_policy_player: {loss.shape}")
    return global_weight * loss


def loss_policy_opponent_samplewise(pred, target, weight, global_weight):
    loss = weight * cross_entropy(pred, target)
    # print(f"loss_policy_opponent: {loss.shape}")
    return 0.15 * global_weight * loss


def loss_value_samplewise(pred, target, global_weight):
    loss = cross_entropy(pred, target)
    # print(f"loss_value: {loss.shape}")
    return 1.20 * global_weight * loss


def loss_td_value_samplewise(pred, target, global_weight):
    loss = cross_entropy(pred, target)
    loss -= cross_entropy(torch.log(target + 1.0e-30), target)
    loss = torch.sum(loss, dim=1)
    # print(f"loss_td_value: {loss.shape}")
    return 0.60 * global_weight * loss


def loss_ownership_samplewise(pred, target, weight, mask, global_weight):
    pred = torch.cat((pred, -pred), 1)
    target = torch.stack(((1 + target) / 2, (1 - target) / 2), 1)
    loss = weight * torch.sum(cross_entropy(pred, target) * mask, dim=(1, 2))
    loss /= torch.sum(mask, dim=(1, 2))
    # print(f"loss_ownership: {loss.shape}")
    return 1.0 * global_weight * loss


def loss_scoring_samplewise(pred, target, weight, mask, global_weight):
    loss = torch.sum(torch.square(pred - target) * mask, dim=(2, 1))
    loss *= weight
    loss /= torch.sum(mask, dim=(1, 2))
    loss = 4.0 * (torch.sqrt(loss * 0.5 + 1.0) - 1.0)
    # print(f"loss_scoring: {loss.shape}")
    return 0.6 * global_weight * loss


def loss_futurepos_samplewise(pred, target, weight, mask, global_weight):
    loss_positionwise = torch.square(torch.tanh(pred) - target) * mask
    loss_positionwise[:, 1, :, :] *= 0.25
    loss = torch.sum(loss_positionwise, dim=(1, 2, 3))
    loss /= torch.sqrt(torch.sum(mask, dim=(1, 2, 3)))
    # print(f"loss_futurepos: {loss.shape}")
    return 0.2 * global_weight * loss


def loss_seki_samplewise(
    pred, target, target_ownership, target_weight_ownership, mask, global_weight
):
    owned_target = torch.square(target_ownership)
    unowned_target = 1.0 - owned_target
    # Not going to use this for now...
    # unowned_proportion = torch.sum(unowned_target * mask, dim=(2, 3)) / (
    #     1.0 + torch.sum(mask, dim=(2, 3))
    # )

    loss1_pred = pred[:, 0:3, :, :]
    loss1_target = torch.stack(
        (
            1.0 - torch.square(target),
            torch.where(target > 0, target, torch.zeros_like(target)),
            torch.where(target < 0, -target, torch.zeros_like(target)),
        ),
        dim=1,
    )
    loss1 = torch.sum(cross_entropy(loss1_pred, loss1_target) * mask, dim=(1, 2))
    # print(f"loss_seki1: {loss1.shape}")

    loss2_pred = torch.stack(
        (pred[:, 3, :, :], torch.zeros_like(target_ownership)), dim=1
    )
    loss2_target = torch.stack((unowned_target, owned_target), dim=1)
    loss2 = torch.sum(cross_entropy(loss2_pred, loss2_target) * mask, dim=(1, 2))
    # print(f"loss_seki2: {loss2.shape}")

    loss = loss1 + 0.5 * loss2
    loss = loss / torch.sum(mask, dim=(1, 2))
    loss *= target_weight_ownership
    # print(f"loss_seki: {loss.shape}")
    return global_weight * loss


def loss_scoremean_samplewise(pred, target, target_weight_ownership, global_weight):
    loss = huber_loss(pred, target, 12.0)
    loss *= target_weight_ownership
    # print(f"loss_scoremean: {loss.shape}")
    return 0.0012 * global_weight * loss


# Requires score belief to be implemented and I don't want to do that
# def loss_scorestdev_samplewise():
#     return


def loss_lead_samplewise(pred, target, weight, global_weight):
    loss = huber_loss(pred, target, 8.0)
    loss *= weight
    # print(f"loss_lead: {loss.shape}")
    return 0.016 * global_weight * loss


# Currently weight is 0.0
# def loss_variance_time_samplewise(pred, target):
#     loss = huber_loss(pred, target, 100.0)
#     return 0.0 * global_weight * loss


def loss_dict_batchwise(
    out_policy,
    out_value,
    out_miscvalue,
    out_ownership,
    out_scoring,
    out_futurepos,
    out_seki,
    batch,
):
    pred_policy = out_policy
    pred_value = out_value
    pred_td_value = torch.reshape(out_miscvalue[:, 4:10], (-1, 3, 2))
    pred_ownership = out_ownership
    pred_scoring = torch.squeeze(out_scoring)
    pred_futurepos = out_futurepos
    pred_seki = out_seki
    pred_scoremean = out_miscvalue[:, 0] * 20.0
    # pred_scorestdev = F.softplus(out_miscvalue[:, 1]) * 20.0
    pred_lead = out_miscvalue[:, 2] * 20.0
    # pred_variance_time = F.softplus(out_miscvalue[:, 3]) * 150.0

    loss_policy_player = loss_policy_player_samplewise(
        pred_policy[:, 0, :],
        batch["target_policy_player"],
        batch["target_weight_policy_player"],
        batch["target_weight_used"],
    ).mean()
    loss_policy_opponent = loss_policy_opponent_samplewise(
        pred_policy[:, 1, :],
        batch["target_policy_player"],
        batch["target_weight_policy_player"],
        batch["target_weight_used"],
    ).mean()
    loss_value = loss_value_samplewise(
        pred_value, batch["target_value"], batch["target_weight_used"]
    ).mean()
    loss_td_value = loss_td_value_samplewise(
        pred_td_value, batch["target_td_value"], batch["target_weight_used"]
    ).mean()
    loss_ownership = loss_ownership_samplewise(
        pred_ownership,
        batch["target_ownership"],
        batch["target_weight_ownership"],
        torch.squeeze(batch["mask"]),
        batch["target_weight_used"],
    ).mean()
    loss_scoring = loss_scoring_samplewise(
        pred_scoring,
        batch["target_scoring"],
        batch["target_weight_scoring"],
        torch.squeeze(batch["mask"]),
        batch["target_weight_used"],
    ).mean()
    loss_futurepos = loss_futurepos_samplewise(
        pred_futurepos,
        batch["target_futurepos"],
        batch["target_weight_futurepos"],
        batch["mask"],
        batch["target_weight_used"],
    ).mean()
    loss_seki = loss_seki_samplewise(
        pred_seki,
        batch["target_seki"],
        batch["target_ownership"],
        batch["target_weight_ownership"],
        torch.squeeze(batch["mask"]),
        batch["target_weight_used"],
    ).mean()
    loss_scoremean = loss_scoremean_samplewise(
        pred_scoremean,
        batch["target_scoremean"],
        batch["target_weight_ownership"],
        batch["target_weight_used"],
    ).mean()
    loss_lead = loss_lead_samplewise(
        pred_lead,
        batch["target_lead"],
        batch["target_weight_lead"],
        batch["target_weight_used"],
    ).mean()

    loss_total = (
        loss_policy_player
        + loss_policy_opponent
        + loss_value
        + loss_td_value
        + loss_ownership
        + loss_scoring
        + loss_futurepos
        + loss_seki
        + loss_scoremean
        + loss_lead
    )

    return {
        "loss_policy_player": loss_policy_player,
        "loss_policy_opponent": loss_policy_opponent,
        "loss_value": loss_value,
        "loss_td_value": loss_td_value,
        "loss_ownership": loss_ownership,
        "loss_scoring": loss_scoring,
        "loss_futurepos": loss_futurepos,
        "loss_seki": loss_seki,
        "loss_scoremean": loss_scoremean,
        "loss_lead": loss_lead,
        "loss": loss_total,
    }
