import random

import torch


def apply_symmetry(tensor, symm):
    """
    Apply a symmetry operation to the given tensor.

    Args:
        tensor (torch.Tensor): Tensor to be rotated. (..., W, W)
        symm (int):
            0, 1, 2, 3: Rotation by symm * pi / 2 radians.
            4, 5, 6, 7: Mirror symmetry on top of rotation.
    """
    assert tensor.shape[-1] == tensor.shape[-2]

    if symm == 0:
        return tensor
    if symm == 1:
        return tensor.transpose(-2, -1).flip(-2)
    if symm == 2:
        return tensor.flip(-1).flip(-2)
    if symm == 3:
        return tensor.transpose(-2, -1).flip(-1)
    if symm == 4:
        return tensor.transpose(-2, -1)
    if symm == 5:
        return tensor.flip(-1)
    if symm == 6:
        return tensor.transpose(-2, -1).flip(-1).flip(-2)
    if symm == 7:
        return tensor.flip(-2)


def apply_symmetry_to_dataset(data, symm):
    to_be_rotated = [
        "input_binary",
        "target_policy_player",
        "target_policy_opponent",
        "target_ownership",
        "target_seki",
        "target_futurepos",
        "target_scoring",
    ]
    data_transformed = {}
    for label, tensor in data.items():
        if label in to_be_rotated:
            if "target_policy" in label:
                tensor_without_pass = tensor[:, :-1].reshape((-1, 19, 19))
                tensor_transformed = apply_symmetry(tensor_without_pass, symm)
                tensor_flattened = tensor_transformed.flatten(start_dim=-2)
                data_transformed[label] = torch.cat(
                    (tensor_flattened, tensor[:, -1:]), dim=1
                )
            else:
                data_transformed[label] = apply_symmetry(tensor, symm)
        else:
            data_transformed[label] = data[label]
    return data_transformed


def apply_random_symmetry_to_dataset(data):
    return apply_symmetry_to_dataset(data, random.randrange(0, 8))
