from argparse import ArgumentParser

import numpy as np
import json
import torch

from model_training import KataGoTrainingModelv8
from model_inference import KataGoInferenceModelv8
from export_model import fill_weights


def main(args):
    with open(args.model_config) as f:
        model_config = json.load(f)
    inference_model = KataGoInferenceModelv8(model_config)
    training_model = KataGoTrainingModelv8(model_config, args)
    fill_weights(inference_model, training_model.state_dict())

    for param_tensor in inference_model.state_dict():
        print(param_tensor, "\t", inference_model.state_dict()[param_tensor].size())

    for param_tensor in training_model.state_dict():
        print(param_tensor, "\t", training_model.state_dict()[param_tensor].size())

if __name__ == "__main__":
    description = """
    Convert KataGo .bin model to .onnx file.
    """
    parser = ArgumentParser(description)
    parser.add_argument(
        "--model-config",
        type=str,
        required=True,
        help="KataGo model.config.json file location (usually archived in the .zip file)",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=3e-5, help="Weight decay"
    )

    args = parser.parse_args()

    main(args)
