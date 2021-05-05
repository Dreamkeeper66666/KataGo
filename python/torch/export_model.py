import datetime
import json
import logging
import struct
from argparse import ArgumentParser
from pathlib import Path
import pathlib

import onnxmltools
import torch
from onnxmltools.utils.float16_converter import convert_float_to_float16

from model_inference import KataGoInferenceModelv8
from model_training import KataGoTrainingModelv8


def fill_weights(model, state_dict):
    for key in model.state_dict():
        if key == "policy_head.linear_pass.weight":
            model.state_dict()[key].data = state_dict[key].data[0:1]
        elif key == "policy_head.conv3_1x1.op.conv.weight":
            model.state_dict()[key].data = state_dict[key].data[0:1]
        elif key == "value_head.linear_miscvaluehead.weight":
            model.state_dict()[key].data = state_dict[key].data[0:4]
        elif key == "value_head.linear_miscvaluehead.bias":
            model.state_dict()[key].data = state_dict[key].data[0:4]
        else:
            model.state_dict()[key].data = state_dict[key].data


def export_to_onnx(model, output_file, quantize):
    dummy_input_binary = torch.randn(10, 22, 19, 19)
    dummy_input_binary[:, 0, :, :] = 1.0
    dummy_input_global = torch.randn(10, 19)
    input_names = ["input_binary", "input_global"]
    output_names = [
        "output_policy",
        "output_value",
        "output_miscvalue",
        "output_ownership",
    ]

    torch.onnx.export(
        model,
        (dummy_input_binary, dummy_input_global),
        str(output_file),
        export_params=True,
        verbose=True,
        opset_version=10,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={
            "input_binary": {0: "batch_size", 2: "y_size", 3: "x_size"},
            "input_global": {0: "batch_size"},
            "output_policy": {0: "batch_size", 1: "board_area + 1"},
            "output_value": {0: "batch_size"},
            "output_miscvalue": {0: "batch_size"},
            "output_ownership": {0: "batch_size", 2: "y_size", 3: "x_size"},
        },
    )

    if quantize:
        fp32_model = onnxmltools.utils.load_model(output_file)
        fp16_model = convert_float_to_float16(fp32_model)

        onnxmltools.utils.save_model(
            fp16_model, str(output_file.with_stem(f"{output_file.stem}_fp16"))
        )


def write_str(f, string):
    f.write(string.encode(encoding="ascii", errors="backslashreplace"))


def write_line(f, val):
    write_str(f, f"{str(val)}\n")


def write_weights(f, weights):
    # NCHW to NHWC
    if weights.dim() == 2:
        weights = weights.permute(1, 0)
    if weights.dim() == 4:
        weights = weights.permute(2, 3, 1, 0)
    # Little endian
    weights = weights.detach().flatten()
    np_weights = weights.numpy()
    write_str(f, "@BIN@")
    f.write(struct.pack(f"<{len(np_weights)}f", *np_weights))
    write_str(f, "\n")


def write_conv(f, name, conv):
    write_line(f, name)
    write_line(f, conv.kernel_size[0])
    write_line(f, conv.kernel_size[1])
    write_line(f, conv.in_channels)
    write_line(f, conv.out_channels)
    write_line(f, conv.dilation[0])
    write_line(f, conv.dilation[1])

    write_weights(f, conv.weight)


def write_bn(f, name, norm):
    write_line(f, name)
    write_line(f, norm.num_features)
    write_line(f, norm.eps)
    write_line(f, 1 if norm.affine else 0)
    write_line(f, 1 if norm.affine else 0)
    write_weights(f, norm._buffers["running_mean"])
    write_weights(f, norm._buffers["running_var"])
    if norm.affine:
        write_weights(f, norm._parameters["weight"].data)
        write_weights(f, norm._parameters["bias"].data)


def write_fixup(f, name, norm):
    write_line(f, name)
    write_line(f, norm.num_features)
    write_line(f, 1e-20)
    write_line(f, 1 if norm.use_gamma else 0)
    write_line(f, 1)
    write_weights(f, torch.zeros(norm.num_features))
    write_weights(f, torch.ones(norm.num_features))
    if norm.use_gamma:
        write_weights(f, norm._parameters["gamma"].data)
    write_weights(f, norm._parameters["beta"].data)


def write_norm(f, name, norm, is_fixup):
    if is_fixup:
        write_fixup(f, name, norm)
    else:
        write_bn(f, name, norm)


def write_act(f, name):
    write_line(f, name)


def write_matmul(f, name, linear):
    write_line(f, name)
    write_line(f, linear.in_features)
    write_line(f, linear.out_features)
    write_weights(f, linear._parameters["weight"].data)


def write_matbias(f, name, linear):
    assert linear._parameters["bias"] is not None
    write_line(f, name)
    write_line(f, linear.out_features)
    write_weights(f, linear._parameters["bias"].data)


def write_regular_block(f, name, block, use_fixup):
    write_line(f, "ordinary_block")
    write_line(f, name)
    write_norm(f, f"{name}/norm1", block.conv1_3x3.op.norm._norm, use_fixup)
    write_act(f, f"{name}/actv1")
    write_conv(f, f"{name}/w1", block.conv1_3x3.op.conv)
    write_norm(f, f"{name}/norm2", block.conv2_3x3.op.norm._norm, use_fixup)
    write_act(f, f"{name}/actv2")
    write_conv(f, f"{name}/w2", block.conv2_3x3.op.conv)


def write_gpool_block(f, name, block, use_fixup):
    write_line(f, "gpool_block")
    write_line(f, name)
    write_norm(f, f"{name}/norm1", block.pool.norm1._norm, use_fixup)
    write_act(f, f"{name}/actv1")
    write_conv(f, f"{name}/w1a", block.pool.conv1_3x3)
    write_conv(f, f"{name}/w1b", block.pool.conv2_3x3)
    write_norm(f, f"{name}/norm1b", block.pool.norm2._norm, use_fixup)
    write_act(f, f"{name}/actv1b")
    write_matmul(f, f"{name}/w1r", block.pool.linear)
    write_norm(f, f"{name}/norm2", block.conv1_3x3.op.norm._norm, use_fixup)
    write_act(f, f"{name}/actv2")
    write_conv(f, f"{name}/w2", block.conv1_3x3.op.conv)


def write_block(f, name, block, use_fixup, block_type):
    if block_type == "regular":
        write_regular_block(f, name, block, use_fixup)
    elif block_type == "gpool":
        write_gpool_block(f, name, block, use_fixup)
    else:
        assert False


def write_trunk(f, model):
    use_fixup = model.use_fixup

    write_line(f, "trunk")
    write_line(f, len(model.blocks))
    write_line(f, model.C)
    write_line(f, model.C_mid)
    write_line(f, model.C_regular)
    write_line(f, model.C_dilated)
    write_line(f, model.C_gpool)

    write_conv(f, "conv1", model.conv1)
    write_matmul(f, "ginputw", model.linear_ginput)
    for i, block in enumerate(model.blocks):
        name = model.block_kind[i][0]
        block_type = model.block_kind[i][1]
        write_block(f, name, block, use_fixup, block_type)
    write_norm(f, "trunk/norm", model.norm1._norm, use_fixup)
    write_act(f, "trunk/actv")


def write_policy_head(f, model):
    use_fixup = model.use_fixup

    write_line(f, "policyhead")
    write_conv(f, "p1/intermediate_conv/w", model.policy_head.conv1_1x1)
    write_conv(f, "g1/w", model.policy_head.conv2_1x1)
    write_norm(f, "g1/norm", model.policy_head.norm1._norm, use_fixup)
    write_act(f, "g1/actv")
    write_matmul(f, "matmulg2w", model.policy_head.linear)
    write_norm(f, "p1/norm", model.policy_head.conv3_1x1.op.norm._norm, use_fixup)
    write_act(f, "p1/actv")
    write_conv(f, "p2/w", model.policy_head.conv3_1x1.op.conv)
    write_matmul(f, "matmulpass", model.policy_head.linear_pass)


def write_value_head(f, model):
    use_fixup = model.use_fixup

    write_line(f, "valuehead")
    write_conv(f, "v1/w", model.value_head.init_conv)
    write_norm(f, "v1/norm", model.value_head.norm1._norm, use_fixup)
    write_act(f, "v1/actv")
    write_matmul(f, "v2/w", model.value_head.linear_after_pool)
    write_matbias(f, "v2/b", model.value_head.linear_after_pool)
    write_act(f, "v2/actv")
    # For the time being, it will be assumed that japanese rules are supported
    write_matmul(f, "v3/w", model.value_head.linear_valuehead)
    write_matbias(f, "v3/b", model.value_head.linear_valuehead)
    # and scoremean is not used as lead
    write_matmul(f, "sv3/w", model.value_head.linear_miscvaluehead)
    write_matbias(f, "sv3/b", model.value_head.linear_miscvaluehead)
    write_conv(f, "vownership/w", model.value_head.conv_ownership)


def export_to_bin(model, model_name, output_file):
    f = open(output_file, "wb")
    write_line(f, model_name)
    version = model.conf["version"]
    write_line(f, version)
    if version == 8:
        write_line(f, 22)
        write_line(f, 19)
    write_trunk(f, model)
    write_policy_head(f, model)
    write_value_head(f, model)
    f.close()


def main(args):
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(Path(args.export_dir) / "log.txt"),
            logging.StreamHandler(),
        ],
    )

    logging.info(f"model_config_json: {args.model_config}")
    logging.info(f"export_dir: {args.export_dir}")
    logging.info(f"filename_prefix: {args.filename_prefix}")

    print("Building model", flush=True)
    with open(args.model_config) as f:
        model_config = json.loads(f.read())
    training_model = KataGoTrainingModelv8.load_from_checkpoint(str(args.checkpoint),conf=model_config,args=args)
    inference_model = KataGoInferenceModelv8(model_config)
    fill_weights(inference_model, training_model.state_dict())

    logging.info(f"export format: {args.format}")

    if args.format == "onnx":
        output_path = Path(args.export_dir) / f"{args.filename_prefix}.onnx"
        export_to_onnx(inference_model, output_path, args.quantize_onnx)
    elif args.format == "bin":
        output_path = Path(args.export_dir) / f"{args.filename_prefix}.bin"
        export_to_bin(inference_model, args.model_name, output_path)
    else:
        assert False

    logging.info("Exported at: ")
    logging.info(f"{str(datetime.datetime.utcnow())} UTC")


if __name__ == "__main__":
    description = """
    Export neural net weights and graph to file.
    """
    parser = ArgumentParser(description=description)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True, help="Training checkpoint")
    parser.add_argument(
        "--model-config", required=True,type=pathlib.Path, help="Model config .json file location",
    )
    parser.add_argument(
        "--export-dir", required=True,type=pathlib.Path, help="Model file directory to save to"
    )
    parser.add_argument(
        "--model-name", required=True, help="Name to record in the model file"
    )
    parser.add_argument(
        "--filename-prefix",
        required=True,
        help="Filename prefix to save to within directory",
    )
    parser.add_argument(
        "--format", required=True, choices=["bin", "onnx"], help="Network file format"
    )
    parser.add_argument(
        "--quantize_onnx",
        action="store_true",
        help="Export FP16 version of the ONNX model as well",
    )

    parser = KataGoTrainingModelv8.add_model_specific_args(parser)

    args = parser.parse_args()
    main(args)


