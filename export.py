import sys
import argparse
import os
import numpy as np
import torch
import onnx
import onnxsim
import blobconverter

from wasr.inference import LitPredictor
import wasr.models as M
from wasr.utils import load_weights


ARCHITECTURE = 'ewasr_resnet18_imu'
OUTPUT_DIR = './output/export'

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="SLR model MODS Inference")
    parser.add_argument("--architecture", type=str, choices=M.model_list, default=ARCHITECTURE,
                        help="Which architecture to use.")
    parser.add_argument("--num_classes", type=int, default=3,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--weights-file", type=str, required=True,
                        help="Path to the weights of the model.")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                        help="Root directory for output prediction saving. Predictions are saved inside model subdir.")
    parser.add_argument("--onnx_only", action='store_true', help="Export only ONNX model.")
    parser.add_argument("--mixer", type=str, default="CCCCSS", help="Token mixers in feature mixer.")
    parser.add_argument("--project", action='store_true', help="Project encoder features to less channels.")
    parser.add_argument("--enricher", type=str, default="SS", help="Token mixers in long-skip feature enricher.")
 
    return parser.parse_args()

def export(args):

    # prepare directory
    os.makedirs(args.output_dir, exist_ok=True)
    onnx_path = os.path.join(args.output_dir, f"{args.architecture}.onnx")

    # load model
    model = M.get_model(args.architecture, num_classes=args.num_classes, pretrained=False, mixer=args.mixer, enricher=args.enricher, project=args.project)
    weights = load_weights(args.weights_file)
    model.load_state_dict(weights)

    # dummy input
    dummy_input = {
        "image" : torch.randn(1, 3, 384, 512),
        "imu_mask" : torch.randn(1,384, 512)
    }
 
    # export onnx
    torch.onnx.export(
        model, 
        {"x" : dummy_input}, 
        f = onnx_path, 
        opset_version=12, 
        #input_names=["image"] if "imu" not in args.architecture else ["image", "imu"],
        output_names=["prediction", "intermediate"]
    )

    # simplify the model
    model_onnx = onnx.load(onnx_path)  # load onnx model
    onnx_model, check = onnxsim.simplify(model_onnx)
    assert check, 'assert check failed'
    onnx.save(onnx_model, onnx_path)

    model_onnx = onnx.load(onnx_path)
    input_names = [input.name for input in model_onnx.graph.input]

    print(f"ONNX stored at: {onnx_path}")

    if args.onnx_only:
        sys.exit(0)

    # export blob
    blob_path = onnx_path.replace(".onnx", ".blob")

    blob_path_temp = blobconverter.from_onnx(
        onnx_path,
        data_type="FP16",
        shaves=6,
        optimizer_params=[
            "--reverse_input_channels",
            f"--mean_values={input_names[0]}[123.675,116.28,103.53]" if "imu" not in args.architecture else f"--mean_values={input_names[0]}[123.675,116.28,103.53],{input_names[1]}[0]",
            f"--scale_values={input_names[0]}[58.395,57.12,57.375]" if "imu" not in args.architecture else f"--scale_values={input_names[0]}[58.395,57.12,57.375],{input_names[1]}[1]",
            #"--output=prediction",
        ],
        version="2022.1",
        use_cache=False
    )

    os.rename(blob_path_temp, blob_path)
    print(f"Blob stored at: {blob_path}")


def main():
    args = get_arguments()
    export(args)

if __name__ == '__main__':
    main()
