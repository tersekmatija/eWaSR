import sys
sys.path.append("/home/paperspace/WaSR")

import argparse
import os
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from functools import partial

from datasets.mods import MODSDataset
from datasets.transforms import PytorchHubNormalization
from wasr.inference import LitPredictor
import wasr.models as M
from wasr.utils import load_weights

ARCHITECTURE = 'ewasr_resnet18_imu'
# Colors corresponding to each segmentation class
SEGMENTATION_COLORS = np.array([
    [247, 195, 37],
    [41, 167, 224],
    [90, 75, 164]
], np.uint8)

BATCH_SIZE = 4
WORKERS = 1
DATASET_FILE = './data/mods/preprocessed/sequence_mapping_fix.txt'
OUTPUT_DIR = './output/predictions/mods'


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="SLR model MODS Inference")
    parser.add_argument("--architecture", type=str, choices=M.model_list.copy().extend(M1.model_list), default=ARCHITECTURE,
                        help="Which architecture to use.")
    parser.add_argument("--method-name", type=str, required=True, help="Method name used in evaluation.")
    parser.add_argument("--weights-file", type=str, required=True,
                        help="Path to the weights of the model.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Minibatch size (number of samples) used on each device.")
    parser.add_argument("--workers", type=int, default=WORKERS,
                        help="Number of dataloader workers.")
    parser.add_argument("--dataset-file", type=str, default=DATASET_FILE,
                        help="Path to the file containing the MODS dataset mapping.")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                        help="Root directory for output prediction saving. Predictions are saved inside model subdir.")
    parser.add_argument("--fp16", action='store_true',
                        help="Use half precision for inference.")
    parser.add_argument("--gpus", default=-1,
                    help="Number of gpus (or GPU ids) used for training.")
    parser.add_argument("--mixer", type=str, default="CCCCSS", help="Token mixers in feature mixer.")
    parser.add_argument("--project", action='store_true', help="Project encoder features to less channels.")
    parser.add_argument("--enricher", type=str, default="SS", help="Token mixers in long-skip feature enricher.")
 
    return parser.parse_args()

def export_predictions(probs, batch, method_name, output_dir=OUTPUT_DIR):
    features, metadata = batch

    # Class prediction
    out_class = probs.argmax(1).astype(np.uint8)

    for i, pred_mask in enumerate(out_class):
            pred_mask = SEGMENTATION_COLORS[pred_mask]
            mask_img = Image.fromarray(pred_mask)

            seq_dir = output_dir / metadata['seq'][i] / method_name
            if not seq_dir.exists():
                seq_dir.mkdir(parents=True)

            out_file = (seq_dir / metadata['name'][i]).with_suffix('.png')
            mask_img.save(out_file)

def predict_mods(args):
    dataset = MODSDataset(args.dataset_file, normalize_t=PytorchHubNormalization())
    dl = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers)

    feed_dict = True
    model = M.get_model(args.architecture, num_classes=args.num_classes, pretrained=False, mixer=args.mixer, enricher=args.enricher, project=args.project)

    weights = load_weights(args.weights_file)
    model.load_state_dict(weights)

    output_dir = Path(args.output_dir)

    export_fn = partial(export_predictions, method_name=args.method_name, output_dir=output_dir)
    predictor = LitPredictor(model, export_fn, feed_dict=feed_dict)

    precision = 16 if args.fp16 else 32
    trainer = pl.Trainer(gpus=args.gpus,
                         accelerator='cuda',
                         precision=precision,
                         logger=False)

    trainer.predict(predictor, dl)

def main():
    args = get_arguments()
    print(args)

    predict_mods(args)

if __name__ == '__main__':
    main()
