import sys
import os

sys.path.append(os.getcwd())

import argparse
import os
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from tqdm.auto import tqdm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import wasr.models as models
from wasr.layers import *
from datasets.transforms import PytorchHubNormalization
from wasr.decoders import *

TRANSFORM_NORMALIZE = PytorchHubNormalization()


def eval_gpu(model, niter, random_input=False):
    model.eval()
    model.cuda()

    img = Image.open("examples/images/example_04.jpg")
    mask = Image.open("examples/imus/example_04.png")

    img, mask = np.array(img), np.array(mask)

    x = dict()
    x["image"] = torch.Tensor(
        np.expand_dims(TRANSFORM_NORMALIZE(img), axis=0)
    ).cuda()
    x["imu_mask"] = torch.from_numpy(
        np.expand_dims(mask, axis=0).astype(bool)
    ).cuda()

    latencies = []

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    latencies = np.zeros((niter,))

    for _ in range(10):
        _ = model(x)

    with torch.no_grad():
        for rep in tqdm(range(niter)):
            starter.record()
            _ = model(x)
            ender.record()

            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            latencies[rep] = curr_time

            if random_input:
                x["image"] = torch.randn(1, 3, 384, 512).cuda()
                x["imu_mask"] = torch.randn(1, 384, 512).cuda()

    del model
    del x
    latency = np.median(latencies)
    fps = 1000 / latency
    return (
        float(latency),
        float(fps),
        np.mean(latencies),
        np.std(latencies),
        latencies
    )


class Wrapper(nn.Module):
    def __init__(self, model):
        super().__init__()

        self.model = model

    def forward(self, x):
        return self.model(x["image"])


def main(device, niter, name=None, random_input=False):
    m1 = models.get_model("ewasr_resnet18_imu", num_classes=3, pretrained=False)
    m2 = models.get_model("wasr_resnet101_imu", num_classes=3, pretrained=False)

    model_dict = {
        "eWaSR": m1, 
        "WaSR": m2
    }

    latencies_all = []
    names_all = []
    res = {}
    for k, v in model_dict.items():
        print(f" --------------- {k} ----------------")
        if device == "GPU":
            res[k] = eval_gpu(v, niter, random_input)
        else:
            raise RuntimeError(f"Device {device} not implemented")

    for k, v in res.items():
        ms, fps, ms_mean, ms_std, latencies= v
        print(
            f"{k:30} ----- {ms:07.3f} [{ms_mean:06.2f}, {ms_std:06.2f}] ms latency ----- {fps:06.2f} FPS"
        )
        latencies_all.extend(latencies)
        names_all.extend([k] * len(latencies))

        df = pd.DataFrame({"latency": latencies_all, "name": names_all})
        df.to_csv(
            f"models_full_{device}.csv" if name is None else f"{name}_{device}.csv"
        )

    plt.figure()
    sns_plot = sns.displot(df, x="latency", hue="name", kind="kde")
    plt.xlim(0, max(df["latency"]))
    plt.savefig(f"models_full_{device}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="WaSR Network GFlops and Mparams estimation"
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        choices=("GPU"),
        default="GPU",
        help="Target device.",
    )
    parser.add_argument("-n", "--name", type=str, default=None, help="Output name")
    parser.add_argument(
        "-niter",
        "--number_of_iterations",
        type=int,
        default=100,
        help="Number of iterations.",
    )
    parser.add_argument(
        "-ri",
        "--random_input",
        action="store_true",
        help="Use random 224x224 input for backbones.",
    )
    args = parser.parse_args()

    main(
        args.device,
        args.number_of_iterations,
        name=args.name,
        random_input=args.random_input,
    )
