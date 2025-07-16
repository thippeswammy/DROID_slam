import sys

sys.path.append('droid_slam')
import yaml
from tqdm import tqdm
import numpy as np
import torch
import lietorch
import cv2
import os
import glob
import time
import argparse

from torch.multiprocessing import Process
from droid import Droid
from droid_async import DroidAsync

import torch.nn.functional as F


def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(1)


def image_stream(imagedir, calib, stride):
    """ image generator """

    calib = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy = calib[:4]

    K = np.eye(3)
    K[0, 0] = fx
    K[0, 2] = cx
    K[1, 1] = fy
    K[1, 2] = cy

    image_list = sorted(os.listdir(imagedir))[::stride]
    print("image_list=>", image_list[0])
    for t, imfile in enumerate(image_list):
        image = cv2.imread(os.path.join(imagedir, imfile))
        if len(calib) > 4:
            image = cv2.undistort(image, K, calib[4:])

        h0, w0, _ = image.shape
        h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
        w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))

        image = cv2.resize(image, (w1, h1))
        image = image[:h1 - h1 % 8, :w1 - w1 % 8]
        image = torch.as_tensor(image).permute(2, 0, 1)

        intrinsics = torch.as_tensor([fx, fy, cx, cy])
        intrinsics[0::2] *= (w1 / w0)
        intrinsics[1::2] *= (h1 / h0)

        yield t, image[None], intrinsics


def save_reconstruction(droid, save_path):
    if hasattr(droid, "video2"):
        video = droid.video2
    else:
        video = droid.video

    t = video.counter.value
    save_data = {
        "tstamps": video.tstamp[:t].cpu(),
        "images": video.images[:t].cpu(),
        "disps": video.disps_up[:t].cpu(),
        "poses": video.poses[:t].cpu(),
        "intrinsics": video.intrinsics[:t].cpu()
    }

    torch.save(save_data, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config file")
    # Core arguments
    parser.add_argument("--imagedir", type=str, help="Path to image directory")
    parser.add_argument("--calib", type=str, help="Path to calibration file")
    parser.add_argument("--t0", type=int, default=0, help="Starting frame")
    parser.add_argument("--stride", type=int, default=3, help="Frame stride")
    parser.add_argument("--weights", type=str, default="droid.pth")
    parser.add_argument("--buffer", type=int, default=1024)
    parser.add_argument("--image_size", nargs='+', type=int, default=[240, 320])
    parser.add_argument("--disable_vis", default=True, action="store_true")
    # VO-specific parameters
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--filter_thresh", type=float, default=2.4)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--keyframe_thresh", type=float, default=4.0)
    parser.add_argument("--frontend_thresh", type=float, default=16.0)
    parser.add_argument("--frontend_window", type=int, default=25)
    parser.add_argument("--frontend_radius", type=int, default=2)
    parser.add_argument("--frontend_nms", type=int, default=1)
    parser.add_argument("--backend_thresh", type=float, default=22.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)
    parser.add_argument("--upsample", action="store_true")
    parser.add_argument("--asynchronous", action="store_true")
    parser.add_argument("--frontend_device", type=str, default="cuda")
    parser.add_argument("--backend_device", type=str, default="cuda")
    parser.add_argument("--reconstruction_path", type=str, help="Path to save reconstruction")

    # Step 1: Parse CLI args first
    cli_args = vars(parser.parse_args())

    # Step 2: Load YAML config (if it exists)
    config_path = cli_args.get("config", "config.yaml")
    config_args = {}
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config_args = yaml.safe_load(f) or {}

    # Step 3: Override config values with CLI-provided ones
    final_args = {}
    for key in cli_args:
        if cli_args[key] is not None:
            final_args[key] = cli_args[key]
        elif key in config_args:
            final_args[key] = config_args[key]
        else:
            final_args[key] = None

    args = argparse.Namespace(**final_args)

    args.stereo = False
    torch.multiprocessing.set_start_method('spawn')

    if args.reconstruction_path is not None:
        args.upsample = True

    droid = None
    tstamps = []
    for (t, image, intrinsics) in tqdm(image_stream(args.imagedir, args.calib, args.stride)):
        if t < args.t0:
            continue

        if not args.disable_vis:
            show_image(image[0])

        if droid is None:
            args.image_size = [image.shape[2], image.shape[3]]
            droid = DroidAsync(args) if args.asynchronous else Droid(args)

        droid.track(t, image, intrinsics=intrinsics)

    traj_est = droid.terminate(image_stream(args.imagedir, args.calib, args.stride))
    import csv

    output_csv_path = "trajectory_output.csv"

    with open(output_csv_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow(["frame", "x", "y", "z", "qx", "qy", "qz", "qw"])

        for i, T in enumerate(traj_est):
            T = np.array(T)
            if T.shape == (7,):
                t = T[:3]  # x, y, z
                q = T[3:]  # qx, qy, qz, qw
                writer.writerow([i] + t.tolist() + q.tolist())

    if args.reconstruction_path is not None:
        save_reconstruction(droid, args.reconstruction_path)
