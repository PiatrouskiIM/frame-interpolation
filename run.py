import os
import math
import time
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets.utils import download_url
import cv2
import numpy as np
import torch
from models import film_net
from interpolator import Interpolator

stack_and_chain = lambda levels: np.stack(levels, axis=1).reshape(-1, *list(levels[0].shape)[1:])


def split_in_patches(x, block_width, block_height):
    x = torch.Tensor(x)
    x_splits = torch.split(x, block_height, dim=2)  # (split in x axis)
    return list(map(lambda x: torch.split(x, block_width, dim=3), x_splits))  # (split in y axis)


floor = lambda x: int(math.floor(x))


def preprocess(x):
    x = x[..., ::-1]  # (change color channels order) BGR -> RGB
    x = x / 255  # (normalization)
    return x.transpose(0, 3, 1, 2)  # (reshape) B x H x W x C -> B x C x H x W


def postprocess(x):
    x = x.transpose(0, 2, 3, 1)  # (reshape) B x C x H x W -> B x H x W x C
    x = np.array(np.clip(x * 255, 0, 255) + 0.5, dtype=np.uint8)  # (changing range) [0, 1] -> [0, 255]
    return x[..., ::-1]


def run_on_video(input_path,
                 output_path,
                 model,
                 block_size=(128, 128),
                 block_multiplier=64,
                 buffer_size=16,
                 scale=2,
                 desired_fps=None,
                 device="cuda"):
    # """Interpolate inbetween frames of the given video to specifies framerate."""
    assert os.path.isfile(input_path), f"{input_path} not found."
    assert all(np.array(block_size) % block_multiplier == 0), f"Block size should be a multiple of {block_multiplier}."
    assert buffer_size >= 3, f"Invalid buffer size {buffer_size}. The buffer size must be at least 3."

    model.to(device)
    model.eval()

    point_time = time.time()


    reader = cv2.VideoCapture(input_path)
    fps = int(math.floor(reader.get(cv2.CAP_PROP_FPS)))

    target_fps = fps * scale
    if desired_fps:
        target_fps = desired_fps

    interpolator = Interpolator(model, fps, target_fps)
    height, width = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))

    writer = cv2.VideoWriter(filename=output_path,
                             fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
                             fps=target_fps,
                             frameSize=(int(width), int(height)),
                             isColor=True)
    total_vertical_pad, total_horizontal_pad = (block_multiplier - height % block_multiplier) % block_multiplier, \
                                               (block_multiplier - width % block_multiplier) % block_multiplier
    top_pad, left_pad = total_vertical_pad // 2, total_horizontal_pad // 2
    bottom_pad, right_pad = total_vertical_pad - top_pad, total_horizontal_pad - left_pad
    block_width, block_height = block_size

    _pad, _crop = lambda src: np.pad(src,
                                     mode="constant",
                                     pad_width=((0, 0), (top_pad, bottom_pad), (left_pad, right_pad), (0, 0)),
                                     constant_values=((0, 0), (0, 0), (0, 0), (0, 0))), \
                  lambda src: src[:, top_pad:-bottom_pad, left_pad:-right_pad]

    number_of_frames = int(buffer_size * fps / target_fps)
    frame_count = int(math.floor(reader.get(cv2.CAP_PROP_FRAME_COUNT)))
    stats = []

    status, last_frame = reader.read()
    frame_no = 0
    while status:
        frames = [last_frame]
        for _ in range(number_of_frames - 1):
            status, frame = reader.read()
            frame_no += 1
            if not status:
                break
            frames.append(frame)
        last_frame = frames[-1]
        split_of_splits = split_in_patches(preprocess(_pad(frames)), block_width=block_width, block_height=block_height)

        predicted_splits_of_splits = []
        batch_time = time.time()
        for i, splits in enumerate(split_of_splits):
            predicted_splits = []
            for j, batch_of_blocks in enumerate(splits):
                batch_of_blocks_gpu = batch_of_blocks.to(device)
                frames = stack_and_chain([block.cpu().detach().numpy()
                                          for block in interpolator(batch_of_blocks_gpu, frame_no)])
                predicted_splits.append(frames)
            predicted_splits_of_splits.append(np.concatenate(predicted_splits, axis=3))
        predicted_frames = np.concatenate(predicted_splits_of_splits, axis=2)
        for frame in _crop(postprocess(predicted_frames)):
            writer.write(frame)
        stats.append(time.time() - batch_time)
        stats = stats[-10:]
        estimate_duration = np.array(stats).mean() * (frame_count/(number_of_frames-1))
        print(f"{frame_no/fps:0.1f}/{frame_count/fps:0.1f} sec. +{(number_of_frames-1)/fps:0.1f} sec."
              f" {time.time() - point_time:0.1f}\~{estimate_duration:0.1f}sec.  +{time.time() - batch_time:0.1f}.")
    writer.write(last_frame)
    print(f"pass {time.time() - point_time:0.2f} sec.")
    reader.release(), writer.release()


film_net_model = film_net()
film_net_model.load_state_dict(torch.load("./checkpoints/film_net.pt"))
film_net_model.to("cuda")
film_net_model.eval()

video_path = "/home/ivan/Downloads/1651304091_looped_1651304091.mp4"
output_path = "/home/ivan/Experiments/FILM/moove.mp4"

run_on_video(video_path, output_path, film_net_model, desired_fps=25)
