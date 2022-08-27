import os
import math
import time
import torch
import torchvision
from torchvision.datasets.utils import download_url
import cv2
import numpy as np
import torch
# from itertools import pairwise
from models import film_net
import itertools
from interpolator import Interpolator

# stack_and_chain = lambda levels: torch.stack(levels, dim=1).reshape(-1, *list(levels[0].size())[1:])
stack_and_chain = lambda levels: np.stack(levels, axis=1).reshape(-1, *list(levels[0].shape)[1:])


def run_on_video(input_path,
                 output_path,
                 model,
                 block_size=(128, 128),
                 block_multiplier=64,
                 buffer_size=6,
                 scale=2,
                 desired_fps=None,
                 devide="cuda"):
    assert os.path.isfile(input_path), "No file"
    assert all(np.array(block_size) % block_multiplier == 0), f"Block size should be multiple of {block_multiplier}."
    # assert not os.path.isfile(output_path), f"File with name {output_path} already exists."
    assert buffer_size > 3, "We must fit at least 3 frames in memory."

    model.to(devide)
    model.eval()

    point_time = time.time()
    reader = cv2.VideoCapture(video_path)
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

    _pad = lambda src: np.pad(src,
                              mode="constant",
                              pad_width=((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)),
                              constant_values=((0, 0), (0, 0), (0, 0)))

    def transform_and_split_in_patches(x):
        x = x[..., ::-1]  # (change color channels order) BGR -> RGB
        x = x / 255  # (normalization)
        x = x.transpose(0, 3, 1, 2)  # (reshape) B x H x W x C -> B x C x H x W
        x = torch.Tensor(x)  # (to tensor)
        x_splits = torch.split(x, block_height, 2)  # (split in x axis)
        return list(map(lambda x: torch.split(x, block_width, 3), x_splits))  # (split in y axis)

    def _read(video_capture, number_of_frames=1):
        assert number_of_frames >= 1, ""
        cached_frames = []
        for _ in range(number_of_frames):
            status, frame = video_capture.read()
            if not status:
                break
            cached_frames.append(_pad(frame))
        return status, np.array(cached_frames)

    number_of_frames = int(buffer_size * fps / target_fps)

    status = True
    while status:
        status, frames = _read(reader, number_of_frames)
        split_of_splits = transform_and_split_in_patches(frames)

        predicted_splits_of_splits = []
        for i, splits in enumerate(split_of_splits):
            predicted_splits = []
            for j, batch_of_blocks in enumerate(splits):
                batch_of_blocks_gpu = batch_of_blocks.to(devide)
                frames = stack_and_chain([frame.cpu().detach().numpy() for frame in interpolator(batch_of_blocks_gpu)])
                predicted_splits.append(frames)
                del batch_of_blocks_gpu
                torch.cuda.empty_cache()
            predicted_splits_of_splits.append(np.concatenate(predicted_splits, axis=3))
        predicted_frames = np.concatenate(predicted_splits_of_splits, axis=2)
        predicted_frames = predicted_frames[:, :, top_pad:-bottom_pad, left_pad:-right_pad]
        predicted_frames = np.array(np.clip(predicted_frames * 255, 0, 255) + 0.5, dtype=np.uint8)
        predicted_frames = predicted_frames.transpose(0, 2, 3, 1)
        predicted_frames = predicted_frames[..., ::-1]
        predicted_frames = tuple(predicted_frames)
        for frame in predicted_frames:
            writer.write(frame)
        print("")
    print(f"pass {time.time() - point_time:0.4f} sec.")
    reader.release(), writer.release()


film_net_model = film_net()
film_net_model.load_state_dict(torch.load("./checkpoints/film_net.pt"))
film_net_model.to("cuda")
film_net_model.eval()

video_path = "/home/ivan/Downloads/1651304091_looped_1651304091.mp4"
output_path = "/home/ivan/Experiments/FILM/moove2.mp4"

run_on_video(video_path, output_path, film_net_model, desired_fps=75)

# for frame_a_no in range(number_of_frames-1):
#     if i0_transforms_mask[frame_a_no]:
#         canvases[0], masks[0] = draw_canvas_a_on_canvas_b(canvases[frame_a_no],
#                                                           masks[frame_a_no],
#                                                           canvases[0],
#                                                           masks[0],
#                                                           i0_transforms[frame_a_no],
#                                                           transform=lambda x: x,
#                                                           **canvas_config)
#         canvas = np.copy(canvases[0])
#         for j in range(len(traces)):
#             trace, trace_mask = traces[j], trace_masks[j]
#             draw_trace(canvas, trace, trace_mask, frame_a_no, 3)
#         cv2.putText(canvas,
#                     f"{frame_a_no}",
#                     org=(40, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(253, 162, 113),
#                     thickness=2,
#                     lineType=cv2.LINE_AA)
#         out.write(canvas)
# out.release()
# cv2.destroyAllWindows()
# print("DONE DONE DONE!")


# batch_size = 4

# rate = desired_fps / fps
# print("fps:", fps,
#       "desired fps:", desired_fps,
#       "upsampling rate:", rate)
# print(
#     "number of pure upsamplings:", int(math.log2(rate)),
#     "number of unhandled points:", desired_fps - 2 ** int(math.log2(rate)) * fps)
#
# number_pure_upsampling = int(math.log2(rate))
# current_fps = 2 ** number_pure_upsampling * fps
# missed_frames = desired_fps - current_fps
#
# holes_pattern_for_missed_frames = itertools.combinations(range(current_fps), missed_frames)
# for pattern in holes_pattern_for_missed_frames:
#     permutation = np.zeros(current_fps)
#     permutation[pattern] = 1
#
# capture.release()
#
# print(capture)
# stream = "video"
# video = torchvision.io.VideoReader(video_path, stream)
# print(video.get_metadata())

# from models import film_net
#
# side = 2560
# first_image, second_image = "/home/ivan/Experiments/FILM/knights_c/one.png", \
#                             "/home/ivan/Experiments/FILM/knights_c/two.png"
#
#
# x, y = np.array(cv2.imread(first_image) / 255, dtype=np.float32), \
#        np.array(cv2.imread(second_image) / 255, dtype=np.float32)
# _crop_n_shape = lambda src: src[np.newaxis, -side:, :side].transpose(0, 3, 1, 2)
# x, y = _crop_n_shape(x), _crop_n_shape(y)
# x, y = x[:, [2, 1, 0]], y[:, [2, 1, 0]]
# x, y = torch.Tensor(x), torch.Tensor(y)
#
# interpolator = film_net()
# interpolator.load_state_dict(torch.load("./checkpoints/film_net.pt"))
# interpolator.eval()
# out = interpolator(x, y)
#
# e = np.array(np.clip(out.permute(0, 2, 3, 1).cpu().detach().numpy()[0] * 255, 0, 255) + 0.5, dtype=np.uint8)
# cv2.imwrite("/home/ivan/Experiments/FILM/knights_c/ress.jpg", e)
# cv2.imshow("res", np.array(
#     np.clip(out.permute(0, 2, 3, 1).cpu().detach().numpy()[0] * 255, 0, 255) + 0.5, dtype=np.uint8))
# cv2.waitKey()
# print(out.size())


# fourcc = cv2.VideoWriter_fourcc(*"mp4v")
# out = cv2.VideoWriter('output3.mp4', fourcc, 10.0, canvases[0].shape[:2], True)
# for frame_a_no in range(number_of_frames-1):
#     if i0_transforms_mask[frame_a_no]:
#         canvases[0], masks[0] = draw_canvas_a_on_canvas_b(canvases[frame_a_no],
#                                                           masks[frame_a_no],
#                                                           canvases[0],
#                                                           masks[0],
#                                                           i0_transforms[frame_a_no],
#                                                           transform=lambda x: x,
#                                                           **canvas_config)
#         canvas = np.copy(canvases[0])
#         for j in range(len(traces)):
#             trace, trace_mask = traces[j], trace_masks[j]
#             draw_trace(canvas, trace, trace_mask, frame_a_no, 3)
#         cv2.putText(canvas,
#                     f"{frame_a_no}",
#                     org=(40, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(253, 162, 113),
#                     thickness=2,
#                     lineType=cv2.LINE_AA)
#         out.write(canvas)
# out.release()
# cv2.destroyAllWindows()
# print("DONE DONE DONE!")
