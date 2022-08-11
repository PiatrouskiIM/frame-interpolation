import os
import math
import time
import torch
import torchvision
from torchvision.datasets.utils import download_url
import cv2
import numpy as np
import torch
import itertools
from models import film_net

point_time = time.time()

video_path = "/home/ivan/Downloads/1651304091_looped_1651304091.mp4"
output_path = "/home/ivan/Experiments/FILM/moove2.mp4"
# stream = "video"
# video = torchvision.io.VideoReader(video_path, stream)
# print(video.get_metadata())


side = 256
batch_size = 2
_crop_n_shape = lambda src: src[np.newaxis, -side:, :side].transpose(0, 3, 1, 2)
interpolator = film_net()
interpolator.load_state_dict(torch.load("./checkpoints/film_net.pt"))
interpolator.to("cuda")
interpolator.eval()

assert os.path.isfile(video_path), "No file"
capture = cv2.VideoCapture(video_path)
fps = math.floor(capture.get(cv2.CAP_PROP_FPS))
height, width = capture.get(cv2.CAP_PROP_FRAME_HEIGHT), capture.get(cv2.CAP_PROP_FRAME_WIDTH)
desired_fps = math.floor(120.0)

out = cv2.VideoWriter(filename=output_path,
                      fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
                      fps=2 * fps,
                      frameSize=(int(side), int(side)),
                      isColor=True)

initial_crop = lambda src: src[-side:, :side]
success, first_image = capture.read()
# out.write(first_image[-side:, :side])

images = [initial_crop(first_image)]
#
# x = np.array(images) / 255
# x = x.transpose(0, 3, 1, 2)
# x = x[:, [2, 1, 0]]
# x = torch.Tensor(x).to("cuda")
#
# full_pyramid_x = interpolator.extract_features_pyramid(x)

while success:
    print("b")
    for _ in range(batch_size):
        success, frame = capture.read()
        if not success:
            break
        images.append(initial_crop(frame))

    x = np.array(images) / 255
    x = x.transpose(0, 3, 1, 2)
    x = x[:, [2, 1, 0]]
    x = torch.Tensor(x).to("cuda")

    full_pyramid_x = interpolator.extract_features_pyramid(x)
    # full_pyramid_x = list(map(lambda x, y: torch.cat((x[[-1]], y), dim=0), full_pyramid_x, pyramid))

    # mid = interpolator(x[:-1], x[1:])
    #
    mid = interpolator.run_on_features(list(map(lambda x: x[:-1], full_pyramid_x)),
                                       list(map(lambda x: x[1:], full_pyramid_x)))

    mid = np.array(np.clip(mid.permute(0, 2, 3, 1).cpu().detach().numpy() * 255, 0, 255) + 0.5, dtype=np.uint8)
    for left, mid in zip(images, mid):
        out.write(left)
        out.write(mid[..., [2, 1, 0]])
    del x, mid
    images = [images[-1]]

print(f"pass {time.time() - point_time:0.4f} sec.")
capture.release()
out.release()
cv2.destroyAllWindows()

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
