import os
import cv2
import time
import math
import pathlib
import torch
from utils.transforms import AssembleFromPatches, SplitIntoPatches
import numpy as np


@torch.no_grad()
def run_on_video(input_path,
                 output_path,
                 model,
                 block_size=256,
                 batch_size=1,
                 device="cuda",
                 max_shift=16) -> None:
    point_time = time.time()
    assert os.path.isfile(input_path), f"{input_path} not found."

    reader = cv2.VideoCapture(input_path)
    fps = int(math.ceil(reader.get(cv2.CAP_PROP_FPS)))
    size = (int(reader.get(cv2.CAP_PROP_FRAME_WIDTH)), int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    writer = cv2.VideoWriter(filename=output_path,
                             fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
                             fps=fps * 2,
                             frameSize=size,
                             isColor=True)
    patch_logic_params = dict(size=size, patch_size=block_size, margin=max_shift)
    split, assemble = SplitIntoPatches(**patch_logic_params), AssembleFromPatches(**patch_logic_params)
    frame_count = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    processing_stats = [0]

    frame_no = 0
    status, frame = reader.read()
    if not status:
        return
    writer.write(frame)
    frames = [frame]
    while status:
        batch_time = time.time()
        for _ in range(batch_size):
            frame_no += 1
            status, frame = reader.read()
            if not status:
                break
            frames.append(frame)
        frames = frames[-batch_size - 1:]
        predicted_patches = []
        for batch in split(preprocess(np.stack(frames))):
            x = torch.Tensor(batch).to(device)
            predicted_patch = model(x[:-1], x[1:]).cpu().detach().numpy()
            predicted_patches.append(predicted_patch)
        for frame, in_between_frame in zip(frames[1:], postprocessing(assemble(np.stack(predicted_patches)))):
            writer.write(frame), writer.write(in_between_frame)

        step_time = time.time() - batch_time
        processing_stats.append(step_time)
        processing_stats = processing_stats[-10:]
        print(f"total time: "
              f"{time.time() - point_time:0.2f}/~{frame_count / batch_size * np.mean(processing_stats):0.2f} sec., "
              f"total frames: {str(frame_no).zfill(3)}/{str(frame_count).zfill(3)} frames.")
    reader.release(), writer.release()
    print(f"pass {time.time() - point_time:0.2f} sec.")


@torch.no_grad()
def run_on_image_pair(first_image_path,
                      second_image_path,
                      output_folder_path,
                      model,
                      device="cuda",
                      batch_size=1,
                      block_size=256,
                      number_of_applications=1,
                      max_shift=16):
    assert os.path.isfile(first_image_path), f"{first_image_path} not found."
    assert os.path.isfile(second_image_path), f"{second_image_path} not found."
    pathlib.Path(output_folder_path).mkdir(parents=True, exist_ok=True)

    model.to(device)
    model.eval()

    image_a, image_b = cv2.imread(first_image_path), cv2.imread(second_image_path)
    height, width = np.minimum(image_a.shape[:2], image_b.shape[:2])
    input_frames = [image_a[:height, :width], image_b[:height, :width]]

    patch_logic_params = dict(size=(width, height), patch_size=block_size, margin=max_shift)
    split, assemble = SplitIntoPatches(**patch_logic_params), AssembleFromPatches(**patch_logic_params)

    while number_of_applications > 0:
        predicted_frames = []
        for i in range(len(input_frames) - batch_size):
            frames = input_frames[i:i + batch_size + 1]

            predicted_patches = []
            for batch in split(preprocess(np.stack(frames))):
                x = torch.Tensor(batch).to(device)
                predicted_patch = model(x[:-1], x[1:]).cpu().detach().numpy()
                predicted_patches.append(predicted_patch)
            predicted_frames.extend(list(postprocessing(assemble(np.stack(predicted_patches, axis=0)))))

        output_frames = input_frames[:1]
        for frame_a, frame_b in zip(predicted_frames, input_frames[1:]):
            output_frames.append(frame_a)
            output_frames.append(frame_b)
        input_frames = output_frames
        number_of_applications -= 1
    for i, frame in enumerate(output_frames):
        frame_output_path = os.path.join(output_folder_path, str(i).rjust(4, "0") + ".jpg")
        cv2.imwrite(frame_output_path, frame)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--input", type=str, required=True, default="input.mp4", help="input video path.")
    parser.add_argument('-o', "--output", type=str, default="output.mp4", help="output video path.")
    parser.add_argument("--device",  default='cuda', choices=["cuda", "cpu"], help="device.")
    parser.add_argument("--batch", default=1, type=int, help="device.")
    args = parser.parse_args()

    from models import film_net, FILMNet_Weights

    film_net_model = film_net(weights=FILMNet_Weights.DEFAULT)
    film_net_model.to(args.device)
    film_net_model.eval()
    preprocess, postprocessing = FILMNet_Weights.transforms(), FILMNet_Weights.postprocessing()

    run_on_video(args.input, args.output, film_net_model, batch_size=args.batch, device=args.device)
