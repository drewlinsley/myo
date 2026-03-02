import os
import numpy as np
import cv2
from skimage.transform import rescale
from glob import glob
import argparse
from src import utils


def main(config):
    """
    Preprocess calcium imaging videos from MP4 format.

    This function reads .mp4 video files, processes them by smoothing and rescaling,
    and saves the results as both .npy files and .mp4 videos.

    Args:
        config (str): Path to the configuration YAML file.

    Returns:
        None
    """
    cfg = utils.read_config(config)
    raw_videos = cfg["ca2_data"]["raw_videos"]
    fps = cfg["ca2_data"]["fps"]
    processed_videos = cfg["ca2_data"]["processed_videos"]
    processed_npys = cfg["ca2_data"]["processed_npys"]

    fns = glob(os.path.join(raw_videos, "*Experiment*", "*", "*.mp4"))

    os.makedirs(processed_videos, exist_ok=True)
    os.makedirs(processed_npys, exist_ok=True)

    for fn in fns:
        print("Preparing {}".format(fn))
        vid_file = os.path.join(raw_videos, fn)
        out_numpy_path = "{}.npy".format(processed_npys + fn.replace(raw_videos, "").replace(".mp4", ""))
        out_video_path = "{}".format(processed_videos + fn.replace(raw_videos, ""))
        if os.path.exists(out_numpy_path):
            continue

        splits = fn.replace(raw_videos, "").split(os.path.sep)
        for s in range(1, len(splits)):
            tnpy = processed_npys + os.path.sep + os.path.sep.join(splits[1:s])
            os.makedirs(tnpy, exist_ok=True)
            tmp4 = processed_videos + os.path.sep + os.path.sep.join(splits[1:s])
            os.makedirs(tmp4, exist_ok=True)

        # Load MP4 video
        cap = cv2.VideoCapture(vid_file)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to grayscale if needed
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(frame)
        
        cap.release()
        narr = np.array(frames)

        # Smooth data
        sm_arr_1 = []
        for n in range(narr.shape[0]):
            sm_arr_1.append(rescale(narr[n], 0.5, preserve_range=True))
        sm_arr_1 = np.asarray(sm_arr_1).transpose(1, 0, 2)

        sm_arr = []
        for n in range(sm_arr_1.shape[0]):
            sm_arr.append(rescale(sm_arr_1[n], [0.5, 1], preserve_range=True))
        sm_arr = np.asarray(sm_arr)

        # Change dtype and range
        sm_arr = sm_arr.astype(np.float32)
        sm_arr = (sm_arr - sm_arr.min()) / (sm_arr.max() - sm_arr.min())
        sm_arr = (sm_arr * 255).astype(np.uint8)

        # Save as numpy
        np.save(out_numpy_path, sm_arr)

        # Save video
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        video_writer = cv2.VideoWriter(out_video_path, fourcc, fps, (sm_arr.shape[2], sm_arr.shape[1]))
        for frame in sm_arr:
            video_writer.write(frame[..., None].repeat(3, -1))
        video_writer.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess calcium imaging videos from MP4 format.")
    parser.add_argument('-c', '--config', default=None, type=str, help="Path to the project configuration YAML file.")
    args = vars(parser.parse_args())
    main(**args)