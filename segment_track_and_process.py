import os
import pandas as pd
import numpy as np
from glob import glob
from scipy import linalg
from matplotlib import pyplot as plt
from src import utils
from tqdm import tqdm
import matplotlib
import argparse
from configs.roi_experiments import directories

matplotlib.use('Agg')

def main(config):
    """
    Track cells, generate stats, and process ROIs.

    This function reads the configuration, processes image data to track cell nuclei
    and cytoplasm, generates ROIs, and optionally creates visualization plots.

    Args:
        config (str): Path to the configuration YAML file.

    Returns:
        None. Outputs are saved to files specified in the configuration.
    """
    cfg = utils.read_config(config)
    # Directories and versioning
    segment_dir = cfg["segmentation_model"]["segments"]
    version = cfg["version"]

    # Data gen settings
    small_vol_thresh = cfg["tracking"]["small_vol_thresh"]
    dilation_radius = cfg["tracking"]["dilation_radius"]
    dilation_radius_far = cfg["tracking"]["dilation_radius_far"]
    roi_size = cfg["tracking"]["roi_size"]
    tracking_strategy = cfg["tracking"]["tracking_strategy"]
    detrend_mean_nucleus = cfg["tracking"]["detrend_mean_nucleus"]
    linear_detrended = cfg["tracking"]["linear_detrended"]
    use_dff = cfg["tracking"]["use_dff"]

    # Visualization settings
    debug = cfg["tracking"]["debug"]
    plot_trended_video = cfg["tracking"]["plot_trended_video"]
    plot_detrended_video = cfg["tracking"]["plot_detrended_video"]
    plot_figures = cfg["tracking"]["plot_figures"]
    generate_rois = cfg["tracking"]["generate_rois"]
    save_npy_rois = cfg["tracking"]["save_npy_rois"]
    normalize_rois = cfg["tracking"]["normalize_rois"]

    # Plotting settings
    fast_process = cfg["tracking"]["fast_process"]

    if fast_process:
        matplotlib.use('Agg')  # Use a non-interactive backend

    # Get directories
    dirs = glob(os.path.join(segment_dir, "*", "*", "*"))
    dirs = [x for x in dirs if ".png" not in x]

    # Get files
    hroi = roi_size // 2
    spreadsheets = []
    for dr in tqdm(dirs, total=len(dirs), desc="Creating images"):
        images = glob(os.path.join(dr, "image*"))
        images = sorted(images, key=os.path.getmtime)
        images = np.asarray(images)
        preds = np.asarray([x.replace("image_", "pred_") for x in images])

        # Track nuclei
        nuclei = utils.track(preds, small_vol_thresh, tracking_strategy)

        # Load video
        video = utils.load_video(images)

        # Get cytoplasms for each nucleus
        nuclei_ids, cytoplasms, cytoplasms_far = utils.get_cytoplasms(nuclei, dilation_radius, dilation_radius_far, tracking_strategy=tracking_strategy)

        if debug:
            if tracking_strategy == "first":
                plt.subplot(131);plt.imshow(video[0]);plt.subplot(132);plt.imshow(nuclei);plt.subplot(133);plt.imshow(cytoplasms);plt.show()
            elif tracking_strategy == "all":
                # plt.subplot(121);plt.imshow(nuclei[0]);plt.subplot(122);plt.imshow(cytoplasms[0]);plt.show()
                utils.colored_video(nuclei, "nuclei.mp4")
                utils.colored_video(cytoplasms, "cytoplasms.mp4")
        else:
            # Save sanity check plots
            sd = dr.replace(segment_dir, directories["sanity_checks"])
            utils.makedirs(sd)
            f = plt.figure()
            if tracking_strategy == "first":
                plt.subplot(121);plt.imshow(nuclei);plt.subplot(122);plt.imshow(cytoplasms)
            elif tracking_strategy == "all":
                plt.subplot(121);plt.imshow(nuclei[0]);plt.subplot(122);plt.imshow(cytoplasms[0])
            else:
                raise NotImplementedError(tracking_strategy)
            try:
                plt.savefig("{}.png".format(sd))
            finally:
                plt.close(f)
                plt.close("all")  # In case there are more figures

        # Take average of nuclei and deconvolve from video
        mean_nuc, std_nuc, mean_cp, std_cp, mean_cpf, std_cpf, nucs, cps, cps_far = utils.get_activities(video, nuclei, cytoplasms, cytoplasms_far, nuclei_ids, tracking_strategy=tracking_strategy)

        if plot_trended_video:
            vvideo = (video - video.min()) / (video.max() - video.min())
            vvideo = (vvideo * 255).astype(np.uint8)
            vvideo = vvideo[..., None].repeat(3, -1)
            utils.create_video(vvideo, "trended.mp4")

        res_video = video.reshape(len(video), -1).mean(1)
        if detrend_mean_nucleus:
            # Remove nuclear trend from video
            X = np.stack((np.ones_like(mean_nuc), mean_nuc), 1)
        else:
            # Remove mean trend
            X = np.stack((np.ones_like(res_video), res_video), 1)
        coef, resids, rank, s = linalg.lstsq(X, res_video)  # Find leastsq fit and remove it for each piece
        detrend_video = video - (X @ coef)[:, None, None]
        mean_nuc, std_nuc, mean_cp, std_cp, mean_cpf, std_cpf, nucs, cps, cps_far = utils.get_activities(video, nuclei, cytoplasms, cytoplasms_far, nuclei_ids, tracking_strategy=tracking_strategy)

        if plot_detrended_video:
            vdetrend_video = (detrend_video - detrend_video.min()) / (detrend_video.max() - detrend_video.min())
            vdetrend_video = (vdetrend_video * 255).astype(np.uint8)
            vdetrend_video = vdetrend_video[..., None].repeat(3, -1)
            utils.create_video(vdetrend_video, "detrended.mp4")

        # Get ROIs
        if generate_rois:
            df = utils.seq_generate_roi_data(dr, nuclei, cytoplasms, cytoplasms_far, video, detrend_video, nuclei_ids, hroi, roi_size, directories, normalize_rois, linear_detrended, use_dff, debug, segment_dir, plot_figures, save_npy_rois)
            spreadsheets.append(df)
    df = pd.concat(spreadsheets)
    df.to_csv("trace_analysis_{}.csv".format(version))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default=None, type=str, help="Pass the path to your project config .yaml file.")
    args = vars(parser.parse_args())
    main(**args)
