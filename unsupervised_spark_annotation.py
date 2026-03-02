import os
import argparse
import numpy as np
import torch
from glob import glob
from src import transforms as video_transforms
from torchvision.transforms import v2 as transforms
from segmentation_models_pytorch_3d.decoders.classifier import Unet
from src.dataloaders import Classification_test_whole
from src import utils
from skimage.measure import label as sklabel
from skimage.measure import regionprops as rp
import joblib
from src.grads import SmoothGrad
import re
import pandas as pd
from configs.roi_experiments import directories
from configs.roi_experiments import test_exps


def main(
        config,
        dims="BCDHW",
        remove_npy_pattern=r'_[^_]*\.npy$',
        timesteps=32,
        HW=128,
        longest_spark=16,
        smallest_spark=3,
        max_area=10000000,
        sd=0.5,
        mask_thresh=95,
        num_frames=16,
        timm_model="resnext50_32x4d"):
    """
    Perform calcium spark detection and analysis on cellular imaging data.

    This script loads a pre-trained model, processes cellular imaging data,
    detects calcium sparks, and generates statistics for each detected spark
    and for each cell. It also optionally creates annotated videos of the
    detected sparks.

    Args:
        config (str): Path to the configuration file.
        dims (str, optional): Dimension order of the input data. Defaults to "BCDHW".
        remove_npy_pattern (str, optional): Regex pattern to remove from filenames. Defaults to r'_[^_]*\.npy$'.
        timesteps (int, optional): Number of timesteps for the model. Defaults to 32.
        HW (int, optional): Height and width of the input images. Defaults to 128.
        longest_spark (int, optional): Maximum allowed duration of a spark. Defaults to 16.
        smallest_spark (int, optional): Minimum size of a spark to be considered. Defaults to 3.
        max_area (int, optional): Maximum area of a spark. Defaults to 10000000.
        sd (float, optional): Standard deviation for Gaussian filtering. Defaults to 0.5.
        mask_thresh (float, optional): Threshold for creating binary masks. Defaults to 0.95.
        num_frames (int, optional): Number of frames to process. Defaults to 16.
        timm_model (str, optional): Name of the timm model to use. Defaults to "resnext50_32x4d".

    Returns:
        None

    Outputs:
        - Annotated videos of detected sparks (if plot_videos is True in config)
        - CSV file with statistics for each detected spark (event_csv_output_path in config)
        - CSV file with average spark statistics for each cell (omni_csv_output_path in config)
    """
    cfg = utils.read_config(config)
    model_type = cfg["sparks"]["model_type"]
    te_bs = cfg["sparks"]["te_bs"]
    test_workers = cfg["sparks"]["test_workers"]

    omni_csv_output_path = cfg["sparks"]["omni_csv_output_path"]
    event_csv_output_path = cfg["sparks"]["event_csv_output_path"]
    plot_videos = cfg["sparks"]["plot_videos"]
    load_ckpt = cfg["sparks"]["ckpt"]

    whole_roi_dir = directories["whole_cell_roi_spl_dir"]
    nucleus_mask_dir = directories["nucleus_mask_roi_dir"]
    annotated_video_dir = directories["annotated_video_dir"]
    model_asset_dir = directories["model_asset_dir"]

    pls_path = os.path.join(model_asset_dir, cfg["sparks"]["pls_file"])
    nmf_path = os.path.join(model_asset_dir, cfg["sparks"]["nmf_file"])
    nmf_arg_path = os.path.join(model_asset_dir, "nmf_score_arg_model_{}.npy".format(model_type))

    pre_HW = HW + 32  # size to random crop from
    time_kernel = [timesteps, 1, 1]
    fl_path = os.path.join(model_asset_dir, "files_{}.npy".format(model_type))
    sc_path = os.path.join(model_asset_dir, "scores_{}.npy".format(model_type))
    os.makedirs(annotated_video_dir, exist_ok=True)

    # Here we additionally normalize them to the pretrained weights of our model
    accelerator, _, tqdm, TIMM = utils.prepare_env(timm_model)
    test_trans = transforms.Compose([
        video_transforms.CenterCrop(HW),
        video_transforms.ToTensor(),
    ])

    # Select files for training and testing
    with accelerator.main_process_first():
        _, test_y, train_y, test_u_conds, whole_files, whole_y = utils.prepare_data(model_type, whole_roi_dir, test_exps, use_whole_files=True)

    print("Preparing models")
    model = Unet(
        encoder_name=timm_model,
        in_channels=1,
        classes=train_y.max() + 1,
        time_kernel=time_kernel,
    )
    print("Using ckpt: {}".format(load_ckpt))
    ckpt_weights = torch.load(load_ckpt, map_location="cuda:0")
    key_check = [x for x in ckpt_weights.keys()][0]
    if key_check.split(".")[0] == "module":
        ckpt_weights = {k.replace("module.", ""): v for k, v in ckpt_weights.items()}
    model.load_state_dict(ckpt_weights)
    model.eval()

    # Load decomposition
    grad_model = SmoothGrad(model, cuda=True)
    pls_clfd = joblib.load(pls_path)
    nmf_clfd = joblib.load(nmf_path)
    nmf_arg = np.load(nmf_arg_path)
    mu = torch.from_numpy(pls_clfd._x_mean).float().cuda()[None]
    std = torch.from_numpy(pls_clfd._x_std).float().cuda()[None]
    rotations = torch.from_numpy(pls_clfd.x_rotations_).float().cuda()

    ##### GET MOST/LEASY SPARKY
    fls = np.load(fl_path)
    scores = np.load(sc_path)
    pattern = r'(.*?)__.*\.npy$'
    replacement = r'\1.npy'
    mask_files = [re.sub(pattern, replacement, x.replace(whole_roi_dir, nucleus_mask_dir)) for x in fls]
    filt_whole_files, filt_y, filt_test_files, filt_masks = utils.filter_whole_files(
        whole_files=whole_files,
        test_y=test_y,
        scores=scores,
        test_u_conds=test_u_conds,
        mask_files=mask_files,
        fls=fls)
    test_dataset = Classification_test_whole(
        filt_test_files,
        filt_whole_files,  # test_files,
        filt_masks,
        filt_y,
        mu=TIMM["mean"],
        std=TIMM["std"],
        pre_HW=pre_HW,
        transform=test_trans,
        time=timesteps,
        dims=dims)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=te_bs, shuffle=False, drop_last=True, num_workers=test_workers)
    model, grad_model, test_loader = accelerator.prepare(model, grad_model, test_loader)

    ### GET GRADS
    all_stats, cell_stats = [], []
    for source in tqdm(test_loader, total=len(test_loader)):
        _, whole_norm, whole, nuc, _, fn = source
        grads, sss = grad_model(whole_norm, rotations, mu, std, nmf_clfd, nmf_arg)
        grads = grads.squeeze()
        nuc = nuc.squeeze()

        # Normalize videos for display
        whole = whole.cpu().numpy()
        whole = (whole - whole.min((2, 3, 4), keepdims=True)) / (whole.max((2, 3, 4), keepdims=True) - whole.min((2, 3, 4), keepdims=True))
        whole = (whole * 255).astype(np.uint8)
        whole = whole[:, :, :num_frames]

        # Trim, normalize, and remove nucleus from grads
        grads = grads[:, :num_frames]
        nuc = (nuc[:, :num_frames] > 0).float().cpu().numpy()
        grads = grads * (1 - nuc)

        # Smooth then threshold grads
        grads = np.asarray([utils.gaussian_filter_3d(x, sd) for x in grads]).astype(np.float32)
        masks = np.asarray([x > np.percentile(x, mask_thresh) for x in grads]).astype(np.float32)

        # Process grads
        video_stats = []
        for mask, vid, file_name, ss in zip(masks, whole, fn, sss):
            vid = vid.squeeze()
            cell_number = int(re.search(remove_npy_pattern, file_name).group().split(".")[0].replace("_", ""))
            mlabs = sklabel(mask)
            rps = rp(mlabs)
            fixed_mask = np.zeros_like(mask)

            # Filter noise and get stats
            for r in rps:
                coords = r.coords
                
                # Get y/x of cell
                rc = file_name.split("rc_")[-1].split("_")
                row = int(rc[0])
                col = int(rc[1])
                perturbation = file_name.split(os.path.sep)[-1].split("__")[0].split("_")[-2]
                if "post" in perturbation:
                    perturbation = "post_iso"
                elif "ca" in perturbation:
                    perturbation = "iso"
                else:
                    perturbation = "baseline"

                stat = {
                    "file_name": file_name,
                    "cell_number": cell_number,
                    "spark_score": ss,
                    "cell_row": row,
                    "cell_col": col,
                    "experimental_condition": file_name.split(os.path.sep)[2],
                    "experiment_name": file_name.split(os.path.sep)[1].split(" ")[1],
                    "perturbation": perturbation,
                    "num_sparks": len(rps)
                }

                if r.label > 0 and r.area > smallest_spark and len(np.unique(coords[:, 0])) <= longest_spark and r.area < max_area:
                    fixed_mask[coords[:, 0], coords[:, 1], coords[:, 2]] = r.label
                    spark_act = vid[r.coords[:, 0], r.coords[:, 1], r.coords[:, 2]]
                    fwhm = utils.calculate_fwhm(spark_act)
                    stat["mass"] = r.area  # Referred to as mass in our discussions
                    stat["mean"] = spark_act.mean()
                    stat["magnitude"] = spark_act.max() - spark_act.min()
                    stat["fwhm"] = fwhm
                    stat["duration"] = len(spark_act)
                    stat["radius"] = r.extent
                    stat["abs_spark_row"] = np.abs(r.centroid[0] - row)
                    stat["abs_spark_col"] = np.abs(r.centroid[1] - col)
                    stat["rel_spark_row"] = r.centroid[0]
                    stat["rel_spark_col"] = r.centroid[1]
                    stat["onset_frame"] = r.coords[0, 0]
                    video_stats.append(stat)

            if len(video_stats):
                all_stats.append(video_stats)
                df = pd.DataFrame.from_dict(video_stats).groupby(["file_name", "experimental_condition", "experiment_name", "perturbation"]).mean().reset_index()
                cell_stats.append(df)

            if plot_videos:
                frames = utils.generate_overlay_video(fixed_mask, vid)
                anno_dir = file_name.replace(model_type, annotated_video_dir)
                utils.makedirs(os.path.sep.join(anno_dir.split(os.path.sep)[:-1]))
                out_path = anno_dir.replace(".npy", ".mp4")
                utils.create_video(frames[:num_frames], out_path)
    
    # Save spreadsheet
    pd.concat(cell_stats).to_csv(omni_csv_output_path)
    try:
       pd.DataFrame.from_dict(all_stats).to_parquet(event_csv_output_path.replace(".csv", ".parquet")  # Big file
    except:
        pd.DataFrame.from_dict(all_stats).to_csv(event_csv_output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 3D segmentation on calcium imaging data.")
    parser.add_argument('-c', '--config', default=None, type=str, help="Path to the project config .yaml file.")
    args = vars(parser.parse_args())
    main(**args)
