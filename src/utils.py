import re
import os
import av
import numpy as np
from PIL import Image
from accelerate import Accelerator
from functools import partial
from accelerate import InitProcessGroupKwargs
from datetime import timedelta
from tqdm import tqdm as std_tqdm
from timm.data import resolve_data_config
from matplotlib import pyplot as plt
from skimage.measure import label
from skimage.morphology import remove_small_objects as rso
from skimage.morphology import isotropic_dilation, binary_erosion
from skimage.measure import regionprops
from collections import Counter
from tqdm import tqdm
from src import ca_stats
import pandas as pd
import gc
import cv2
import yaml
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter
import torch
from glob import glob
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from scipy import optimize


def read_config(cfg_file):
    """
    Read a YAML config file and return its contents.

    Args:
        cfg_file (str): Path to the YAML config file.

    Returns:
        dict: Parsed contents of the config file.

    Raises:
        AssertionError: If no config file is provided.
    """
    assert cfg_file is not None, "No config file has been passed. Please read the README.MD file to learn more about configs."
    with open(cfg_file) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg


def makedirs(experiment_name, debug=False):
    """
    Recursively create directories for the given experiment name.

    Args:
        experiment_name (str): Path of the experiment directory to create.
        debug (bool, optional): If True, print created directories. Defaults to False.

    Returns:
        bool: True if directories were successfully created.
    """
    for i in range(1, len(experiment_name.split(os.path.sep)) + 1):
        path = os.path.sep.join(experiment_name.split(os.path.sep)[:i])
        if not len(path):
            continue

        os.makedirs(path, exist_ok=True)

    if debug:
        for i in range(1, len(experiment_name.split(os.path.sep)) + 1):
            print(os.path.sep.join(experiment_name.split(os.path.sep)[:i]))
    return True


def one_hot_encode(vector):
    """
    Convert a 1D vector of integer labels to a one-hot encoded matrix.

    Args:
        vector (numpy.ndarray): 1D array of integer labels.

    Returns:
        numpy.ndarray: 2D array of one-hot encoded labels.
            Shape: (len(vector), num_classes)

    Example:
        >>> one_hot_encode(np.array([0, 2, 1, 0]))
        array([[1., 0., 0.],
               [0., 0., 1.],
               [0., 1., 0.],
               [1., 0., 0.]])
    """
    # Get the number of classes (max value + 1)
    num_classes = np.max(vector) + 1

    # Create a zero matrix of shape (vector length, num_classes)
    one_hot = np.zeros((len(vector), num_classes))

    # Set the appropriate indices to 1
    one_hot[np.arange(len(vector)), vector] = 1

    return one_hot


def filter_whole_files(
        whole_files,
        test_y,
        scores,
        test_u_conds,
        wildcard="_ca",
        mask_files=None,
        fls=None,
        # remove_npy_pattern=None
        ):
    """
    Filter, process, and sort whole files based on existence, scores, and conditions.

    This function processes lists of whole files, their corresponding labels,
    scores, and unique conditions. It filters out non-existent files, adjusts
    labels based on conditions, and sorts the results based on scores.

    Args:
        whole_files (list): List of file paths for whole ROIs.
        test_y (list): List of labels corresponding to whole_files.
        scores (list): List of scores corresponding to whole_files.
        test_u_conds (numpy.ndarray): Array of unique conditions for testing.
        wildcard (str, optional): String to check for in file paths. Defaults to "_ca".

    Returns:
        tuple: A tuple containing three numpy arrays:
            - filt_whole_files (numpy.ndarray): Filtered and sorted whole file paths.
            - filt_y (numpy.ndarray): Filtered and sorted labels.
            - filt_scores (numpy.ndarray): Filtered and sorted scores.

    Note:
        - The function filters out non-existent files.
        - Labels are modified based on the presence of the wildcard in the file path.
        - Results are sorted based on the provided scores.
    """
    filt_whole_files, filt_masks, filt_test_files, filt_y, filt_scores = [], [], [], [], []
    intercept = len(test_u_conds)  # 4
    if mask_files is not None:
        for f, m, t, y, sc in zip(whole_files, mask_files, fls, test_y, scores):
            if os.path.exists(f) and os.path.exists(m):
                filt_whole_files.append(f)
                # m = "{}.npy".format(re.sub(remove_npy_pattern, '', m))
                filt_masks.append(m)
                f = f.split(os.path.sep)[2].split("-")[0].lower()
                y = np.where(test_u_conds == f)[0][0]
                modu = intercept if "_ca" in f else 0
                y += modu
                filt_y.append(y)
                filt_scores.append(sc)
                filt_test_files.append(t)
        filt_whole_files, filt_y, filt_scores = np.asarray(filt_whole_files), np.asarray(filt_y), np.asarray(filt_scores)
        filt_test_files = np.asarray(filt_test_files)
        filt_masks = np.asarray(filt_masks)
        filt_arg = np.argsort(filt_scores)  # [::-1]
        filt_whole_files = filt_whole_files[filt_arg]
        filt_y = filt_y[filt_arg]
        filt_test_files = filt_test_files[filt_arg]
        filt_masks = filt_masks[filt_arg]
        return filt_whole_files, filt_y, filt_test_files, filt_masks
    else:
        for f, y, sc in zip(whole_files, test_y, scores):
            if os.path.exists(f):
                filt_whole_files.append(f)
                f = f.split(os.path.sep)[2].split("-")[0].lower()
                y = np.where(test_u_conds == f)[0][0]
                modu = intercept if wildcard in f else 0
                y += modu
                filt_y.append(y)
                filt_scores.append(sc)
        filt_whole_files, filt_y, filt_scores = np.asarray(filt_whole_files), np.asarray(filt_y), np.asarray(filt_scores)
        filt_arg = np.argsort(filt_scores)
        filt_whole_files = filt_whole_files[filt_arg]
        filt_y = filt_y[filt_arg]
        return filt_whole_files, filt_y, filt_scores


def generate_overlay_video(object_volume, cellular_video, alpha=0.2):
    """
    Generate a video with colored object overlays on cellular video.
    
    Args:
        object_volume (numpy.ndarray): Array of shape (T, H, W) containing integer labels.
        cellular_video (numpy.ndarray): Array of shape (T, H, W, 3) containing RGB cellular video.
        alpha (float, optional): Transparency of the overlay (0.0 to 1.0). Defaults to 0.2.
    
    Returns:
        numpy.ndarray: Array of shape (T, H, W, 3) containing the blended video frames.
    """
    assert object_volume.shape == cellular_video.shape[:3], "Shapes of object_volume and cellular_video must match"

    T, H, W = object_volume.shape

    # Get unique object labels (excluding background)
    unique_labels = np.unique(object_volume)
    unique_labels = unique_labels[unique_labels != 0]

    # Generate colors for each unique label
    np.random.seed(0)
    cmap = plt.get_cmap("hsv")
    colors = np.asarray([cmap(i / len(unique_labels)) for i in range(len(unique_labels))])
    colors = colors[np.random.permutation(len(colors))]
    color_dict = {label: mcolors.rgb2hex(color[:3]) for label, color in zip(unique_labels, colors)}

    blends = []
    for t in range(T):
        # Get current frame and object mask
        cell_frame = cellular_video[t]
        cell_frame = 255 - (cell_frame[..., None].repeat(3, -1) * 255).astype(np.uint8)
        obj_mask = object_volume[t]

        # Create colored overlay
        overlay = np.zeros((H, W, 3), dtype=np.uint8)
        for label, color in color_dict.items():
            mask = obj_mask == label
            overlay[mask] = np.array(mcolors.hex2color(color)) * 255

        # Blend cellular frame with colored overlay
        blended = cv2.addWeighted(cell_frame, 1 - alpha, overlay, alpha, 0)
        blends.append(blended)
    return np.asarray(blends)


def gaussian_filter_3d(volume, sigma=1):
    """
    Apply a Gaussian filter to a 3D volume.

    Args:
        volume (numpy.ndarray): 3D array representing the volume (e.g., video).
        sigma (float, optional): Standard deviation for Gaussian kernel. Defaults to 1.

    Returns:
        numpy.ndarray: Filtered 3D volume.
    """
    return gaussian_filter(volume, sigma=sigma)


def pad_image_3d(x, hw):
    """
    Pad 3D images with zeros to be hw x hw pixels in the spatial dimensions.

    Args:
        x (numpy.ndarray): Input 3D image array.
        hw (int): Target height and width for padding.

    Returns:
        numpy.ndarray: Padded image array.
    """
    xshape = x.shape
    hdiff, wdiff = hw - xshape[1], hw - xshape[2]
    pad = ((0, 0), (0, hdiff), (0, wdiff), (0, 0))
    if not np.sum(pad):
        return x
    pad_x = np.pad(x, pad, mode='constant')
    return pad_x


def pad_image_2d(x, hw):
    """
    Pad 2D images with zeros to be hw x hw pixels.

    Args:
        x (numpy.ndarray): Input 2D image array.
        hw (int): Target height and width for padding.

    Returns:
        numpy.ndarray: Padded image array.
    """
    xshape = x.shape
    hdiff, wdiff = hw - xshape[0], hw - xshape[1]
    pad = ((0, hdiff), (0, wdiff), (0, 0))
    if not np.sum(pad):
        return x
    pad_x = np.pad(x, pad, mode='constant')
    return pad_x


def normalize_ca2(x, mu, std):
    """
    Normalize a numpy image or a list of images using the same routine nuclear seg models were trained with.

    Args:
        x (numpy.ndarray or list): Input image array or list of image arrays.
        mu (float): Mean value for normalization.
        std (float): Standard deviation value for normalization.

    Returns:
        numpy.ndarray or list: Normalized image array(s).
    """
    if isinstance(x, list):
        return [normalize_ca2(img, mu, std) for img in x]

    znorm_x = (x - x.mean()) / (x.std() + 1e-8)  # Add small epsilon to avoid division by zero
    mmnorm_x = (znorm_x - znorm_x.min()) / (znorm_x.max() - znorm_x.min() + 1e-8)
    zmmnorm_x = (mmnorm_x - mu) / std
    return zmmnorm_x


def prepare_env(timm_model, seconds=5400):
    """
    Set up environment variables for modeling and printing.

    Args:
        timm_model: TIMM model to use for data config resolution.
        seconds (int, optional): Timeout in seconds. Defaults to 5400.

    Returns:
        tuple: Accelerator, device, tqdm function, and TIMM data config.
    """
    process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=seconds))
    accelerator = Accelerator(kwargs_handlers=[process_group_kwargs])
    device = accelerator.device
    tqdm = partial(std_tqdm, dynamic_ncols=True)
    TIMM = resolve_data_config({}, model=timm_model)
    return accelerator, device, tqdm, TIMM


def get_activities(video, nuclei, cytoplasms, cytoplasms_far, nuclei_ids, tracking_strategy):
    """
    Mask video and store activities for nuclei and cytoplasms.

    Args:
        video (numpy.ndarray): Input video array.
        nuclei (numpy.ndarray): Nuclei mask array.
        cytoplasms (numpy.ndarray): Cytoplasms mask array.
        cytoplasms_far (numpy.ndarray): Far cytoplasms mask array.
        nuclei_ids (list): List of nuclei IDs.
        tracking_strategy (str): Strategy for tracking ('first' or 'all').

    Returns:
        tuple: Mean and standard deviation of activities for nuclei, cytoplasms, and far cytoplasms, along with activity arrays.
    """
    nuc_activity, cp_activity, cpf_activity = [], [], []
    for ni in nuclei_ids:
        if tracking_strategy == "first":
            nuc_activity.append(video[:, nuclei == ni].mean(1))
            cp_activity.append(video[:, cytoplasms == ni].mean(1))
            cpf_activity.append(video[:, cytoplasms_far == ni].mean(1))
        else:
            nuc_activity.append(np.asarray([v[n == ni].mean() for v, n in zip(video, nuclei)]))
            cp_activity.append(np.asarray([v[n == ni].mean() for v, n in zip(video, cytoplasms)]))
            cpf_activity.append(np.asarray([v[n == ni].mean() for v, n in zip(video, cytoplasms_far)]))
    nuc_activity = np.asarray(nuc_activity)
    cp_activity = np.asarray(cp_activity)
    cpf_activity = np.asarray(cpf_activity)
    mean_nuc = nuc_activity.mean(0)
    std_nuc = nuc_activity.std(0)
    mean_cp = cp_activity.mean(0)
    std_cp = cp_activity.std(0)
    mean_cpf = cpf_activity.mean(0)
    std_cpf = cpf_activity.std(0)
    return mean_nuc, std_nuc, mean_cp, std_cp, mean_cpf, std_cpf, nuc_activity, cp_activity, cpf_activity


def plot_lines(nucs, cps, alpha=0.5, output=None):
    """
    Plot each entry with a different color.

    Args:
        nucs (list): List of nucleus activity arrays.
        cps (list): List of cytoplasm activity arrays.
        alpha (float, optional): Alpha value for plot transparency. Defaults to 0.5.
        output (str, optional): Path to save the plot. If None, the plot is shown. Defaults to None.
    """
    lp = plt.get_cmap("Reds")
    cp = plt.get_cmap("Blues")
    nuc_cm = lp(np.linspace(0, 1, len(nucs)))
    cps_cm = cp(np.linspace(0, 1, len(cps)))

    f = plt.figure()
    for idx, (n, c) in enumerate(zip(nucs, nuc_cm)):
        if idx == 0:
            plt.plot(n, color=c, alpha=alpha, label="Nucleus")
        else:
            plt.plot(n, color=c, alpha=alpha)
    for idx, (cp, c) in enumerate(zip(cps, cps_cm)):
        if idx == 0:
            plt.plot(cp, color=c, alpha=alpha, label="Cytoplasm")
        else:
            plt.plot(cp, color=c, alpha=alpha)
    leg = plt.legend()
    for lh in leg.legend_handles:
        lh.set_alpha(1)

    try:
        if output is not None:
            dirs = os.path.sep.join(output.split(os.path.sep)[:-1])
            makedirs(dirs)
            plt.savefig(output)
        else:
            plt.show()
    finally:
        plt.close(f)


def create_video(frames, path, fps=24):
    """
    Create a video from frames and save it to the specified path.

    Args:
        frames (list): List of frames to create the video.
        path (str): Path to save the video.
        fps (int, optional): Frames per second for the video. Defaults to 24.
    """
    container = av.open(path, mode="w")

    stream = container.add_stream("h264", rate=fps, options={'b:a': '192000', 'maxrate': '192000', 'minrate': '192000'})
    stream.width = frames[0].shape[1]
    stream.height = frames[0].shape[0]
    stream.pix_fmt = "yuv420p"

    for img in frames:
        frame = av.VideoFrame.from_ndarray(img, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)

    # Flush stream
    for packet in stream.encode():
        container.mux(packet)

    # Close the file
    container.close()


def center_video(video, HW):
    """
    Zero pad a video so that it's centered.

    Args:
        video (numpy.ndarray): Input video array.
        HW (int): Target height and width for padding.

    Returns:
        numpy.ndarray: Padded video array.
    """
    hh = HW - video.shape[1]
    ww = HW - video.shape[2]
    padding = ((0, 0), (int(np.ceil(hh / 2)), hh // 2), (int(np.ceil(ww / 2)), ww // 2), (0, 0))
    padded_video = np.pad(video, padding)
    return padded_video


def center_image(image, HW):
    """
    Zero pad an image so that it's centered.

    Args:
        image (numpy.ndarray): Input image array.
        HW (int): Target height and width for padding.

    Returns:
        numpy.ndarray: Padded image array.
    """
    hh = HW - image.shape[0]
    ww = HW - image.shape[1]
    padding = ((int(np.ceil(hh / 2)), hh // 2), (int(np.ceil(ww / 2)), ww // 2))
    
    # Add padding for channel dimension if present
    if image.ndim == 3:
        padding += ((0, 0),)
    
    padded_image = np.pad(image, padding)
    return padded_image


def dff(x, eps, d0="all"):
    """
    dF/F normalize x.

    Args:
        x (numpy.ndarray): Input array to normalize.
        eps (float): Small value to avoid division by zero.
        d0 (str, optional): Normalization strategy ('first' or 'all'). Defaults to "all".

    Returns:
        numpy.ndarray: Normalized array.
    """
    dtype = x.dtype
    x = x.astype(np.float32)
    mask = x != 0
    mask_0 = x[0] != 0

    nonzeros = x[mask]
    if d0 == "first":
        # Normalizer is from t0
        min_nonzero = x[0][mask_0].min()
    elif d0 == "all":
        min_nonzero = nonzeros.min()
    else:
        raise RuntimeError("Normalize by first or all frames.")

    dff_nonzeros = (nonzeros - min_nonzero) / (min_nonzero + eps)
    x[mask] = dff_nonzeros

    # Stretch contrast, assuming uint8 was the encoding
    x = (x - x.min()) / (x.max() - x.min())
    x = x * 255

    # Convert back
    x = x.astype(dtype)
    return x


def nonzero_mean(x):
    """
    Calculate the mean of non-zero elements along specified axes.

    Args:
        x (numpy.ndarray): Input array.

    Returns:
        numpy.ndarray: Mean of non-zero elements.
    """
    sums = x.sum((1, 2, 3)).astype(np.float32)
    if np.any(sums == 0):
        return None
    denom = (x != 0).sum((1, 2, 3)).astype(np.float32)
    return sums / denom


def track(files, small_vol_thresh, tracking_strategy, debug=False, filter_weird=False, e_thresh=0.8):
    """
    Track nuclei using a fast approximation or a slower matching algorithm.

    Args:
        files (list): List of file paths for nucleus segments.
        small_vol_thresh (int): Threshold for removing small objects.
        tracking_strategy (str): Strategy for tracking ('first' or 'all').
        debug (bool, optional): If True, print debug information. Defaults to False.
        filter_weird (bool, optional): If True, filter out cells with weird shapes. Defaults to False.
        e_thresh (float, optional): Eccentricity threshold for filtering weird shapes. Defaults to 0.8.

    Returns:
        numpy.ndarray: Labeled nuclei array.
    """
    if tracking_strategy == "first":
        # Use the first frame's nucleus segments for all images
        mask = rso(np.asarray(Image.open(files[0])) > 0, small_vol_thresh)
        nuclei = label(mask)
        return nuclei
    elif tracking_strategy == "all":
        nuclei = []
        for idx, pred in enumerate(files):
            im = np.asarray(Image.open(pred))
            im = rso(binary_erosion(im > 0, np.ones((3, 3))), small_vol_thresh)
            it_lab = label(im)
            if idx == 0:
                rps = np.asarray(regionprops(it_lab))
                keep_idx = np.full(len(rps), fill_value=True)
                if filter_weird:
                    # Lets remove cells with weird shapes if requested
                    eccentricity = [x.eccentricity for x in rps]
                    mask = np.asarray(eccentricity) < e_thresh
                    keep_idx[~mask] = False
                rp_ids = get_id(rps, it_lab)
                proposal = it_lab
            else:
                im = im.astype(np.float32)
                proposal = np.zeros_like(im)
                for r_idx, (r, rid, keep) in enumerate(zip(rps, rp_ids, keep_idx)):
                    if rid == 0 or not keep:
                        continue
                    
                    # Get overlap of prev frame's rp with this frame
                    coords = np.asarray(r.coords).astype(int)
                    idn = it_lab[coords[:, 0], coords[:, 1]]
                    rep_id = Counter(idn).most_common()[0][0]
                    if np.sum(idn) == 0 or rep_id == 0:
                        keep_idx[r_idx] = False  # Kill this track
                    else:
                        proposal[it_lab == rep_id] = rid
            nuclei.append(proposal)
        nuclei = np.asarray(nuclei)

        # Do a final filter stage to remove nuclei that are no in every frame
        rem_ids = np.where(~keep_idx)[0]
        for r in rem_ids:
            rid = rp_ids[r]
            nuclei[nuclei == rid] = 0
        return nuclei
    else:
        raise NotImplementedError(tracking_strategy)


def generate_vids(image, idx):
    """
    Generate videos from image tensors.

    Args:
        image (torch.Tensor): Input image tensor.
        idx (int): Index for naming the output video file.

    Returns:
        None
    """
    image = image[idx].squeeze()
    image_vid = image.cpu().detach().squeeze()
    image_vid = (image_vid - image_vid.min()) / (image_vid.max() - image_vid.min())
    image_vid = image_vid.numpy()
    image_vid = (255 * image_vid).astype(np.uint8)
    create_video(image_vid[..., None].repeat(3, -1), "group_{}.mp4".format(idx))


def generate_grads_vids(image, grad, idx=0):
    """
    Generate videos from image tensors and their gradients.

    Args:
        image (torch.Tensor): Input image tensor.
        grad (torch.Tensor): Gradient tensor.
        idx (int, optional): Index for selecting the image and gradient. Defaults to 0.

    Returns:
        None
    """
    image_vid = image[idx]
    image_vid = (image_vid - image_vid.min()) / (image_vid.max() - image_vid.min())
    image_vid = (255 * image_vid).astype(np.uint8)
    grad = grad[idx]

    combo = (image_vid.astype(np.float32) * grad).astype(np.uint8)
    grad = (grad * 255).astype(np.uint8)

    create_video(image_vid[..., None].repeat(3, -1), "temp_vid.mp4")
    create_video(grad[..., None].repeat(3, -1), "temp_grad.mp4")
    create_video(combo[..., None].repeat(3, -1), "temp_combo.mp4")


def get_id(rps, im):
    """
    Get the modal label for each regionprop.

    Args:
        rps (list): List of regionprops.
        im (numpy.ndarray): Input image array.

    Returns:
        numpy.ndarray: Array of modal labels for each regionprop.
    """
    out = []
    for r in rps:
        coords = np.asarray(r.coords).astype(int)
        idn = im[coords[:, 0], coords[:, 1]]
        fidn = idn[idn > 0]
        try:
            midn = Counter(fidn).most_common()[0][0]
            out.append(midn)
        except:
            pass
    return np.asarray(out)


def load_video(images):
    """
    Load a video frame-by-frame with PIL.

    Args:
        images (list): List of image file paths.

    Returns:
        numpy.ndarray: Array of loaded video frames.
    """
    return np.asarray([np.asarray(Image.open(x)) for x in images])


def get_cytoplasms(nuclei, dilation_radius, dilation_radius_far, tracking_strategy):
    """
    Return cytoplasms for each nucleus.

    Args:
        nuclei (numpy.ndarray): Nuclei mask array.
        dilation_radius (int): Radius for near cytoplasm dilation.
        dilation_radius_far (int): Radius for far cytoplasm dilation.
        tracking_strategy (str): Strategy for tracking ('first' or 'all').

    Returns:
        tuple: Nuclei IDs, near cytoplasms mask, and far cytoplasms mask.
    """
    if tracking_strategy == "all":
        nuclei_ids = np.unique(nuclei[0])
    else:
        nuclei_ids = np.unique(nuclei)
    nuclei_ids = nuclei_ids[nuclei_ids > 0]  # Remove background

    if tracking_strategy == "all":
        cytoplasms = np.zeros_like(nuclei)
        cytoplasms_far = np.zeros_like(nuclei)
        for i in range(len(cytoplasms)):
            for ni in nuclei_ids:
                # Near cyto
                mask = nuclei[i] == ni
                ctp = isotropic_dilation(mask, dilation_radius)
                ctp[mask] = 0
                cytoplasms[i, ctp > 0] = ni

                # Far cyto
                ctp_far = isotropic_dilation(ctp, dilation_radius_far)
                ctp_far[ctp] = 0
                cytoplasms_far[i, ctp_far > 0] = ni
    else:
        cytoplasms = np.zeros_like(nuclei)
        cytoplasms_far = np.zeros_like(nuclei)
        for ni in nuclei_ids:
            # Near cyto
            mask = nuclei == ni
            ctp = isotropic_dilation(mask, dilation_radius)
            ctp[mask] = 0
            cytoplasms[ctp > 0] = ni

            # Far cyto
            ctp_far = isotropic_dilation(ctp, dilation_radius_far)
            ctp_far[np.logical_or(ctp, mask)] = 0
            cytoplasms_far[ctp_far > 0] = ni

    return nuclei_ids, cytoplasms, cytoplasms_far


def colored_video(nuclei, output):
    """
    Create a colored video from nuclei masks.

    Args:
        nuclei (numpy.ndarray): Nuclei mask array.
        output (str): Path to save the colored video.
    """
    vnuclei = nuclei.astype(np.float32)
    vnuclei = (vnuclei - vnuclei.min()) / (vnuclei.max() - vnuclei.min())
    vnuclei = (vnuclei * 255).astype(np.uint8)
    cvnuclei = []
    for frame in vnuclei:
        colored = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
        colored[colored.sum(-1) == 128, :] = 128
        cvnuclei.append(colored)
    vnuclei = np.asarray(cvnuclei)
    create_video(vnuclei, output)


def prepare_data(model_type, whole_roi_dir, test_exps, use_whole_files=False, filter="_postca"):
    """
    Prepare data for model training and testing.

    This function processes files from a given model type directory, separates them into
    training and testing sets, and prepares labels for each set. It also handles whole ROI
    files for testing.

    Args:
        model_type (str): The directory path containing the model type data.
        whole_roi_dir (str): The directory path containing whole ROI data.
        test_exps (list): A list of experiment names to be used for testing.

    Returns:
        tuple: A tuple containing:
            - new_test_files (list): List of file paths for testing.
            - new_test_ys (list): List of labels for testing files.
            - train_y (numpy.ndarray): Labels for training files.
            - test_u_conds (numpy.ndarray): Unique conditions for testing files.

    """
    if use_whole_files:
        model_type = whole_roi_dir
    files = glob(os.path.join(model_type, "*", "*", "*.npy"))
    assert len(files), "Is {} an actual folder?".format(model_type)
    files = [x for x in files if filter not in x]

    train_files, test_files = [], []
    for f in files:
        for filt in test_exps:
            if filt in f:
                test_files.append(f)
                continue
        train_files.append(f)

    train_u_conds, train_y = np.unique([x.split(os.path.sep)[2].split("-")[0].lower() for x in train_files], return_inverse=True)
    train_u_exps, train_y_exps = np.unique([True if "_ca" in x else False for x in train_files], return_inverse=True)
    train_y_exps[train_y_exps == 1] = train_y.max() + 1
    train_y = train_y + train_y_exps

    test_u_conds, test_y = np.unique([x.split(os.path.sep)[2].split("-")[0].lower() for x in test_files], return_inverse=True)
    test_u_exps, test_y_exps = np.unique([True if "_ca" in x else False for x in test_files], return_inverse=True)
    test_y_exps[test_y_exps == 1] = test_y.max() + 1
    test_y = test_y + test_y_exps

    test_y = torch.from_numpy(test_y)
    whole_files = []
    new_test_files = []
    new_test_ys = []
    for x, y in zip(test_files, test_y):
        if os.path.exists(x.replace(model_type, whole_roi_dir)):
            whole_files.append(x.replace(model_type, whole_roi_dir))
            new_test_files.append(x)
            new_test_ys.append(y)
    whole_files = np.asarray(whole_files)
    new_test_files = np.asarray(new_test_files)
    new_test_ys = np.asarray(new_test_ys)

    return new_test_files, new_test_ys, train_y, test_u_conds, whole_files, new_test_ys


def seq_generate_roi_data(dr, nuclei, cytoplasms, cytoplasms_far, video, detrend_video, nuclei_ids, hroi, roi_size, directories, normalize_rois, use_detrended, use_dff, debug, image_dir, plot_figures, save_npy_rois, eps=1e-7):
    """
    Run stats and extract videos for each nucleus and near/far cytoplasm.

    Args:
        dr (str): Directory path for the experiment.
        nuclei (numpy.ndarray): Nuclei mask array.
        cytoplasms (numpy.ndarray): Cytoplasms mask array.
        cytoplasms_far (numpy.ndarray): Far cytoplasms mask array.
        video (numpy.ndarray): Input video array.
        detrend_video (numpy.ndarray): Detrended video array.
        nuclei_ids (list): List of nuclei IDs.
        hroi (int): Half of the region of interest size.
        roi_size (int): Size of the region of interest.
        directories (dict): Dictionary of directory paths for saving data.
        normalize_rois (bool): Whether to normalize ROIs.
        use_detrended (bool): Whether to use detrended video.
        use_dff (bool): Whether to use dF/F normalization.
        debug (bool): Whether to print debug information.
        plot_figures (bool): Whether to plot figures.
        version (str): Version identifier for saving data.
        save_npy_rois (bool): Whether to save ROIs as .npy files.
        eps (float, optional): Small value to avoid division by zero. Defaults to 1e-7.
        fast_plot (bool, optional): Whether to use fast plotting. Defaults to True.

    Returns:
        pandas.DataFrame: DataFrame containing the extracted data.
    """
    nucleus_roi_dir = directories["nucleus_roi_dir"]
    nucleus_mask_roi_dir = directories["nucleus_mask_roi_dir"]
    cyto_roi_dir = directories["cyto_roi_dir"]
    cyto_far_roi_dir = directories["cyto_far_roi_dir"]
    cyto_roi_spl_dir = directories["cyto_roi_spl_dir"]
    cyto_far_roi_spl_dir = directories["cyto_far_roi_spl_dir"]
    nucleus_roi_plot_dir = directories["nucleus_roi_plot_dir"]
    cyto_roi_plot_dir = directories["cyto_roi_plot_dir"]
    cyto_far_roi_plot_dir = directories["cyto_far_roi_plot_dir"]
    whole_cell_roi_spl_dir = directories["whole_cell_roi_spl_dir"]

    spreadsheet = []
    nuc_traces, cp_traces, cpf_traces = [], [], []
    if len(nuclei.shape) == 2:
        centroids = regionprops(nuclei)
        all_frames = False
    else:
        centroids = []
        for n in nuclei:
            centroids.append(regionprops(n))
        all_frames = True
    for idx, (cn, ni) in tqdm(enumerate(zip(centroids, nuclei_ids)), total=len(centroids)):
        # Handle first vs. all frame segdifferently
        if all_frames:
            nuc_vid, cp_vid, cpf_vid = [], [], []
            for f, (cn, nuclei_i, cytoplasms_i, cytoplasms_far_i) in enumerate(zip(centroids, nuclei, cytoplasms, cytoplasms_far)):  # Frames
                nmask = nuclei_i == ni
                cmask = cytoplasms_i == ni
                cfmask = cytoplasms_far_i == ni
                centroid = cn.centroid
                bbox = np.asarray([centroid[0] - hroi, centroid[0] + hroi, centroid[1] - hroi, centroid[1] + hroi]).astype(int)
                bbox = np.maximum(bbox, 0)
                if use_detrended:
                    vid_roi = detrend_video[f, bbox[0]: bbox[1], bbox[2]: bbox[3]][..., None]
                else:
                    vid_roi = video[f, bbox[0]: bbox[1], bbox[2]: bbox[3]][..., None]
                nuc_mask = nmask[bbox[0]: bbox[1], bbox[2]: bbox[3]][..., None]
                cp_mask = cmask[bbox[0]: bbox[1], bbox[2]: bbox[3]][..., None]
                cpf_mask = cfmask[bbox[0]: bbox[1], bbox[2]: bbox[3]][..., None]
                f_nuc_vid = vid_roi * nuc_mask
                f_cp_vid = vid_roi * cp_mask
                f_cpf_vid = vid_roi * cpf_mask
                nuc_vid.append(f_nuc_vid)
                cp_vid.append(f_cp_vid)
                cpf_vid.append(f_cpf_vid)
            nuc_vid = np.stack(nuc_vid, axis=0)
            cp_vid = np.stack(cp_vid, axis=0)
            cpf_vid = np.stack(cpf_vid, axis=0)
        else:
            nmask = nuclei == ni
            cmask = cytoplasms == ni
            cfmask = cytoplasms_far == ni
            centroid = cn.centroid
            bbox = np.asarray([centroid[0] - hroi, centroid[0] + hroi, centroid[1] - hroi, centroid[1] + hroi]).astype(int)
            bbox = np.maximum(bbox, 0)
            if use_detrended:
                vid_roi = detrend_video[:, bbox[0]: bbox[1], bbox[2]: bbox[3]][..., None]
            else:
                vid_roi = video[:, bbox[0]: bbox[1], bbox[2]: bbox[3]][..., None]

            nuc_mask = nmask[bbox[0]: bbox[1], bbox[2]: bbox[3]][None, ..., None]
            cp_mask = cmask[bbox[0]: bbox[1], bbox[2]: bbox[3]][None, ..., None]
            cpf_mask = cfmask[bbox[0]: bbox[1], bbox[2]: bbox[3]][None, ..., None]
            nuc_vid = vid_roi * nuc_mask
            cp_vid = vid_roi * cp_mask
            cpf_vid = vid_roi * cpf_mask

        nuc_vid[nuc_vid == 0] = nuc_vid.min()
        cp_vid[cp_vid == 0] = cp_vid.min()
        cpf_vid[cpf_vid == 0] = cpf_vid.min()

        if normalize_rois:
            nuc_vid = (((nuc_vid - nuc_vid.min()) / (nuc_vid.max() - nuc_vid.min())) * 255.).astype(np.uint8)
            cp_vid = (((cp_vid - cp_vid.min()) / (cp_vid.max() - cp_vid.min())) * 255.).astype(np.uint8)
            cpf_vid = (((cpf_vid - cpf_vid.min()) / (cpf_vid.max() - cpf_vid.min())) * 255.).astype(np.uint8)

        if debug:
            f = plt.figure()
            plt.subplot(121);plt.plot(nonzero_mean(nuc_vid), label="Nucleus");plt.plot(nonzero_mean(cp_vid), label="near");plt.plot(nonzero_mean(cpf_vid), label="far");plt.legend()

        if use_dff:
            nuc_vid = dff(nuc_vid, eps)
            cp_vid = dff(cp_vid, eps)
            cpf_vid = dff(cpf_vid, eps)

        if debug:
            plt.subplot(122);plt.plot(nonzero_mean(nuc_vid), label="Nucleus");plt.plot(nonzero_mean(cp_vid), label="near");plt.plot(nonzero_mean(cpf_vid), label="far");plt.legend()
            try:
                plt.show()
            finally:
                plt.close(f)

        nucd = dr.replace(image_dir, nucleus_roi_dir)
        nucmd = dr.replace(image_dir, nucleus_mask_roi_dir)
        cpd = dr.replace(image_dir, cyto_roi_dir)
        cpfd = dr.replace(image_dir, cyto_far_roi_dir)
        spl_cpd = dr.replace(image_dir, cyto_roi_spl_dir)
        spl_cpfd = dr.replace(image_dir, cyto_far_roi_spl_dir)
        spl_whole = dr.replace(image_dir, whole_cell_roi_spl_dir)
        plot_nucd = dr.replace(image_dir, nucleus_roi_plot_dir)
        plot_cpd = dr.replace(image_dir, cyto_roi_plot_dir)
        plot_cpfd = dr.replace(image_dir, cyto_far_roi_plot_dir)

        if idx == 0:
            # Make directories
            makedirs(os.path.sep.join(nucd.split(os.path.sep)[:-1]))
            makedirs(os.path.sep.join(nucmd.split(os.path.sep)[:-1]))
            makedirs(os.path.sep.join(cpd.split(os.path.sep)[:-1]))
            makedirs(os.path.sep.join(cpfd.split(os.path.sep)[:-1]))
            makedirs(os.path.sep.join(spl_cpd.split(os.path.sep)[:-1]))
            makedirs(os.path.sep.join(spl_cpfd.split(os.path.sep)[:-1]))
            makedirs(os.path.sep.join(spl_whole.split(os.path.sep)[:-1]))
            makedirs(os.path.sep.join(plot_nucd.split(os.path.sep)[:-1]))
            makedirs(os.path.sep.join(plot_cpd.split(os.path.sep)[:-1]))
            makedirs(os.path.sep.join(plot_cpfd.split(os.path.sep)[:-1]))

        # Run and save statistics on each trace
        if plot_figures:
            nuc_stats, _, _, nuc_trace = ca_stats.compute_stats(nuc_vid, show_plot=False, output="{}_{}.png".format(plot_nucd, idx), frame=video[0], mask=nmask)
            cp_stats, cp_splits, _, cp_trace = ca_stats.compute_stats(cp_vid, show_plot=False, output="{}_{}.png".format(plot_cpd, idx), frame=video[0], mask=cmask)
            cpf_stats, cpf_splits, _, cpf_trace = ca_stats.compute_stats(cpf_vid, show_plot=False, output="{}_{}.png".format(plot_cpfd, idx), frame=video[0], mask=cfmask)
            nuc_traces.append(nuc_trace)
            cp_traces.append(cp_trace)
            cpf_traces.append(cpf_trace)
        else:
            nuc_stats, _, _, _ = ca_stats.compute_stats(nuc_vid, show_plot=False, frame=video[0], mask=nmask)
            cp_stats, cp_splits, _,  _ = ca_stats.compute_stats(cp_vid, show_plot=False, frame=video[0], mask=cmask)
            cpf_stats, cpf_splits, _, _ = ca_stats.compute_stats(cpf_vid, show_plot=False, frame=video[0], mask=cfmask)

        df = pd.concat((pd.DataFrame.from_dict(nuc_stats), pd.DataFrame.from_dict(cp_stats), pd.DataFrame.from_dict(cpf_stats)))
        df["experiment"] = dr
        df["cell"] = idx
        df["region"] = np.arange(len(df))
        spreadsheet.append(df)

        if save_npy_rois:
            # Prepare full video data for modeling
            r, c = centroid
            r, c = int(r), int(c)
            nuc_vid = center_video(nuc_vid, roi_size)
            cp_vid = center_video(cp_vid, roi_size)
            cpf_vid = center_video(cpf_vid, roi_size)
            vid_roi = center_video(vid_roi, roi_size)
            np.save("{}_{}_rc_{}_{}".format(nucd, idx, r, c), nuc_vid)
            np.save("{}_{}_rc_{}_{}".format(cpd, idx, r, c), cp_vid)
            np.save("{}_{}_rc_{}_{}".format(cpfd, idx, r, c), cpf_vid)

            # Also save nuclear mask for later (spark detection)
            nuc_mask = center_image(nuc_mask.squeeze(), roi_size)
            np.save("{}_{}".format(nucmd, idx), nuc_mask)

            # Prepare truncated video data for modeling
            if cp_splits is not None:
                for i, spl in enumerate(cp_splits):
                    np.save("{}_{}__rc_{}_{}_num_{}".format(spl_cpd, idx, r, c, i), cp_vid[spl])
            if cpf_splits is not None:
                for i, spl in enumerate(cpf_splits):
                    np.save("{}_{}__rc_{}_{}_num_{}".format(spl_cpfd, idx, r, c, i), cpf_vid[spl])
                    np.save("{}_{}__rc_{}_{}_num_{}".format(spl_whole, idx, r, c, i), vid_roi[spl])
        gc.collect()
    df = pd.concat(spreadsheet)
    return df

def calculate_fwhm(y):
    """
    Calculate the Full Width at Half Maximum (FWHM) of a 1D signal.
    
    Args:
    y (array-like): The y-coordinates of the signal
    
    Returns:
    float: The FWHM of the signal
    """
    # Find the peak
    x = np.arange(len(y))
    peaks, _ = find_peaks(y)
    if len(peaks) == 0:
        return 0
    
    peak_index = peaks[np.argmax(y[peaks])]
    peak_value = y[peak_index]
    
    # Calculate half maximum
    half_max = peak_value / 2
    
    # Create interpolation function
    interp_func = interp1d(x, y - half_max, kind='cubic')
    
    # Find roots (where signal crosses half maximum)
    roots = []
    for i in range(len(x) - 1):
        if (interp_func(x[i]) * interp_func(x[i+1])) <= 0:
            root = optimize.brentq(interp_func, x[i], x[i+1])
            roots.append(root)
    
    if len(roots) < 2:
        return 0
    
    # Calculate FWHM
    fwhm = roots[-1] - roots[0]
    
    return fwhm

