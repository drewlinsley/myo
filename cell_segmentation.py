import os 
import numpy as np
import torch
from glob import glob
import segmentation_models_pytorch_3d as smp
from segmentation_models_pytorch_3d.decoders.unet import Unet
from matplotlib import pyplot as plt
from skimage import io
from src import utils
import argparse


def main(
        config,
        debug=False,
        timm_model="resnext50_32x4d",
        HW=384,
        timesteps=32,
        time_kernel=[32, 1, 1],
        score_thresh=0.2,
        stride=10,
        compile=True  # Pytorch2.0 to increase speed
        ):
    """
    Test 3D segmentation model on calcium imaging data.

    Args:
        config (str): Path to the configuration file.
        debug (bool, optional): Enable debug mode. Defaults to False.
        timm_model (str, optional): Name of the TIMM model to use. Defaults to "resnext50_32x4d".
        HW (int, optional): Height and width of the input patches. Defaults to 384.
        timesteps (int, optional): Number of timesteps to consider. Defaults to 32.
        time_kernel (list, optional): Kernel size for temporal dimension. Defaults to [32, 1, 1].
        score_thresh (float, optional): Threshold for segmentation scores. Defaults to 0.2.

    Returns:
        None
    """
    cfg = utils.read_config(config)
    ddir = cfg["ca2_data"]["processed_npys"]
    load_ckpt = cfg["segmentation_model"]["ckpt"]
    file_dir = ddir.split(os.path.sep)[-1]
    output_images = cfg["segmentation_model"]["segments"]

    experiments = glob(os.path.join(ddir, "*", "*", "*.npy"))

    # Prepare environment
    accelerator, device, tqdm, TIMM = utils.prepare_env(timm_model)
    mu = np.asarray(TIMM["mean"][0])[None, None, None, None]
    std = np.asarray(TIMM["std"][0])[None, None, None, None]

    print("Preparing models")
    model = smp.Unet(
        encoder_name=timm_model,
        in_channels=1,
        classes=2,
        time_kernel=time_kernel,
    )
    print("Using ckpt: {}".format(load_ckpt))
    ckpt_weights = torch.load(load_ckpt, map_location=torch.device('cpu'))
    key_check = [x for x in ckpt_weights.keys()][0]
    if key_check.split(".")[0] == "module":
        ckpt_weights = {k.replace("module.", ""): v for k, v in ckpt_weights.items()}
    model.load_state_dict(ckpt_weights)
    if compile:
        model = torch.compile(model)
    model = accelerator.prepare(model)
    model.eval()

    stride_HW = HW - stride  # Overlap
    with torch.no_grad():
        for experiment in tqdm(experiments, total=len(experiments), desc="Processing"):
            experiment_data = np.load(experiment).astype(np.float32)[..., None]
            experiment_name = experiment.replace(".nd2.npy", "")
            experiment_name = "{}{}".format(output_images, experiment_name.split(file_dir)[-1])
            utils.makedirs(experiment_name, debug=debug)
            frame_range = np.arange(timesteps, len(experiment_data))  # Indices for time
            for idx in frame_range:
                images = experiment_data[idx - timesteps: idx]
                image = images[-1].squeeze()  # Image we are decoding
                pred_image = np.zeros_like(image)
                hs, ws = np.arange(0, image.shape[0], stride_HW), np.arange(0, image.shape[1], stride_HW)
                preps, hws, pre_shapes = [], [], []
                for h in hs:
                    for w in ws:
                        timage = images[:, h: h + HW, w: w + HW]
                        norm_timage = utils.normalize_ca2(timage, mu, std)
                        pre_shape = norm_timage.shape[1:]
                        padded_timage = utils.pad_image_3d(norm_timage, HW)
                        prep_image = torch.from_numpy(padded_timage.squeeze()[None, None]).float().to(device)
                        preps.append(prep_image)
                        hws.append((h, w))
                        pre_shapes.append(pre_shape)

                # Batch the image up and run in one shot
                preps = torch.concat(preps)
                preds = model(preps)
                for p, (h, w), pre_shape in zip(preds, hws, pre_shapes):
                    p = p[..., :pre_shape[0], :pre_shape[1]].squeeze().softmax(0)[1].cpu().numpy()
                    pred_image[h: h + pre_shape[0], w: w + pre_shape[1]] = p
                pred_image[pred_image < score_thresh] = 0
                pred_image = (pred_image * 255).astype(np.uint8)
                image = image.astype(np.uint8)

                # Note that we are segmenting each frame separately
                if debug:
                    plt.subplot(121);plt.imshow(image);plt.subplot(122);plt.imshow(pred_image);plt.show()
                io.imsave(os.path.join(experiment_name, "image_{}.png".format(idx)), image, check_contrast=False)
                io.imsave(os.path.join(experiment_name, "pred_{}.png".format(idx)), pred_image, check_contrast=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 3D segmentation on calcium imaging data.")
    parser.add_argument('-c', '--config', default=None, type=str, help="Path to the project config .yaml file.")
    args = vars(parser.parse_args())
    main(**args)

