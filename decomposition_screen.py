import os
import argparse
import numpy as np
import torch
import src.transforms as video_transforms
from torchvision.transforms import v2 as transforms
from segmentation_models_pytorch_3d.decoders.classifier import Unet
from src.dataloaders import Classification_test
from src import utils
from matplotlib import pyplot as plt
from sklearn.decomposition import NMF
from sklearn.cross_decomposition import PLSRegression
import seaborn as sns
import pandas as pd
from scipy.spatial.distance import cdist
import joblib
from configs.roi_experiments import directories
from src import utils
from configs.roi_experiments import test_exps


def main(
        config,
        debug_videos=False,
        te_bs=32,
        timesteps=32,
        test_workers=4,
        timm_model="resnext50_32x4d",
        cls_dataset=1000,  # Number of +/- instances for a next-stage classification model
        HW=128,
        alpha_W=1e-2,
        DIMS="BCDHW"):
    """
    Analyze and visualize calcium spark data using dimensionality reduction techniques.

    This function loads a pre-trained DNN, processes test data, applies dimensionality
    reduction (NMF and PLS), and generates plots to visualize the results of drug screening
    experiments.

    We use NMF to score cells for their sparks and PLS (which is a differentiable model) to annotate sparks.

    Args:
        config (str): Path to the configuration file.
        debug_videos (bool, optional): If True, generates debug videos. Defaults to False.
        te_bs (int, optional): Test batch size. Defaults to 32.
        timesteps (int, optional): Number of timesteps in the input data. Defaults to 32.
        test_workers (int, optional): Number of worker processes for data loading. Defaults to 4.
        timm_model (str, optional): Name of the TIMM model to use. Defaults to "resnext50_32x4d".
        HW (int, optional): Height and width of the input images. Defaults to 128.
        DIMS (str, optional): Dimension order of the input data. Defaults to "BCDHW".

    Returns:
        None

    Saves:
        - Numpy arrays: stats, files, and labels for the processed data.
        - Joblib files: NMF and PLS dimensionality reduction models.
        - Numpy arrays: NMF scores and arguments.
        - PNG files: Control plot, drug screen plot, and normalized drug screen plot.

    The function performs the following main steps:
    1. Loads and prepares the test data.
    2. Loads a pre-trained model and extracts latent features.
    3. Applies NMF and PLS dimensionality reduction techniques.
    4. Generates and saves visualization plots for drug screening results.
    """
    cfg = utils.read_config(config)
    load_ckpt = cfg["sparks"]["ckpt"]
    control_plot = cfg["sparks"]["drug_screen_plot"]
    drug_screen_plot = cfg["sparks"]["drug_screen_plot"]
    drug_screen_normalized_plot = cfg["sparks"]["drug_screen_normalized_plot"]
    model_type = cfg["sparks"]["model_type"]
    spark_cls_data = cfg["sparks"]["spark_classifier_file"]
    regenerate_nmf = cfg["sparks"]["regenerate_nmf"]

    whole_roi_dir = directories["whole_cell_roi_spl_dir"]
    model_asset_dir = directories["model_asset_dir"]
    os.makedirs(model_asset_dir, exist_ok=True)
    stats_path = os.path.join(model_asset_dir, "stats_{}.npy".format(model_type))
    fl_path = os.path.join(model_asset_dir, "files_{}.npy".format(model_type))
    sc_path = os.path.join(model_asset_dir, "scores_{}.npy".format(model_type))
    label_path = os.path.join(model_asset_dir, "labels_{}.npy".format(model_type))
    pls_model_path = os.path.join(model_asset_dir, cfg["sparks"]["pls_file"])
    nmf_model_path = os.path.join(model_asset_dir, cfg["sparks"]["nmf_file"])

    # Here we additionally normalize them to the pretrained weights of our model
    time_kernel = [timesteps, 1, 1]
    pre_HW = HW + 32  # size to random crop from
    accelerator, _, tqdm, TIMM = utils.prepare_env(timm_model)
    test_trans = transforms.Compose([
        video_transforms.CenterCrop(HW),
        video_transforms.ToTensor(),
    ])

    # Select files for training and testing
    with accelerator.main_process_first():
        _, _, train_y, _, whole_files, whole_y = utils.prepare_data(model_type, whole_roi_dir, test_exps, use_whole_files=True)
        test_dataset = Classification_test(  # Run classification model on whole cell movies
            whole_files,
            whole_y,
            mu=TIMM["mean"],
            std=TIMM["std"],
            pre_HW=pre_HW,
            transform=test_trans,
            time=timesteps,
            dims=DIMS)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=te_bs, shuffle=False, drop_last=True, num_workers=test_workers)
        print("Finished loading data")

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
    model, test_loader = accelerator.prepare(model, test_loader)
    model.eval()

    print("Building dataloaders for latents")
    stats, rlabels, fls = [], [], []
    with torch.no_grad():
        for source in tqdm(test_loader, total=len(test_loader)):
            image, label, fl = source
            _, _, latents  = model(image, return_features=True)
            latents = latents[:, :, :latents.shape[2] // 2].cpu().numpy()
            latents = latents.mean((2, 3, 4))
            stats.append(latents)
            rlabels.append(label.cpu().numpy())
            fls.append(fl)
    stats = np.concatenate(stats)
    rlabels = np.concatenate(rlabels)
    fls = np.concatenate(fls)
    np.save(stats_path, stats)
    np.save(fl_path, fls)
    np.save(label_path, rlabels)

    # Generate a PLS and NMF model. The latter is better for sorting, the former is for annotation.
    nmf_arg_path = os.path.join(model_asset_dir, "nmf_score_arg_model_{}.npy".format(model_type))
    if regenerate_nmf:
        # Non-deterministic, allow saving.
        nmf_clfd = NMF(n_components=128, max_iter=10000, alpha_W=alpha_W).fit(stats)
        joblib.dump(nmf_clfd, nmf_model_path)
    else:
        nmf_clfd = joblib.load(nmf_model_path)
    nmf_scores = nmf_clfd.transform(stats)
    labs = rlabels.reshape(1, -1).astype(np.float32)
    labs = (labs - labs.min()) / (labs.max() - labs.min())
    dims = cdist(nmf_scores.T, labs)
    spark_dim = np.argmin(dims.squeeze())
    nmf_scores = nmf_scores[:, spark_dim]
    np.save(nmf_arg_path, spark_dim)
    np.save(sc_path, nmf_scores)

    # PLS for unsupervised spark annotation
    pls_score_path = os.path.join(model_asset_dir, "pls_dim_reduction_scores_{}.joblib".format(model_type))
    stats = (stats - stats.mean(0)) / stats.std(0)
    labs = utils.one_hot_encode(rlabels.ravel().astype(int))
    pls_clfd = PLSRegression(n_components=1, scale=True).fit(stats, y=labs)
    joblib.dump(pls_clfd, pls_model_path)
    pls_scores = pls_clfd.transform(stats).squeeze()
    np.save(pls_score_path, pls_scores)
    
    # DEBUG What is the direction/ID of the PCs?
    if debug_videos:
        with torch.no_grad():
            for source in tqdm(test_dataset, total=len(test_dataset)):
                image, label, fl = source
                # import pdb;pdb.set_trace()  # Debug here and look at videos to verify enrichment/not of sparks
                utils.generate_vids(image, idx=0)

    # Save file paths for training a spark classifier model
    sort_idx = np.argsort(nmf_scores)[::-1]  # Reverse order because we select dim according to distance to labels
    sort_files = whole_files[sort_idx]
    cls_files = np.concatenate((sort_files[:cls_dataset], sort_files[-cls_dataset:]))
    cls_labels = np.concatenate((np.zeros(cls_dataset), np.ones(cls_dataset)))
    np.savez(spark_cls_data, files=cls_files, labels=cls_labels)

    # Generate plots
    conditions = [x.split(os.path.sep)[1].split(" ")[1].lower() for x in fls]
    exps = [f.split(os.path.sep)[-2].lower().split("-")[0] for f in fls]
    hexps, labels = {}, []
    for cond, exp, f, score in zip(conditions, exps, fls, nmf_scores):
        if "_ca+" in f.lower():
            manip = "ca"
        elif "_post" in f.lower():
            manip = "post"
        elif "baseline" in f.lower():
            manip = "baseline"
        elif "el20" in f.split(os.path.sep)[-1].lower():
            manip = "baseline"
        else:
            raise RuntimeError(f)

        labels.append("{}_{}_{}".format(cond, exp, manip))

        if cond not in hexps:
            hexps[cond] = {}

        if exp not in hexps[cond]:
            hexps[cond][exp] = {}

        if manip not in hexps[cond][exp]:
            hexps[cond][exp][manip] = []

        hexps[cond][exp][manip].append(score)

    # Create control plot
    f, axs = plt.subplots(1, len(hexps))
    for (k, v), ax in zip(hexps.items(), axs):
        idf = np.stack((np.concatenate([[k] * len(v) for k, v in v["control"].items()]), np.concatenate([x for x in v["control"].values()])), 1)
        idf = pd.DataFrame(idf, columns=["Condition", "Spark score"])
        idf["Spark score"] = pd.to_numeric(idf["Spark score"])
        idf = idf[idf["Condition"] != "post"]  # Dont plot post
        idf["Condition"] = pd.Categorical(idf["Condition"], categories=["baseline", "ca"], ordered=True)
        sns.pointplot(data=idf, x="Condition", y="Spark score", ax=ax, estimator=np.mean)
        ax.set_title(k)
    plt.savefig(control_plot)
    plt.show()

    # Create control-normalized experimental plot
    res, cond, cis = [], [], []
    iterations = 1000
    for (k, v) in hexps.items():
        exp_baseline = np.mean(v[k]["baseline"])
        control_baseline = np.mean(v["control"]["baseline"])
        res.append(control_baseline - exp_baseline)
        cond.append(k)

        bcis = []
        for i in np.arange(iterations):
            exp_baseline = np.mean(np.asarray(v[k]["baseline"])[np.random.randint(low=0, high=len(v[k]["baseline"]), size=len(v[k]["baseline"]))])
            control_baseline = np.mean(np.asarray(v["control"]["baseline"])[np.random.randint(low=0, high=len(v["control"]["baseline"]), size=len(v["control"]["baseline"]))])
            bcis.append(control_baseline - exp_baseline)
        bcis = np.asarray(bcis)
        upper = np.percentile(bcis, 80)
        lower = np.percentile(bcis, 20)
        cis.append(upper - lower)

    f, ax = plt.subplots(1, 1)
    ax.scatter(np.arange(len(res)), res)
    ax.errorbar(np.arange(len(res)), res, yerr=cis, ls='none', capsize=3)
    ax.set_ylabel("Spark reduction score")  #  (> 0 is better than control)")
    ax.set_xticks(np.arange(len(cond)))
    ax.set_xticklabels(cond)
    plt.suptitle("control_baseline - exp_baseline")
    plt.savefig(drug_screen_plot)
    plt.show()

    # Create control-normalized experimental plot
    res, cond, cis = [], [], []
    iterations = 1000
    for (k, v) in hexps.items():
        exp_baseline = np.mean(v[k]["baseline"])
        control_baseline = np.mean(v["control"]["baseline"])
        control_ca = np.mean(v["control"]["ca"])
        res.append((control_baseline - exp_baseline) / (control_ca - control_baseline))
        cond.append(k)

        bcis = []
        for i in np.arange(iterations):
            exp_baseline = np.mean(np.asarray(v[k]["baseline"])[np.random.randint(low=0, high=len(v[k]["baseline"]), size=len(v[k]["baseline"]))])
            control_baseline = np.mean(np.asarray(v["control"]["baseline"])[np.random.randint(low=0, high=len(v["control"]["baseline"]), size=len(v["control"]["baseline"]))])
            control_ca = np.mean(np.asarray(v["control"]["ca"])[np.random.randint(low=0, high=len(v["control"]["ca"]), size=len(v["control"]["ca"]))])
            bcis.append(np.maximum((control_baseline - exp_baseline) / (control_ca - control_baseline), 0))
        bcis = np.asarray(bcis)
        upper = np.percentile(bcis, 80)
        lower = np.percentile(bcis, 20)
        cis.append(upper - lower)

    f, ax = plt.subplots(1, 1)
    ax.scatter(np.arange(len(res)), res)
    ax.errorbar(np.arange(len(res)), res, yerr=cis, ls='none', capsize=3)
    ax.set_ylabel("Spark reduction score")  #  (> 0 is better than control)")
    ax.set_xticks(np.arange(len(cond)))
    ax.set_xticklabels(cond)
    plt.suptitle("(control_baseline - exp_baseline) / (control_ca - control_baseline)")
    plt.savefig(drug_screen_normalized_plot)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default=None, type=str, help="Path to the project configuration YAML file.")
    args = vars(parser.parse_args())
    main(**args)
