# Declare paths
directories = {
    "output_plot_dir": "nucleus_plots",
    "output_dplot_dir": "nucleus_detrend_plots",
    "output_fdplot_dir": "nucleus_far_detrend_plots",
    "nucleus_roi_dir": "nucleus_rois",
    "nucleus_mask_roi_dir": "nucleus_mask_rois",
    "cyto_roi_dir": "cytoplasm_rois",
    "cyto_far_roi_dir": "cytoplasm_far_rois",
    "cyto_roi_spl_dir": "cytoplasm_split_rois",
    "cyto_far_roi_spl_dir": "cytoplasm_far_split_rois",
    "whole_cell_roi_spl_dir": "whole_cell_split_rois",
    "nucleus_roi_plot_dir": "nucleus_rois_plots",
    "cyto_roi_plot_dir": "cytoplasm_rois_plots",
    "cyto_far_roi_plot_dir": "cytoplasm_far_rois_plots",
    "sanity_checks": "nucleus_sanity",
    "annotated_video_dir": "annotated_rois",
    "model_asset_dir": "auto_annotation_models"
}

keys = [
    ["ra306", "baseline"],
    ["aip", "baseline"],
    ["el20", "baseline"],
    ["control", "baseline"],
    ["ra306", "_post"],
    ["aip", "_post"],
    ["el20", "_post"],
    ["control", "_post"],
    ["ra306", "_ca+"],
    ["aip", "_ca+"],
    ["el20", "_ca+"],
    ["control", "_ca+"],
    ["el20", "el20_"]
]

test_exps = [
    "20230412 RA306 Experiment 10uM",
    "20230428 AIP Experiment 585nM Plate2",
    "20230906 EL20 Experiment 1uM",
]
