import pytest
import numpy as np
from src.utils import seq_generate_roi_data

def test_seq_generate_roi_data():
    # Create mock input data
    dr = "test_dir"
    nuclei = np.random.rand(10, 100, 100)
    cytoplasms = np.random.rand(10, 100, 100)
    cytoplasms_far = np.random.rand(10, 100, 100)
    video = np.random.rand(10, 100, 100)
    detrend_video = np.random.rand(10, 100, 100)
    nuclei_ids = [1, 2, 3]
    hroi = 32
    roi_size = 64
    directories = {
        "nucleus_roi_dir": "test_nucleus_dir",
        "cyto_roi_dir": "test_cyto_dir",
        "cyto_far_roi_dir": "test_cyto_far_dir",
        "cyto_roi_spl_dir": "test_cyto_spl_dir",
        "cyto_far_roi_spl_dir": "test_cyto_far_spl_dir",
        "nucleus_roi_plot_dir": "test_nucleus_plot_dir",
        "cyto_roi_plot_dir": "test_cyto_plot_dir",
        "cyto_far_roi_plot_dir": "test_cyto_far_plot_dir",
        "whole_cell_roi_spl_dir": "test_whole_cell_spl_dir",
        "image_dir": "test_image_dir"
    }

    # Call the function
    result = seq_generate_roi_data(dr, nuclei, cytoplasms, cytoplasms_far, video, detrend_video, nuclei_ids, hroi, roi_size, directories, 
                                   normalize_rois=True, use_detrended=False, use_dff=True, debug=False, plot_figures=False, 
                                   version="test", save_npy_rois=False)

    # Assert the expected output
    assert isinstance(result, pd.DataFrame)
    assert "experiment" in result.columns
    assert "cell" in result.columns
    assert "region" in result.columns

