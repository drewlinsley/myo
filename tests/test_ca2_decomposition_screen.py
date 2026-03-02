import pytest
import numpy as np
from ca2_decomposition_screen import main

def test_main_function():
    # Mock the necessary inputs
    model_type = "nucleus_rois"
    debug_videos = False
    moments_dir = "test_moments_dir"
    movies_dir = "test_movies_dir"

    # Call the main function
    result = main(model_type, debug_videos, moments_dir, movies_dir)

    # Assert expected outcomes
    assert os.path.exists(moments_dir)
    assert os.path.exists(movies_dir)
    # Add more assertions based on the expected behavior of the main function

@pytest.mark.parametrize("model_type", ["nucleus_rois", "cytoplasm_rois", "cytoplasm_far_rois"])
def test_main_function_model_types(model_type):
    # Test the main function with different model types
    result = main(model_type)
    # Add assertions specific to each model type

def test_data_loading():
    # Test the data loading part of the main function
    files = glob(os.path.join("test_data", "*", "*", "*.npy"))
    train_files, test_files = [], []
    for f in files:
        for filt in test_exps:
            if filt in f:
                test_files.append(f)
                break
        else:
            train_files.append(f)
    
    assert len(train_files) > 0
    assert len(test_files) > 0
    # Add more assertions about the loaded data

