import numpy as np
import pytest
import yaml
from pipeline.depth import DepthEstimator


@pytest.fixture(scope="module")
def cfg():
    with open("config/pipeline.yaml") as f:
        return yaml.safe_load(f)

@pytest.fixture(scope="module")
def depth(cfg):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return DepthEstimator(cfg, device)

def test_depth_loads(depth):
    assert depth.model is not None

def test_estimate_returns_two_outputs(depth):
    frame  = np.random.randint(0, 255, (320, 640, 3), dtype=np.uint8)
    result = depth.estimate(frame)
    assert len(result) == 2

def test_depth_norm_range(depth):
    frame      = np.random.randint(0, 255, (320, 640, 3), dtype=np.uint8)
    _, norm    = depth.estimate(frame)
    assert norm.min() >= 0
    assert norm.max() <= 255

def test_depth_output_shape(depth):
    frame      = np.random.randint(0, 255, (320, 640, 3), dtype=np.uint8)
    colored, _ = depth.estimate(frame)
    assert colored.shape == (320, 640, 3)