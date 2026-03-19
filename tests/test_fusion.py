import numpy as np
import pytest
import yaml
from pipeline.fusion import SensorFusion


@pytest.fixture(scope="module")
def cfg():
    with open("config/pipeline.yaml") as f:
        return yaml.safe_load(f)

@pytest.fixture(scope="module")
def fusion(cfg):
    return SensorFusion(cfg)

def dummy_tracks():
    return [{
        "track_id" : 1,
        "class"    : "car",
        "bbox"     : (100, 100, 200, 180),
        "center"   : (150, 140),
    }]

def test_fusion_returns_objects(fusion):
    frame      = np.zeros((320, 640, 3), dtype=np.uint8)
    depth_norm = np.random.randint(0, 255, (320, 640), dtype=np.uint8)
    _, objects = fusion.fuse(frame, dummy_tracks(), depth_norm)
    assert len(objects) == 1

def test_fusion_3d_keys(fusion):
    frame      = np.zeros((320, 640, 3), dtype=np.uint8)
    depth_norm = np.random.randint(0, 255, (320, 640), dtype=np.uint8)
    _, objects = fusion.fuse(frame, dummy_tracks(), depth_norm)
    for key in ["x3d", "y3d", "z3d", "proximity", "depth_raw"]:
        assert key in objects[0]

def test_unproject_close_object(fusion):
    x, y, z = fusion._unproject(320, 160, 255)
    assert z == pytest.approx(0.01, abs=0.01)

def test_unproject_far_object(fusion):
    x, y, z = fusion._unproject(320, 160, 0)
    assert z == pytest.approx(255 * 0.05, rel=0.01)

def test_bev_no_crash_empty(fusion):
    bev = fusion.bird_eye_view([], lambda tid: (255, 255, 255))
    assert bev.shape == (320, 320, 3)