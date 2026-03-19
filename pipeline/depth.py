import cv2
import numpy as np
import torch
import torch.nn.functional as F


class DepthEstimator:
    def __init__(self, cfg, device):
        self.device = device
        self.model  = torch.hub.load(
            "intel-isl/MiDaS", cfg["model"]["depth"]["model"], trust_repo=True
        )
        self.transforms = torch.hub.load(
            "intel-isl/MiDaS", "transforms", trust_repo=True
        ).small_transform
        self.model.to(device).eval()
        print(f"[Depth] MiDaS loaded on {device}")

    def estimate(self, frame_bgr):
        rgb          = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        input_tensor = self.transforms(rgb).to(self.device)

        with torch.no_grad():
            depth = self.model(input_tensor)
            depth = F.interpolate(
                depth.unsqueeze(1),
                size=frame_bgr.shape[:2],
                mode="bicubic",
                align_corners=False
            ).squeeze().cpu().numpy()

        depth_norm    = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_MAGMA)
        cv2.putText(depth_colored, "Depth (MiDaS)", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        return depth_colored, depth_norm