import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor


SEG_PALETTE = np.array([
    [128, 64,128], [244, 35,232], [ 70, 70, 70], [102,102,156],
    [190,153,153], [153,153,153], [250,170, 30], [220,220,  0],
    [107,142, 35], [152,251,152], [ 70,130,180], [220, 20, 60],
    [255,  0,  0], [  0,  0,142], [  0,  0, 70], [  0, 60,100],
    [  0, 80,100], [  0,  0,230], [119, 11, 32],
], dtype=np.uint8)

SEG_LABELS = [
    "road","sidewalk","building","wall","fence","pole",
    "traffic light","traffic sign","vegetation","terrain","sky",
    "person","rider","car","truck","bus","train","motorcycle","bicycle"
]


class SceneSegmentor:
    def __init__(self, cfg, device):
        name            = cfg["model"]["segmentation"]["model"]
        self.alpha      = cfg["model"]["segmentation"]["blend_alpha"]
        self.device     = device
        self.processor  = SegformerImageProcessor.from_pretrained(name)
        self.model      = SegformerForSemanticSegmentation.from_pretrained(name)
        self.model.to(device).eval()
        print(f"[Segmentor] SegFormer loaded on {device}")

    def segment(self, frame_bgr):
        rgb    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits

        up      = F.interpolate(logits, size=frame_bgr.shape[:2],
                                mode="bilinear", align_corners=False)
        seg_map = up.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)

        seg_bgr = cv2.cvtColor(SEG_PALETTE[seg_map], cv2.COLOR_RGB2BGR)
        blended = cv2.addWeighted(frame_bgr, self.alpha, seg_bgr, 1 - self.alpha, 0)
        cv2.putText(blended, "Segmentation (SegFormer-B0)", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        unique, counts = np.unique(seg_map, return_counts=True)
        total          = seg_map.size
        class_dist     = {SEG_LABELS[i]: round(c / total * 100, 2)
                          for i, c in zip(unique, counts) if i < len(SEG_LABELS)}
        return blended, seg_map, class_dist

    @staticmethod
    def dist_metrics(class_dist):
        return {
            "seg_road_pct"       : class_dist.get("road",       0.0),
            "seg_car_pct"        : class_dist.get("car",        0.0),
            "seg_sky_pct"        : class_dist.get("sky",        0.0),
            "seg_vegetation_pct" : class_dist.get("vegetation", 0.0),
        }