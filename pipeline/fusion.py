import cv2
import numpy as np


class SensorFusion:
    def __init__(self, cfg):
        dcfg         = cfg["model"]["depth"]
        pcfg         = cfg["proximity"]
        self.fx      = dcfg["fx"]
        self.fy      = dcfg["fy"]
        self.cx      = dcfg["cx"]
        self.cy      = dcfg["cy"]
        self.scale   = dcfg["scale"]
        self.close_t = pcfg["close_threshold_m"]
        self.mid_t   = pcfg["mid_threshold_m"]

        bcfg              = cfg["bev"]
        self.bev_size     = bcfg["size"]
        self.max_depth    = bcfg["max_depth"]
        self.max_lat      = bcfg["max_lateral"]
        self.rings        = bcfg["rings"]

    def _unproject(self, u, v, depth_val_norm):
        z = (255.0 - float(depth_val_norm)) * self.scale
        z = max(z, 0.01)
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        return x, y, z

    def fuse(self, frame, tracks, depth_norm):
        vis     = frame.copy()
        objects = []

        for t in tracks:
            cx, cy = t["center"]
            cx_c   = np.clip(cx, 0, depth_norm.shape[1] - 1)
            cy_c   = np.clip(cy, 0, depth_norm.shape[0] - 1)
            d_val  = float(depth_norm[cy_c, cx_c])

            x3d, y3d, z3d = self._unproject(cx, cy, d_val)

            proximity = ("CLOSE" if z3d < self.close_t else
                         "MID"   if z3d < self.mid_t   else "FAR")
            p_color   = ((0, 0, 255)   if proximity == "CLOSE" else
                         (0, 165, 255) if proximity == "MID"   else (0, 200, 0))

            cv2.putText(vis, f"z={z3d:.1f}m {proximity}",
                        (t["bbox"][0], t["bbox"][3] + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, p_color, 1, cv2.LINE_AA)

            objects.append({**t,
                            "depth_raw" : round(d_val, 2),
                            "x3d"       : round(x3d, 2),
                            "y3d"       : round(y3d, 2),
                            "z3d"       : round(z3d, 2),
                            "proximity" : proximity})
        return vis, objects

    def bird_eye_view(self, objects, track_color_fn):
        s      = self.bev_size
        bev    = np.full((s, s, 3), (30, 30, 30), dtype=np.uint8)
        ego_x  = s // 2
        ego_y  = s - 30

        for i in range(0, s, s // 8):
            cv2.line(bev, (i, 0), (i, s), (50, 50, 50), 1)
            cv2.line(bev, (0, i), (s, i), (50, 50, 50), 1)

        for ring_m in self.rings:
            ring_px = int(ring_m / self.max_depth * (s - 40))
            cv2.circle(bev, (ego_x, ego_y), ring_px, (60, 60, 60), 1)
            cv2.putText(bev, f"{ring_m:.0f}m", (ego_x + ring_px + 2, ego_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (80, 80, 80), 1)

        cv2.rectangle(bev, (ego_x - 8, ego_y - 16), (ego_x + 8, ego_y), (0, 200, 255), -1)
        cv2.putText(bev, "ego", (ego_x - 10, ego_y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 200, 255), 1)

        for obj in objects:
            z, x = obj["z3d"], obj["x3d"]
            if z <= 0 or z > self.max_depth:
                continue
            bx = np.clip(int((x / self.max_lat + 1.0) * 0.5 * s), 5, s - 5)
            by = np.clip(int(ego_y - (z / self.max_depth) * (s - 40)), 5, s - 5)
            c  = track_color_fn(obj["track_id"])
            cv2.circle(bev, (bx, by), 7, c, -1)
            cv2.putText(bev, f"ID:{obj['track_id']}", (bx + 8, by + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, c, 1)

        cv2.putText(bev, "Bird eye view", (6, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(bev, "^ forward", (s - 68, s - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
        return bev

    @staticmethod
    def compute_metrics(objects):
        return {
            "num_close_objects" : sum(1 for o in objects if o["proximity"] == "CLOSE"),
            "avg_depth_z"       : round(
                float(np.mean([o["z3d"] for o in objects])) if objects else 0.0, 2),
        }