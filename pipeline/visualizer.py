import cv2
import numpy as np


class Visualizer:
    @staticmethod
    def compose(fused_frame, seg_frame, depth_colored, bev_frame):
        """
        Compose 4 panels into a 2x2 output grid:
          top    : fused (tracking + 3D labels)  |  segmentation
          bottom : depth map                      |  bird's eye view
        """
        h, w   = fused_frame.shape[:2]
        seg_r  = cv2.resize(seg_frame,    (w, h))
        dep_r  = cv2.resize(depth_colored,(w, h))
        bev_r  = cv2.resize(bev_frame,    (w, h))

        top    = np.hstack([fused_frame, seg_r])
        bottom = np.hstack([dep_r,       bev_r])
        return  np.vstack([top,          bottom])

    @staticmethod
    def add_header(frame, text, pos=(8, 20)):
        cv2.putText(frame, text, pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        return frame