import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort


class ObjectTracker:
    def __init__(self, cfg):
        tcfg          = cfg["tracker"]
        self.tracker  = DeepSort(
            max_age              = tcfg["max_age"],
            n_init               = tcfg["n_init"],
            max_cosine_distance  = tcfg["max_cosine_distance"],
            nn_budget            = tcfg["nn_budget"],
        )
        self._colors  = {}

    def _get_color(self, track_id):
        if track_id not in self._colors:
            seed = int(abs(hash(str(track_id))) % (2**31))
            rng  = np.random.default_rng(seed)
            self._colors[track_id] = tuple(int(c) for c in rng.integers(80, 230, 3))
        return self._colors[track_id]

    def update(self, frame_bgr, yolo_results, yolo_names):
        vis  = frame_bgr.copy()
        dets = []

        for box in yolo_results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf     = float(box.conf[0])
            cls_name = yolo_names[int(box.cls[0])]
            dets.append(([x1, y1, x2 - x1, y2 - y1], conf, cls_name))

        tracks        = self.tracker.update_tracks(dets, frame=frame_bgr)
        active_tracks = []

        for track in tracks:
            if not track.is_confirmed():
                continue
            tid             = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            cls_name        = track.get_det_class() or "object"
            color           = self._get_color(tid)

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            label       = f"ID:{tid} {cls_name}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(vis, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(vis, (cx, cy), 4, color, -1)

            active_tracks.append({
                "track_id" : tid,
                "class"    : cls_name,
                "bbox"     : (x1, y1, x2, y2),
                "center"   : (cx, cy),
            })

        cv2.putText(vis, f"Tracks: {len(active_tracks)}", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        return vis, active_tracks, {"num_active_tracks": len(active_tracks)}