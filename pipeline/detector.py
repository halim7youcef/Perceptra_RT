import cv2
import numpy as np
from ultralytics import YOLO


class ObjectDetector:
    def __init__(self, cfg):
        self.conf       = cfg["model"]["yolo"]["conf"]
        self.av_classes = set(cfg["model"]["yolo"]["av_classes"])
        self.model      = YOLO(cfg["model"]["yolo"]["weights"])
        self.names      = self.model.names

        self.av_class_ids = {
            cid for cid, name in self.names.items()
            if name in self.av_classes
        }
        print(f"[Detector] AV class IDs: "
              f"{sorted([(i, self.names[i]) for i in self.av_class_ids])}")

    def detect(self, frame_bgr):
        results          = self.model(frame_bgr, verbose=False, conf=self.conf)[0]
        keep             = [i for i, c in enumerate(results.boxes.cls)
                            if int(c) in self.av_class_ids]
        results.boxes    = results.boxes[keep]
        annotated        = results.plot()

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            box_h     = y2 - y1
            proximity = "CLOSE" if box_h > 80 else "FAR"
            color     = (0, 0, 255) if proximity == "CLOSE" else (255, 200, 0)
            cv2.putText(annotated, proximity, (x1, max(y1 - 5, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        num_cars    = sum(1 for b in results.boxes if self.names[int(b.cls[0])] == "car")
        num_persons = sum(1 for b in results.boxes if self.names[int(b.cls[0])] == "person")
        avg_conf    = round(float(results.boxes.conf.mean())
                            if len(results.boxes) else 0.0, 4)

        return annotated, results, {
            "num_detections" : len(results.boxes),
            "num_cars"       : num_cars,
            "num_persons"    : num_persons,
            "avg_confidence" : avg_conf,
        }