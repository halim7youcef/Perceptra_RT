import numpy as np


def aggregate(all_metrics):
    def avg(key):
        return round(float(np.mean([m[key] for m in all_metrics])), 4)

    return {
        "avg_total_ms"      : avg("time_total_ms"),
        "avg_detections"    : avg("num_detections"),
        "avg_tracks"        : avg("num_active_tracks"),
        "avg_confidence"    : avg("avg_confidence"),
        "avg_yolo_ms"       : avg("time_yolo_ms"),
        "avg_tracking_ms"   : avg("time_tracking_ms"),
        "avg_depth_ms"      : avg("time_depth_ms"),
        "avg_fusion_ms"     : avg("time_fusion_ms"),
        "avg_seg_ms"        : avg("time_seg_ms"),
        "avg_road_pct"      : avg("seg_road_pct"),
        "avg_close_objects" : avg("num_close_objects"),
        "avg_depth_z"       : avg("avg_depth_z"),
    }