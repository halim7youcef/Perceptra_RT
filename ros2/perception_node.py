"""
ROS2 perception node stub.
Subscribes to /camera/image_raw, publishes fused detections to /av/detections.
Run with: ros2 run av_pipeline perception_node
"""
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from std_msgs.msg import String
    import json
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    print("[ROS2] rclpy not found — running in stub mode")


class PerceptionNode:
    """
    Wraps the full AV pipeline as a ROS2 node.
    Topics:
      sub : /camera/image_raw  (sensor_msgs/Image)
      pub : /av/detections     (std_msgs/String — JSON payload)
      pub : /av/depth          (sensor_msgs/Image)
    """

    NODE_NAME = "av_perception"
    SUB_TOPIC = "/camera/image_raw"
    PUB_DETECTIONS = "/av/detections"
    PUB_DEPTH      = "/av/depth"

    def __init__(self):
        if not ROS2_AVAILABLE:
            print("[PerceptionNode] Stub mode — ROS2 not available")
            return

        rclpy.init()
        self.node = Node(self.NODE_NAME)

        self.sub = self.node.create_subscription(
            Image, self.SUB_TOPIC, self._callback, 10
        )
        self.pub_det = self.node.create_publisher(String, self.PUB_DETECTIONS, 10)
        self.pub_dep = self.node.create_publisher(Image,  self.PUB_DEPTH,      10)

        self.node.get_logger().info(f"[{self.NODE_NAME}] Node started")

    def _callback(self, msg):
        """Process incoming frame and publish detections."""
        # convert ROS Image → numpy (BGR)
        import numpy as np
        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, -1
        )
        # --- plug in pipeline here ---
        # _, results, _ = detector.detect(frame)
        # tracked_frame, tracks, _ = tracker.update(frame, results)
        # depth_colored, depth_norm = depth.estimate(frame)
        # fused_frame, objects = fusion.fuse(tracked_frame, tracks, depth_norm)

        payload = String()
        payload.data = json.dumps({
            "num_tracks"   : 0,   # replace with len(tracks)
            "objects"      : [],  # replace with objects
        })
        self.pub_det.publish(payload)

    def spin(self):
        if ROS2_AVAILABLE:
            rclpy.spin(self.node)

    def destroy(self):
        if ROS2_AVAILABLE:
            self.node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    node = PerceptionNode()
    node.spin()