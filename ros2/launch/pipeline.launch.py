from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package    = "av_pipeline",
            executable = "perception_node",
            name       = "av_perception",
            output     = "screen",
            parameters = [{"use_sim_time": False}],
        )
    ])