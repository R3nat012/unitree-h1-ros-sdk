"""H1 / H1_2 in Gazebo walking with Unitree's RL policy. Drive it with /cmd_vel.

    ros2 launch h1_control rl_walk.launch.py robot:=h1_2
(set H1_ROBOT=h1_2 in the environment too, so h1_gazebo.launch.py loads the H1_2 URDF)
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_h1_gazebo = get_package_share_directory('h1_gazebo')

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_h1_gazebo, 'launch', 'h1_gazebo.launch.py')),
        launch_arguments={'use_sim_time': 'true',
                          'rviz': LaunchConfiguration('rviz')}.items())

    walker = TimerAction(period=3.0, actions=[Node(
        package='h1_control',
        executable='rl_walk_controller.py',
        name='rl_walk_controller',
        output='screen',
        parameters=[{'use_sim_time': True,
                     'robot': LaunchConfiguration('robot'),
                     'cmd_trim': [0.0, 0.0, 0.0]}],
    )])

    return LaunchDescription([
        DeclareLaunchArgument('rviz', default_value='false'),
        DeclareLaunchArgument('robot', default_value=os.environ.get('H1_ROBOT', 'h1'),
                              description="'h1' or 'h1_2'"),
        gazebo,
        walker,
    ])
