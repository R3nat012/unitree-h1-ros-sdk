#!/usr/bin/env python3
"""
RL walking controller for the Unitree H1 / H1_2 in Ignition Gazebo.

Runs Unitree's pretrained walking policy (unitree_rl_gym deploy/pre_train/<robot>/motion.pt,
converted to numpy weights) at 50 Hz and sends joint position targets to the Gazebo
JointPositionController plugins. The PD gains of those plugins must match the policy's
gains (see README). Pick the robot with the `robot` parameter: 'h1' or 'h1_2'.

Subscribed:
    /cmd_vel      (geometry_msgs/Twist)     linear.x, linear.y, angular.z
    /h1/arm_cmd   (sensor_msgs/JointState)  arm/torso targets by joint name (rad)
    /h1/hand_cmd  (sensor_msgs/JointState)  h1_2 only: finger targets by joint name (rad),
                                            or 'left_hand' / 'right_hand' = 0 (open) .. 1 (closed)
    /joint_states (sensor_msgs/JointState)  joint positions and velocities from Gazebo
    /imu          (sensor_msgs/Imu)         torso IMU from Gazebo
Published:
    <cmd_topic_format>.format(joint)  (std_msgs/Float64)  one per controlled joint
"""
import os
import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, qos_profile_sensor_data
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Twist
from ament_index_python.packages import get_package_share_directory

from h1_control.h1_policy_np import H1Policy

# ---- per-robot settings (from unitree_rl_gym deploy/deploy_real/configs/*.yaml) --------
ROBOTS = {
    'h1': dict(
        policy='h1_policy.npz',
        legs=['left_hip_yaw_joint', 'left_hip_roll_joint', 'left_hip_pitch_joint',
              'left_knee_joint', 'left_ankle_joint',
              'right_hip_yaw_joint', 'right_hip_roll_joint', 'right_hip_pitch_joint',
              'right_knee_joint', 'right_ankle_joint'],
        default=[0.0, 0.0, -0.1, 0.3, -0.2] * 2,
        arm_limits={
            'torso_joint': (-2.35, 2.35),
            'left_shoulder_pitch_joint': (-2.87, 2.87), 'right_shoulder_pitch_joint': (-2.87, 2.87),
            'left_shoulder_roll_joint': (-0.34, 3.11), 'right_shoulder_roll_joint': (-3.11, 0.34),
            'left_shoulder_yaw_joint': (-1.3, 4.45), 'right_shoulder_yaw_joint': (-4.45, 1.3),
            'left_elbow_joint': (-1.25, 2.61), 'right_elbow_joint': (-1.25, 2.61),
        },
        hands=None,
    ),
    'h1_2': dict(
        policy='h1_2_policy.npz',
        legs=['left_hip_yaw_joint', 'left_hip_pitch_joint', 'left_hip_roll_joint',
              'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
              'right_hip_yaw_joint', 'right_hip_pitch_joint', 'right_hip_roll_joint',
              'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint'],
        default=[0.0, -0.16, 0.0, 0.36, -0.2, 0.0] * 2,
        arm_limits={
            'torso_joint': (-2.35, 2.35),
            'left_shoulder_pitch_joint': (-3.14, 1.57), 'right_shoulder_pitch_joint': (-3.14, 1.57),
            'left_shoulder_roll_joint': (-0.38, 3.4), 'right_shoulder_roll_joint': (-3.4, 0.38),
            'left_shoulder_yaw_joint': (-2.66, 3.01), 'right_shoulder_yaw_joint': (-3.01, 2.66),
            'left_elbow_joint': (-0.95, 3.18), 'right_elbow_joint': (-0.95, 3.18),
            'left_wrist_roll_joint': (-3.01, 2.75), 'right_wrist_roll_joint': (-2.75, 3.01),
            'left_wrist_pitch_joint': (-0.4625, 0.4625), 'right_wrist_pitch_joint': (-0.4625, 0.4625),
            'left_wrist_yaw_joint': (-1.27, 1.27), 'right_wrist_yaw_joint': (-1.27, 1.27),
        },
        # actuated finger joint -> (lower, upper, [(mimic joint, multiplier), ...])
        hands={
            f'{s}_{n}': lim for s in ('L', 'R') for n, lim in {
                'thumb_proximal_yaw_joint': (-0.1, 1.3, []),
                'thumb_proximal_pitch_joint': (-0.1, 0.6, [('thumb_intermediate_joint', 1.6),
                                                            ('thumb_distal_joint', 2.4)]),
                'index_proximal_joint': (0.0, 1.7, [('index_intermediate_joint', 1.0)]),
                'middle_proximal_joint': (0.0, 1.7, [('middle_intermediate_joint', 1.0)]),
                'ring_proximal_joint': (0.0, 1.7, [('ring_intermediate_joint', 1.0)]),
                'pinky_proximal_joint': (0.0, 1.7, [('pinky_intermediate_joint', 1.0)]),
            }.items()
        },
    ),
}

ANG_VEL_SCALE = 0.25
DOF_POS_SCALE = 1.0
DOF_VEL_SCALE = 0.05
ACTION_SCALE = 0.25
CMD_SCALE = np.array([2.0, 2.0, 0.25])
MAX_CMD = np.array([0.8, 0.5, 1.57])
GAIT_PERIOD = 0.8
ARM_MAX_SPEED = 1.0   # rad/s; arm targets ramp toward the commanded value
HAND_MAX_SPEED = 2.0  # rad/s


def quat_to_mat(w, x, y, z):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def ramp(now, goal, step):
    for n, g in goal.items():
        now[n] = now[n] + min(max(g - now[n], -step), step)


class RLWalkController(Node):
    def __init__(self):
        super().__init__('rl_walk_controller')

        self.declare_parameter('robot', 'h1')
        robot = self.get_parameter('robot').value
        if robot not in ROBOTS:
            raise ValueError(f"robot must be one of {list(ROBOTS)}, got '{robot}'")
        cfg = ROBOTS[robot]
        self.legs = cfg['legs']
        self.default = np.array(cfg['default'])
        self.arm_limits = cfg['arm_limits']
        self.hands = cfg['hands'] or {}

        default_policy = os.path.join(
            get_package_share_directory('h1_control'), 'policy', cfg['policy'])
        self.declare_parameter('policy_path', default_policy)
        self.declare_parameter('cmd_topic_format', '/model/h1/joint/{}/cmd_pos')
        self.declare_parameter('control_dt', 0.02)
        # Added to /cmd_vel to cancel drift (vx, vy, wz), e.g. [0.1, 0.0, 0.0]
        self.declare_parameter('cmd_trim', [0.0, 0.0, 0.0])

        policy_path = self.get_parameter('policy_path').value
        topic_fmt = self.get_parameter('cmd_topic_format').value
        self.dt = float(self.get_parameter('control_dt').value)
        self.trim = np.array(self.get_parameter('cmd_trim').value, dtype=float)

        self.policy = H1Policy(policy_path)
        self.get_logger().info(f'Robot {robot}: loaded policy {policy_path}')

        self.q = {}
        self.dq = {}
        self.imu = None
        self.cmd = np.zeros(3)
        self.action = np.zeros(len(self.legs))
        self.step = 0
        self.fallen = False
        self.arm_goal = {j: 0.0 for j in self.arm_limits}
        self.arm_now = dict(self.arm_goal)
        self.hand_goal = {j: 0.0 for j in self.hands}
        self.hand_now = dict(self.hand_goal)

        controlled = list(self.legs) + list(self.arm_limits)
        for j, (_, _, mimics) in self.hands.items():
            controlled += [j] + [j[:2] + m for m, _ in mimics]

        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST, depth=10)
        self.create_subscription(JointState, '/joint_states', self.on_joints, qos)
        self.create_subscription(Imu, '/imu', self.on_imu, qos_profile_sensor_data)
        self.create_subscription(Twist, '/cmd_vel', self.on_cmd, qos)
        self.create_subscription(JointState, '/h1/arm_cmd', self.on_arm_cmd, qos)
        if self.hands:
            self.create_subscription(JointState, '/h1/hand_cmd', self.on_hand_cmd, qos)

        self.pubs = {j: self.create_publisher(Float64, topic_fmt.format(j), qos)
                     for j in controlled}
        self.create_timer(self.dt, self.loop)
        self.create_timer(2.0, self.status)

    # ---- callbacks -------------------------------------------------------
    def on_joints(self, msg):
        # Ignore joint_state_publisher (positions only, usually zeros);
        # Gazebo's JointStatePublisher sends velocities too.
        if len(msg.velocity) != len(msg.name):
            return
        for n, p, v in zip(msg.name, msg.position, msg.velocity):
            self.q[n] = p
            self.dq[n] = v

    def on_imu(self, msg):
        self.imu = msg

    def on_arm_cmd(self, msg):
        for n, p in zip(msg.name, msg.position):
            if n not in self.arm_limits:
                self.get_logger().warn(f'/h1/arm_cmd: unknown or leg joint "{n}" ignored')
                continue
            lo, hi = self.arm_limits[n]
            self.arm_goal[n] = min(max(float(p), lo), hi)

    def on_hand_cmd(self, msg):
        for n, p in zip(msg.name, msg.position):
            if n in ('left_hand', 'right_hand'):
                side = 'L_' if n == 'left_hand' else 'R_'
                frac = min(max(float(p), 0.0), 1.0)
                for j, (lo, hi, _) in self.hands.items():
                    if j.startswith(side) and 'yaw' not in j:
                        self.hand_goal[j] = max(lo, 0.0) + frac * (hi - max(lo, 0.0))
            elif n in self.hands:
                lo, hi, _ = self.hands[n]
                self.hand_goal[n] = min(max(float(p), lo), hi)
            else:
                self.get_logger().warn(
                    f'/h1/hand_cmd: "{n}" is not an actuated finger joint (mimic joints follow '
                    'automatically); use left_hand / right_hand for open-close')

    def on_cmd(self, msg):
        c = np.array([msg.linear.x, msg.linear.y, msg.angular.z])
        self.cmd = np.clip(c, -MAX_CMD, MAX_CMD)

    # ---- control ---------------------------------------------------------
    def publish(self, targets):
        for j, v in targets.items():
            m = Float64()
            m.data = float(v)
            self.pubs[j].publish(m)

    def upper_body_targets(self):
        ramp(self.arm_now, self.arm_goal, ARM_MAX_SPEED * self.dt)
        ramp(self.hand_now, self.hand_goal, HAND_MAX_SPEED * self.dt)
        t = dict(self.arm_now)
        for j, v in self.hand_now.items():
            t[j] = v
            for mimic, mult in self.hands[j][2]:
                t[j[:2] + mimic] = mult * v
        return t

    def base_state(self):
        """Pelvis gravity vector and angular velocity from the torso IMU."""
        o = self.imu.orientation
        w = self.imu.angular_velocity
        r_torso = quat_to_mat(o.w, o.x, o.y, o.z)
        waist = self.q.get('torso_joint', 0.0)
        waist_dq = self.dq.get('torso_joint', 0.0)
        rz = rot_z(waist)
        r_pelvis = r_torso @ rz.T
        omega = rz @ np.array([w.x, w.y, w.z]) - np.array([0.0, 0.0, waist_dq])
        gravity = r_pelvis.T @ np.array([0.0, 0.0, -1.0])
        return gravity, omega

    def loop(self):
        targets = self.upper_body_targets()
        ready = self.imu is not None and all(j in self.q for j in self.legs)
        if not ready or self.fallen:
            targets.update(dict(zip(self.legs, self.default)))
            self.publish(targets)
            return

        gravity, omega = self.base_state()
        if self.step == 0:
            self.get_logger().info(
                f'Starting policy. Gravity in pelvis frame: {np.round(gravity, 2)} '
                '(upright is about [0, 0, -1])')
        if gravity[2] > -0.5:  # tilted more than ~60 degrees
            self.fallen = True
            self.get_logger().warn(
                f'Robot fell after {self.step} policy steps '
                f'(gravity {np.round(gravity, 2)}); holding default pose. Restart to try again.')
            return

        qj = np.array([self.q[j] for j in self.legs])
        dqj = np.array([self.dq[j] for j in self.legs])
        self.step += 1
        phase = (self.step * self.dt) % GAIT_PERIOD / GAIT_PERIOD

        obs = np.concatenate([
            omega * ANG_VEL_SCALE,
            gravity,
            np.clip(self.cmd + self.trim, -MAX_CMD, MAX_CMD) * CMD_SCALE,
            (qj - self.default) * DOF_POS_SCALE,
            dqj * DOF_VEL_SCALE,
            self.action,
            [math.sin(2 * math.pi * phase), math.cos(2 * math.pi * phase)],
        ])
        self.action = self.policy(obs)
        targets.update(dict(zip(self.legs, self.action * ACTION_SCALE + self.default)))
        self.publish(targets)

    def status(self):
        if self.imu is None or not self.q:
            self.get_logger().info('Waiting for /imu and /joint_states from Gazebo...')
        elif not self.fallen:
            self.get_logger().info(
                f'Walking | cmd vx={self.cmd[0]:.2f} vy={self.cmd[1]:.2f} wz={self.cmd[2]:.2f}')


def main():
    rclpy.init()
    node = RLWalkController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
