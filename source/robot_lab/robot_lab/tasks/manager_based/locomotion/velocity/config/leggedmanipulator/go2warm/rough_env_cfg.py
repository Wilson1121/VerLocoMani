# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp
from robot_lab.tasks.manager_based.locomotion.velocity.veh_velocity_env_cfg import (
    ARM_JOINT_NAMES,
    EventCfg,
    LEG_JOINT_NAMES,
    LocomotionVelocityRoughEnvCfg,
    WHEEL_BODY_NAMES,
    WHEEL_JOINT_NAMES,
)

##
# Pre-defined configs
##
from robot_lab.assets.unitree import UNITREE_Go2WArm_CFG  # isort: skip


@configclass
class UnitreeGo2WArmEventCfg(EventCfg):
    """Domain randomization for the Go2W chassis carrying a Piper arm."""

    ###### Contact properties #######

    randomize_wheel_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=WHEEL_BODY_NAMES),
            "static_friction_range": (0.7, 1.1),
            "dynamic_friction_range": (0.6, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )

    ###### Rigid-body properties #######

    # The chassis, arm mount, and Piper base are merged into the root body by
    # the URDF converter. This term covers fixed equipment and assembly error.
    randomize_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-0.3, 0.5),
            "operation": "add",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )

    randomize_base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {
                "x": (-0.015, 0.015),
                "y": (-0.010, 0.010),
                "z": (-0.005, 0.010),
            },
        },
    )

    randomize_arm_link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[f"link{i}" for i in range(1, 7)]),
            "mass_distribution_params": (0.95, 1.05),
            "operation": "scale",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )

    ###### Actuator properties #######

    randomize_leg_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES),
            "stiffness_distribution_params": (0.95, 1.05),
            "damping_distribution_params": (0.90, 1.10),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    randomize_wheel_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES),
            "damping_distribution_params": (0.90, 1.10),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    randomize_arm_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES),
            "stiffness_distribution_params": (0.95, 1.05),
            "damping_distribution_params": (0.95, 1.05),
            "operation": "scale",
            "distribution": "uniform",
        },
    )


@configclass
class UnitreeGo2WArmRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Rough-road velocity-tracking configuration for Unitree Go2W with Piper arm."""

    ###### Robot names #######

    base_link_name = "base"
    foot_link_name = r".*_foot"
    leg_joint_names = LEG_JOINT_NAMES
    arm_joint_names = ARM_JOINT_NAMES
    wheel_joint_names = WHEEL_JOINT_NAMES

    events: UnitreeGo2WArmEventCfg = UnitreeGo2WArmEventCfg()

    def __post_init__(self):
        super().__post_init__()

        ###### Scene and sensors #######

        self.scene.robot = UNITREE_Go2WArm_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # 机器人周围的一张局部高程图
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        # 扫描 base 下方很小的区域
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.update_period = self.decimation * self.sim.dt

        ###### Observations #######

        # The policy uses the deployable mocap-derived base linear velocity.
        # The critic additionally keeps terrain heights as privileged observations.
        self.observations.policy.base_lin_vel.scale = 2.0
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        self.observations.policy.velocity_commands.scale = (2.0, 2.0, 0.25)
        self.observations.policy.height_scan = None

        self.observations.critic.base_lin_vel.scale = 2.0
        self.observations.critic.base_ang_vel.scale = 0.25
        self.observations.critic.joint_pos.scale = 1.0
        self.observations.critic.joint_vel.scale = 0.05
        # 测量速度和速度指令使用同一个缩放比例，便于计算误差
        self.observations.critic.velocity_commands.scale = (2.0, 2.0, 0.25)

        ###### Actions #######

        # Normalized policy actions. A 15 rad/s wheel range covers 1 m/s
        # rolling commands and leaves headroom for differential steering.
        self.actions.joint_pos.scale = {r".*_hip_joint": 0.125, r"^(?!.*_hip_joint).*": 0.25}
        self.actions.joint_pos.clip = {r".*": (-100.0, 100.0)}
        self.actions.joint_pos.low_pass_cutoff_hz = 4.0
        self.actions.joint_vel.scale = 15.0
        self.actions.joint_vel.clip = {r".*": (-100.0, 100.0)}

        ###### Events #######

        # Preserve the inherited asset_cfg and only replace the randomization
        # ranges. Roll, pitch, and height start from the valid rolling stance.
        self.events.randomize_reset_base.params["pose_range"] = {
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (-math.pi, math.pi),
        }
        self.events.randomize_reset_base.params["velocity_range"] = {}

        ###### Rewards: velocity tracking #######

        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.5
        self.rewards.track_ang_vel_z_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.params["std"] = 0.5

        ###### Rewards: base stability #######

        self.rewards.base_height_l2.weight = -5.0
        self.rewards.base_height_l2.params["target_height"] = 0.40
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.flat_orientation_l2.weight = -2.0

        ###### Rewards: leg regularization #######

        self.rewards.joint_torques_l2.weight = -2.5e-5
        self.rewards.joint_torques_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_vel_l2.weight = -0.005
        self.rewards.joint_vel_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.hip_joint_vel_l2.weight = -0.20
        self.rewards.hip_joint_vel_l2.params["asset_cfg"].joint_names = r".*_hip_joint"
        self.rewards.joint_acc_l2.weight = -2.5e-7
        self.rewards.joint_acc_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_deviation_l1.weight = -0.5
        self.rewards.joint_deviation_l1.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.hip_joint_deviation_l1.weight = -1.0
        self.rewards.hip_joint_deviation_l1.params["asset_cfg"].joint_names = r".*_hip_joint"
        self.rewards.joint_pos_limits.weight = -5.0
        self.rewards.joint_pos_limits.params["asset_cfg"].joint_names = self.leg_joint_names

        ###### Rewards: wheel regularization #######

        self.rewards.joint_torques_wheel_l2.weight = -1.0e-5
        self.rewards.joint_torques_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_vel_wheel_l2.weight = 0.0
        self.rewards.joint_vel_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_acc_wheel_l2.weight = -2.5e-9
        self.rewards.joint_acc_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_vel_limits.weight = -1.0
        self.rewards.joint_vel_limits.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_vel_limits.params["soft_ratio"] = 1.0

        self.rewards.wheel_same_side_sync_l2.weight = -0.5
        self.rewards.wheel_same_side_sync_l2.params["velocity_scale"] = 15.0

        self.rewards.wheel_diff_drive_tracking_exp.weight = 0.0
        self.rewards.wheel_diff_drive_tracking_exp.params["wheel_radius"] = 0.086
        self.rewards.wheel_diff_drive_tracking_exp.params["track_width"] = 0.380
        self.rewards.wheel_diff_drive_tracking_exp.params["linear_std"] = 0.5
        self.rewards.wheel_diff_drive_tracking_exp.params["angular_std"] = 0.5

        self.rewards.wheel_stance_xy_l2.weight = -0.40
        self.rewards.wheel_stance_xy_l2.params["position_scale"] = 0.05

        self.rewards.wheel_relative_xy_vel_l2.weight = 0.0
        self.rewards.wheel_relative_xy_vel_l2.params["velocity_scale"] = 0.1

        self.rewards.wheel_contact_loss.weight = -3.0
        self.rewards.wheel_contact_loss.params["sensor_cfg"].body_names = self.foot_link_name
        self.rewards.wheel_contact_loss.params["grace_period"] = 0.02

        self.rewards.wheel_spin_without_cmd.weight = -0.5
        self.rewards.wheel_spin_without_cmd.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.wheel_spin_without_cmd.params["linear_command_threshold"] = 0.05
        self.rewards.wheel_spin_without_cmd.params["angular_command_threshold"] = 0.05
        self.rewards.wheel_spin_without_cmd.params["linear_velocity_scale"] = 0.1
        self.rewards.wheel_spin_without_cmd.params["angular_velocity_scale"] = 0.2

        ###### Rewards: action regularization #######

        self.rewards.action_rate_l2.weight = -0.01

        ###### Rewards: contact constraints #######

        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = r"^(?!.*_foot$).*"
        self.rewards.undesired_contacts.params["threshold"] = 1.0

        ###### Rewards: termination #######

        self.rewards.is_terminated.weight = -200.0

        ###### Terminations #######

        self.terminations.bad_orientation.params["limit_angle"] = 0.6
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [
            self.base_link_name,
            r".*_hip",
        ]
        self.terminations.illegal_contact.params["threshold"] = 1.0

        ###### Commands #######

        # Direct planar twist commands for car-like differential steering.
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.rel_standing_envs = 0.1
        self.commands.base_velocity.debug_vis = False

        # Keep the arm fixed by default. Staged training directly overrides
        # these final command fields through Hydra after __post_init__ runs.
        self.commands.arm_joint_trajectory.fixed_default = True
        self.commands.arm_joint_trajectory.init_range = 0.0
        self.commands.arm_joint_trajectory.debug_vis = False

        ###### Zero-weight reward cleanup #######

        # Delay cleanup for subclasses so the flat configuration can still
        # override inherited reward terms before removing zero-weight entries.
        if self.__class__.__name__ == "UnitreeGo2WArmRoughEnvCfg":
            self.disable_zero_weight_rewards()
