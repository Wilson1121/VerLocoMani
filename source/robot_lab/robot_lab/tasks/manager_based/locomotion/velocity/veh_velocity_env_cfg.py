# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp

##
# Pre-defined configs
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip


# Joint ordering shared by actions, observations, commands, events, and rewards.
LEG_JOINT_NAMES = [
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
]
ARM_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
WHEEL_JOINT_NAMES = ["FL_foot_joint", "FR_foot_joint", "RL_foot_joint", "RR_foot_joint"]
POSITION_OBS_JOINT_NAMES = LEG_JOINT_NAMES + ARM_JOINT_NAMES
VELOCITY_OBS_JOINT_NAMES = POSITION_OBS_JOINT_NAMES + WHEEL_JOINT_NAMES


##
# Scene definition
##


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = MISSING
    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    height_scanner_base = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=(0.1, 0.1)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformThresholdVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformThresholdVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), 
            lin_vel_y=(-1.0, 1.0), 
            ang_vel_z=(-1.0, 1.0), 
            heading=(-math.pi, math.pi)
        ),
    )

    arm_joint_trajectory = mdp.ArmJointTrajectoryCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0e6, 1.0e6),
        trajectory_time=(8.0, 12.0),
        hold_time=(1.0, 2.0),
        fixed_default=False,
        init_range=0.9,
        debug_vis=False,
        joint_names=ARM_JOINT_NAMES,
    )


@configclass
class ActionsCfg:
    """Policy leg/wheel actions with command-driven arm position targets."""

    joint_pos = mdp.CommandArmPolicyLegJointPositionActionCfg(
        asset_name="robot",
        joint_names=LEG_JOINT_NAMES,
        arm_joint_names=ARM_JOINT_NAMES,
        arm_command_name="arm_joint_trajectory",
        scale={".*_hip_joint": 0.125, "^(?!.*_hip_joint).*": 0.25},
        use_default_offset=True,
        clip={".*": (-100.0, 100.0)},
        preserve_order=True,
    )

    joint_vel = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=WHEEL_JOINT_NAMES,
        scale=5.0,
        use_default_offset=True,
        clip={".*": (-100.0, 100.0)},
        preserve_order=True,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        # Projected gravity, 3 Dim.
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        # Base linear velocity, 3 Dim.
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        # Base angular velocity, 3 Dim.
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        # Joint position relative to default, 18 Dim.
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=POSITION_OBS_JOINT_NAMES,
                    preserve_order=True,
                )
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        # Joint velocity relative observation, 22 Dim.
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=VELOCITY_OBS_JOINT_NAMES, 
                    preserve_order=True,
                )
            },
            noise=Unoise(n_min=-0.2, n_max=0.2),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        # Base velocity command, 3 Dim.
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        # Arm joint trajectory command, 6 Dim.
        arm_joint_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "arm_joint_trajectory"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        # Last policy action for leg and wheel joints, 16 Dim.
        actions = ObsTerm(
            func=mdp.last_action,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        # Terrain height scan.
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
            scale=1.0,
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        # observation terms (order preserved)
        # Projected gravity, 3 Dim.
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        # Base linear velocity, 3 Dim.
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        # Base angular velocity, 3 Dim.
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        # Joint position relative to default, 18 Dim.
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=POSITION_OBS_JOINT_NAMES,
                    preserve_order=True,
                )
            },
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        # Joint velocity relative observation, 22 Dim.
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=VELOCITY_OBS_JOINT_NAMES,
                    preserve_order=True,
                )
            },
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        # Base velocity command, 3 Dim.
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        # Arm joint trajectory command, 6 Dim.
        arm_joint_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "arm_joint_trajectory"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        # Last policy action for leg and wheel joints, 16 Dim.
        actions = ObsTerm(
            func=mdp.last_action,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        # Terrain height scan.
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
            scale=1.0,
        )
        # joint_effort = ObsTerm(
        #     func=mdp.joint_effort,
        #     clip=(-100, 100),
        #     scale=0.01,
        # )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    # randomize_rigid_body_material = EventTerm(
    #     func=mdp.randomize_rigid_body_material,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
    #         "static_friction_range": (0.3, 1.0),
    #         "dynamic_friction_range": (0.3, 0.8),
    #         "restitution_range": (0.0, 0.5),
    #         "num_buckets": 64,
    #     },
    # )

    # randomize_rigid_body_mass_base = EventTerm(
    #     func=mdp.randomize_rigid_body_mass,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=""),
    #         "mass_distribution_params": (-1.0, 3.0),
    #         "operation": "add",
    #         "recompute_inertia": True,
    #     },
    # )

    # randomize_rigid_body_mass_others = EventTerm(
    #     func=mdp.randomize_rigid_body_mass,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
    #         "mass_distribution_params": (0.7, 1.3),
    #         "operation": "scale",
    #         "recompute_inertia": True,
    #     },
    # )

    # Skip: inertia updated via mass randomization by setting recompute_inertia=True
    # randomize_rigid_body_inertia = EventTerm(
    #     func=mdp.randomize_rigid_body_inertia,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
    #         "inertia_distribution_params": (0.5, 1.5),
    #         "operation": "scale",
    #     },
    # )

    # randomize_com_positions = EventTerm(
    #     func=mdp.randomize_rigid_body_com,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
    #         "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
    #     },
    # )

    # reset
    # randomize_apply_external_force_torque = EventTerm(
    #     func=mdp.apply_external_force_torque,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=""),
    #         "force_range": (-10.0, 10.0),
    #         "torque_range": (-10.0, 10.0),
    #     },
    # )

    randomize_reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {},
        },
    )

    randomize_reset_leg_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES),
            "position_range": (-0.0, 0.0),
            "velocity_range": (-0.0, 0.0),
        },
    )

    randomize_reset_arm_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES),
            "position_range": (-0.0, 0.0),
            "velocity_range": (-0.0, 0.0),
        },
    )

    randomize_reset_wheel_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES),
            "position_range": (-0.0, 0.0),
            "velocity_range": (-0.0, 0.0),
        },
    )

    # randomize_leg_wheel_actuator_gains = EventTerm(
    #     func=mdp.randomize_actuator_gains,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES + WHEEL_JOINT_NAMES),
    #         "stiffness_distribution_params": (0.5, 2.0),
    #         "damping_distribution_params": (0.5, 2.0),
    #         "operation": "scale",
    #         "distribution": "uniform",
    #     },
    # )


    # interval
    # randomize_push_robot = EventTerm(
    #     func=mdp.push_by_setting_velocity,
    #     mode="interval",
    #     interval_range_s=(10.0, 15.0),
    #     params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    # )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # 线速度跟踪奖励项
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=0.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    # 角速度跟踪奖励项
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    # 惩罚机体高度与目标高度之间的偏差
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("height_scanner_base"),
            "target_height": 0.45,
        },
    )
    # 惩罚机体沿竖直方向的线速度，减少上下跳动和振荡。
    lin_vel_z_l2 = RewTerm(
        func=mdp.lin_vel_z_l2,
        weight=0.0,
    )
    # 惩罚机体绕横滚轴和俯仰轴的角速度，提高姿态稳定性。
    ang_vel_xy_l2 = RewTerm(
        func=mdp.ang_vel_xy_l2,
        weight=0.0,
    )
    # 腿部关节力矩正则项。
    joint_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=LEG_JOINT_NAMES,
                preserve_order=True,
            )
        },
    )
    # 腿部关节速度正则项。
    joint_vel_l2 = RewTerm(
        func=mdp.joint_vel_l2,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=LEG_JOINT_NAMES,
                preserve_order=True,
            )
        },
    )
    # 腿部关节加速度正则项。
    joint_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=LEG_JOINT_NAMES,
                preserve_order=True,
            )
        },
    )
    # 轮子关节力矩正则项。
    joint_torques_wheel_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=WHEEL_JOINT_NAMES,
                preserve_order=True,
            )
        },
    )
    # 轮子关节速度正则项。
    joint_vel_wheel_l2 = RewTerm(
        func=mdp.joint_vel_l2,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=WHEEL_JOINT_NAMES,
                preserve_order=True,
            )
        },
    )
    # 轮子关节加速度正则项。
    joint_acc_wheel_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=WHEEL_JOINT_NAMES,
                preserve_order=True,
            )
        },
    )
    # 惩罚相邻控制周期之间的动作变化，提升控制平滑性。
    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2,
        weight=0.0,
    )
    

    # General
    # 根据非超时终止状态施加奖励或惩罚，通常用于惩罚机器人提前失败。
    is_terminated = RewTerm(func=mdp.is_terminated, weight=0.0)

    # Root penalties
    # 惩罚机体沿竖直方向的线速度，减少上下跳动和振荡。
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=0.0)
    # 惩罚机体绕横滚轴和俯仰轴的角速度，提高姿态稳定性。
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=0.0)
    # 惩罚机体相对水平姿态的倾斜，促使机身保持水平。
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.0)
    # 惩罚机体高度与目标高度之间的偏差。
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=""),
            "sensor_cfg": SceneEntityCfg("height_scanner_base"),
            "target_height": 0.0,
        },
    )
    # 惩罚指定刚体的线加速度，抑制机身剧烈冲击和抖动。
    body_lin_acc_l2 = RewTerm(
        func=mdp.body_lin_acc_l2,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="")},
    )

    # Joint penalties
    # 惩罚指定关节的驱动力矩，降低执行器负载和能耗。
    joint_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}
    )
    # 惩罚指定关节的转动速度，抑制过快的关节运动。
    joint_vel_l2 = RewTerm(
        func=mdp.joint_vel_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}
    )
    # 惩罚指定关节的加速度，减少高频运动和机械冲击。
    joint_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}
    )
    # 单独惩罚轮子关节的驱动力矩，约束轮毂电机负载。
    joint_torques_wheel_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES)},
    )
    # 单独惩罚轮子关节的转速，避免不必要的高速空转。
    joint_vel_wheel_l2 = RewTerm(
        func=mdp.joint_vel_l2,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES)},
    )
    # 单独惩罚轮子关节的角加速度，使轮速变化更加平滑。
    joint_acc_wheel_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES)},
    )

    # 动态创建指定关节相对默认姿态的 L1 偏差奖励项。
    def create_joint_deviation_l1_rewterm(self, attr_name, weight, joint_names_pattern):
        rew_term = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=weight,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=joint_names_pattern)},
        )
        setattr(self, attr_name, rew_term)

    # 惩罚关节位置超出软位置限位的程度，降低触碰机械限位的风险。
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}
    )
    # 惩罚关节速度超过软速度限位的程度。
    joint_vel_limits = RewTerm(
        func=mdp.joint_vel_limits,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*"), "soft_ratio": 1.0},
    )
    # 惩罚关节力矩与关节速度乘积的绝对值，降低机械功率消耗。
    joint_power = RewTerm(
        func=mdp.joint_power,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        },
    )

    # 速度命令接近零时惩罚关节偏离默认姿态，促使机器人稳定静止。
    stand_still = RewTerm(
        func=mdp.stand_still,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        },
    )

    # 惩罚关节偏离默认姿态，并在静止状态下提高惩罚强度。
    joint_pos_penalty = RewTerm(
        func=mdp.joint_pos_penalty,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.5,
            "command_threshold": 0.1,
        },
    )

    # 运动时惩罚离地轮空转，静止时惩罚所有轮子转动。
    wheel_vel_penalty = RewTerm(
        func=mdp.wheel_vel_penalty,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=""),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
            "command_name": "base_velocity",
            "velocity_threshold": 0.5,
            "command_threshold": 0.1,
        },
    )

    # 惩罚成对对称关节的位置差异，促进对称的腿部姿态。
    joint_mirror = RewTerm(
        func=mdp.joint_mirror,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mirror_joints": [["FR.*", "RL.*"], ["FL.*", "RR.*"]],
        },
    )

    # 惩罚成对对称关节的动作幅值差异，促进对称控制输出。
    action_mirror = RewTerm(
        func=mdp.action_mirror,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mirror_joints": [["FR.*", "RL.*"], ["FL.*", "RR.*"]],
        },
    )

    # 惩罚同一关节组内动作不一致，促进多条腿同步运动。
    action_sync = RewTerm(
        func=mdp.action_sync,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "joint_groups": [
                ["FR_hip_joint", "FL_hip_joint", "RL_hip_joint", "RR_hip_joint"],
                ["FR_thigh_joint", "FL_thigh_joint", "RL_thigh_joint", "RR_thigh_joint"],
                ["FR_calf_joint", "FL_calf_joint", "RL_calf_joint", "RR_calf_joint"],
            ],
        },
    )

    # Action penalties
    # 惩罚执行器实际力矩超过软力矩限位的程度。
    applied_torque_limits = RewTerm(
        func=mdp.applied_torque_limits,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    # 惩罚相邻控制周期之间的动作变化，提升控制平滑性。
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=0.0)
    # smoothness_1 = RewTerm(func=mdp.smoothness_1, weight=0.0)  # Same as action_rate_l2
    # smoothness_2 = RewTerm(func=mdp.smoothness_2, weight=0.0)  # Unvaliable now

    # Contact sensor
    # 惩罚指定身体部位发生超过阈值的非期望接触。
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
            "threshold": 1.0,
        },
    )
    # 惩罚指定接触部位的接触力超过给定阈值。
    contact_forces = RewTerm(
        func=mdp.contact_forces,
        weight=0.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=""), "threshold": 100.0},
    )

    # Velocity-tracking rewards
    # 奖励机体在水平面内跟踪目标线速度。
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=0.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    # 奖励机体跟踪绕竖直轴的目标角速度。
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    # Others
    # 运动命令存在时奖励合适的单腿支撑和腾空持续时间。
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "threshold": 0.5,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
        },
    )

    # 度量各足腾空与接触时间的方差，通常用于惩罚步态时序不一致。
    feet_air_time_variance = RewTerm(
        func=mdp.feet_air_time_variance_penalty,
        weight=0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="")},
    )

    # 根据指定足端配对奖励同步与交替接触时序，塑造目标步态。
    feet_gait = RewTerm(
        func=mdp.GaitReward,
        weight=0.0,
        params={
            "std": math.sqrt(0.5),
            "command_name": "base_velocity",
            "max_err": 0.2,
            "velocity_threshold": 0.5,
            "command_threshold": 0.1,
            "synced_feet_pair_names": (("", ""), ("", "")),
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces"),
        },
    )

    # 度量首次接触足数量与期望数量是否不一致，通常作为接触数量惩罚。
    feet_contact = RewTerm(
        func=mdp.feet_contact,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
            "command_name": "base_velocity",
            "expect_contact_num": 2,
        },
    )

    # 无运动命令时奖励足端重新接触地面，帮助机器人保持稳定支撑。
    feet_contact_without_cmd = RewTerm(
        func=mdp.feet_contact_without_cmd,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
            "command_name": "base_velocity",
        },
    )

    # 惩罚足端侧向碰撞竖直障碍物等绊倒事件。
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
        },
    )

    # 惩罚足端接触地面时的切向滑动速度。
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
            "asset_cfg": SceneEntityCfg("robot", body_names=""),
        },
    )

    # 运动时惩罚足端世界坐标高度偏离目标摆动高度。
    feet_height = RewTerm(
        func=mdp.feet_height,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=""),
            "tanh_mult": 2.0,
            "target_height": 0.05,
            "command_name": "base_velocity",
        },
    )

    # 运动时惩罚足端在机体坐标系中的高度偏离目标值。
    feet_height_body = RewTerm(
        func=mdp.feet_height_body,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=""),
            "tanh_mult": 2.0,
            "target_height": -0.3,
            "command_name": "base_velocity",
        },
    )

    # 奖励足端横向间距接近期望站立宽度。
    feet_distance_y_exp = RewTerm(
        func=mdp.feet_distance_y_exp,
        weight=0.0,
        params={
            "std": math.sqrt(0.25),
            "asset_cfg": SceneEntityCfg("robot", body_names=""),
            "stance_width": float,
        },
    )

    # feet_distance_xy_exp = RewTerm(
    #     func=mdp.feet_distance_xy_exp,
    #     weight=0.0,
    #     params={
    #         "std": math.sqrt(0.25),
    #         "asset_cfg": SceneEntityCfg("robot", body_names=""),
    #         "stance_length": float,
    #         "stance_width": float,
    #     },
    # )

    # 奖励机体保持竖直朝上的姿态，降低翻倒倾向。
    upward = RewTerm(func=mdp.upward, weight=0.0)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # MDP terminations
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # command_resample
    terrain_out_of_bounds = DoneTerm(
        func=mdp.terrain_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 3.0},
        time_out=True,
    )

    # Contact sensor
    illegal_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=""), "threshold": 1.0},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)

    command_levels_lin_vel = CurrTerm(
        func=mdp.command_levels_lin_vel,
        params={
            "reward_term_name": "track_lin_vel_xy_exp",
            "range_multiplier": (0.1, 1.0),
        },
    )

    command_levels_ang_vel = CurrTerm(
        func=mdp.command_levels_ang_vel,
        params={
            "reward_term_name": "track_ang_vel_z_exp",
            "range_multiplier": (0.1, 1.0),
        },
    )


##
# Environment configuration
##


@configclass
class LocomotionVelocityRoughEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False

    def disable_zero_weight_rewards(self):
        """If the weight of rewards is 0, set rewards to None"""
        for attr in dir(self.rewards):
            if not attr.startswith("__"):
                reward_attr = getattr(self.rewards, attr)
                if not callable(reward_attr) and reward_attr.weight == 0:
                    setattr(self.rewards, attr, None)
