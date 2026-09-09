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
WHEEL_BODY_NAMES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
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
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
        force_threshold=1.0,
    )
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
        rel_standing_envs=0.1,  # 略微增加静止站立比例
        rel_heading_envs=0.0,   # 暂不设计航向
        heading_command=False,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformThresholdVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-0.0, 0.0),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
    )

    arm_joint_trajectory = mdp.ArmJointTrajectoryCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0e6, 1.0e6),
        trajectory_time=(8.0, 12.0),
        hold_time=(1.0, 2.0),
        fixed_default=True,
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

    ######################## Velocity-tracking rewards ##########################
    # 线速度跟踪奖励项。
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=0.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    # 角速度跟踪奖励项。
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    ############################### Base penalties ##############################
    # 惩罚机体高度与目标高度之间的偏差。
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
    # 惩罚机体静态横滚和俯仰倾斜。
    flat_orientation_l2 = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=0.0,
    )

    ############################## Leg-joint penalties ##############################
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
    # 额外惩罚髋关节侧向运动速度，抑制接地行驶时的腿部横摆。
    hip_joint_vel_l2 = RewTerm(
        func=mdp.joint_vel_l2,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=r".*_hip_joint",
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
    # 惩罚腿部关节偏离默认姿态，避免出现极端腿型。
    joint_deviation_l1 = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=LEG_JOINT_NAMES,
                preserve_order=True,
            )
        },
    )
    # 额外惩罚髋关节偏离默认角度，抑制持续性的腿部侧向偏移。
    hip_joint_deviation_l1 = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=r".*_hip_joint",
                preserve_order=True,
            )
        },
    )
    # 惩罚腿部关节超出软位置限位的部分。
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=LEG_JOINT_NAMES,
                preserve_order=True,
            )
        },
    )

    ############################## Wheel-joint penalties ##############################
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
    # 惩罚轮子关节超出软速度限位的部分。
    joint_vel_limits = RewTerm(
        func=mdp.joint_vel_limits,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=WHEEL_JOINT_NAMES,
                preserve_order=True,
            ),
            "soft_ratio": 1.0,
        },
    )
    # 惩罚同侧前后轮的速度差，减少轮间拖拽并保持差速底盘运动一致性。
    wheel_same_side_sync_l2 = RewTerm(
        func=mdp.wheel_same_side_sync_l2,
        weight=0.0,
        params={
            "velocity_scale": 15.0,
            "left_wheel_cfg": SceneEntityCfg(
                "robot",
                joint_names=["FL_foot_joint", "RL_foot_joint"],
                preserve_order=True,
            ),
            "right_wheel_cfg": SceneEntityCfg(
                "robot",
                joint_names=["FR_foot_joint", "RR_foot_joint"],
                preserve_order=True,
            ),
        },
    )
    # 奖励四轮速度满足差速运动学，并跟踪底盘前向速度和偏航角速度指令。
    wheel_diff_drive_tracking_exp = RewTerm(
        func=mdp.wheel_diff_drive_tracking_exp,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "wheel_radius": 0.086,
            "track_width": 0.380,
            "linear_std": 0.5,
            "angular_std": 0.5,
            "left_wheel_cfg": SceneEntityCfg(
                "robot",
                joint_names=["FL_foot_joint", "RL_foot_joint"],
                preserve_order=True,
            ),
            "right_wheel_cfg": SceneEntityCfg(
                "robot",
                joint_names=["FR_foot_joint", "RR_foot_joint"],
                preserve_order=True,
            ),
        },
    )
    # 惩罚四个轮子在机体水平面内偏离默认轮式构型，防止策略通过挪动轮子辅助转向。
    wheel_stance_xy_l2 = RewTerm(
        func=mdp.wheel_stance_xy_l2,
        weight=0.0,
        params={
            "position_scale": 0.05,
            "target_positions": [
                (0.186, 0.142),
                (0.186, -0.142),
                (-0.200, 0.142),
                (-0.200, -0.142),
            ],
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=WHEEL_BODY_NAMES,
                preserve_order=True,
            ),
        },
    )
    # 惩罚没有与地面保持接触的轮子数量。
    wheel_contact_loss = RewTerm(
        func=mdp.wheel_contact_loss,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=r".*_foot"),
            "grace_period": 0.02,
        },
    )
    # 底盘速度指令接近零时，惩罚轮子继续转动。
    wheel_spin_without_cmd = RewTerm(
        func=mdp.wheel_spin_without_cmd,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "linear_command_threshold": 0.05,
            "angular_command_threshold": 0.05,
            "linear_velocity_scale": 0.1,
            "angular_velocity_scale": 0.2,
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=WHEEL_JOINT_NAMES,
                preserve_order=True,
            ),
        },
    )

    ################################## Action penalties ##################################
    # 惩罚相邻控制周期之间的动作变化，提升控制平滑性。
    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2,
        weight=0.0,
    )

    ################################# Contact penalties ##################################
    # 惩罚轮子以外的机体部件与环境发生非期望接触。
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=r"^(?!.*_foot$).*"),
            "threshold": 1.0,
        },
    )

    ################################ Termination penalty ################################
    # 根据非超时终止状态施加失败惩罚。
    is_terminated = RewTerm(
        func=mdp.is_terminated,
        weight=0.0,
    )


@configclass
class TerminationsCfg:
    """Termination terms for the Go2WArm rolling task."""

    # 达到最大 episode 时长，属于正常时间截断而不是控制失败。
    time_out = DoneTerm(
        func=mdp.time_out,
        time_out=True,
    )

    # 机器人驶出有限地形边界时截断；对于无限平面地形，该项始终不触发。
    terrain_out_of_bounds = DoneTerm(
        func=mdp.terrain_out_of_bounds,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "distance_buffer": 3.0,
        },
        time_out=True,
    )

    # 机身倾角过大时提前终止，避免继续采集明显失稳或翻倒后的状态。
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "limit_angle": 0.8,
        },
    )

    # base 或 hip 发生地面接触时终止；其他非轮体接触由奖励项进行软约束。
    illegal_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[r"base", r".*_hip"],
            ),
            "threshold": 1.0,
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)

    # command_levels_lin_vel = CurrTerm(
    #     func=mdp.command_levels_lin_vel,
    #     params={
    #         "reward_term_name": "track_lin_vel_xy_exp",
    #         "range_multiplier": (0.1, 1.0),
    #     },
    # )

    # command_levels_ang_vel = CurrTerm(
    #     func=mdp.command_levels_ang_vel,
    #     params={
    #         "reward_term_name": "track_ang_vel_z_exp",
    #         "range_multiplier": (0.1, 1.0),
    #     },
    # )


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
