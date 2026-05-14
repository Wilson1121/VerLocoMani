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
from robot_lab.tasks.manager_based.locomotion.velocity.mdp import observations as mdp_obs

##
# Pre-defined configs
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip


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
    base_pose = mdp.BasePoseCommandCfg(
        asset_name="robot",
        torso_body_name="base",
        resampling_time_range=(10.0, 10.0),
        debug_vis=False,
        ranges=mdp.BasePoseCommandCfg.Ranges(
            roll=(-0.3, 0.3),
            pitch=(-0.3, 0.3),
            height=(0.25, 0.5),
        ),
    )
    arm_joint_trajectory = mdp.ArmJointTrajectoryCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0e6, 1.0e6),
        trajectory_time=(6.0, 8.0),
        hold_time=(1.0, 2.0),
        fixed_default=True,
        debug_vis=False,
        joint_names=[
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
        ],
    )
    feet_swing_height = mdp.DesiredFeetSwingHeightCommandCfg(
        resampling_time_range=(10.0, 10.0),
        max_height=0.08,
        gait_frequency=2.0,
        phase_offsets=(0.0, 0.5, 0.5, 0.0),
        clip_to_positive=True,
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.CommandArmPolicyLegJointPositionActionCfg(
        asset_name="robot",
        joint_names=[
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
        ],
        arm_joint_names=[
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
        ],
        arm_command_name="arm_joint_trajectory",
        scale=0.4,
        use_default_offset=True,
        clip=None,
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
                    joint_names=[
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
                        "joint1",
                        "joint2",
                        "joint3",
                        "joint4",
                        "joint5",
                        "joint6",
                    ],
                    preserve_order=True,
                )
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        # Joint velocity relative observation, 18 Dim.
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
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
                        "joint1",
                        "joint2",
                        "joint3",
                        "joint4",
                        "joint5",
                        "joint6",
                    ],
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
        # Torso base pose command [roll, pitch, height], 3 Dim.
        base_pose_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_pose"},
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
        # Desired feet swing height command for [FL, FR, RL, RR], 4 Dim.
        feet_swing_height_command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "feet_swing_height"},
            clip=(0.0, 1.0),
            scale=1.0,
        )
        # Last policy action for leg joints, 12 Dim.
        actions = ObsTerm(
            func=mdp.last_action,
            clip=(-100.0, 100.0),
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
                    joint_names=[
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
                        "joint1",
                        "joint2",
                        "joint3",
                        "joint4",
                        "joint5",
                        "joint6",
                    ],
                    preserve_order=True,
                )
            },
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        # Joint velocity relative observation, 18 Dim.
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
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
                        "joint1",
                        "joint2",
                        "joint3",
                        "joint4",
                        "joint5",
                        "joint6",
                    ],
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
        # Torso base pose command [roll, pitch, height], 3 Dim.
        base_pose_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_pose"},
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
        # Desired feet swing height command for [FL, FR, RL, RR], 4 Dim.
        feet_swing_height_command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "feet_swing_height"},
            clip=(0.0, 1.0),
            scale=1.0,
        )
        # Last policy action for leg joints, 12 Dim.
        actions = ObsTerm(
            func=mdp.last_action,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        # Feet binary contact state for [FL, FR, RL, RR], 4 Dim.
        feet_contact_state = ObsTerm(
            func=mdp.feet_contact_state,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
                ),
                "threshold": 1.0,
            },
            clip=(0.0, 1.0),
            scale=1.0,
        )
        # Current feet air time for [FL, FR, RL, RR], 4 Dim.
        feet_air_time = ObsTerm(
            func=mdp_obs.feet_air_time,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
                )
            },
            clip=(0.0, 1.0),
            scale=1.0,
        )
        # # Terrain height scan around the base, 160 Dim.
        # height_scan = ObsTerm(
        #     func=mdp.height_scan,
        #     params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        #     clip=(-1.0, 1.0),
        #     scale=1.0,
        # )
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

    # reset
    randomize_reset_base_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            # reset_root_state_uniform samples offsets from the default root pose.
            # Go2Arm default root z is 0.4, so this gives absolute base height U(0.2, 1.3).
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



@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # Velocity-tracking rewards
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=0.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_base_orientation_exp = RewTerm(
        func=mdp.track_base_orientation_exp,
        weight=0.0,
        params={"command_name": "base_pose", "std": math.sqrt(0.15)},
    )
    track_base_height_exp = RewTerm(
        func=mdp.track_base_height_exp,
        weight=0.0,
        params={"command_name": "base_pose", "std": math.sqrt(0.05)},
    )
    base_lin_vel_z_exp = RewTerm(
        func=mdp.base_lin_vel_z_exp,
        weight=0.0,
        params={
            # 抑制 base 在机体系 z 方向的线速度；std**2 是指数核分母。
            "std": math.sqrt(0.2),
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
        },
    )
    base_ang_vel_xy_exp = RewTerm(
        func=mdp.base_ang_vel_xy_exp,
        weight=0.0,
        params={
            # 抑制 base 的 roll/pitch 角速度；std**2 是指数核分母。
            "std": math.sqrt(0.2),
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
        },
    )

    # 关节与动作正则项。
    joint_torques_exp = RewTerm(
        func=mdp.joint_torques_exp,
        weight=0.0,
        params={
            # 腿部策略控制关节的力矩 exp 正则项，不包含由 command 驱动的机械臂关节。
            "std": math.sqrt(1.0),
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
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
                ],
                preserve_order=True,
            ),
        },
    )
    joint_vel_exp = RewTerm(
        func=mdp.joint_vel_exp,
        weight=0.0,
        params={
            # 腿部策略控制关节的速度 exp 正则项。
            "std": math.sqrt(1.0),
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
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
                ],
                preserve_order=True,
            ),
        },
    )
    action_rate_exp = RewTerm(
        func=mdp.action_rate_exp,
        weight=0.0,
        params={"std": math.sqrt(1.0)},
    )

    # 接触时序奖励。
    feet_contact = RewTerm(
        func=mdp.feet_contact_schedule,
        weight=0.0,
        params={
            # 期望摆动高度命令。接近 0 表示支撑相，正高度表示摆动相。
            "command_name": "feet_swing_height",
            # 接触传感器中的足端顺序，必须与 feet_swing_height 命令顺序一致：[FL, FR, RL, RR]。
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
                preserve_order=True,
            ),
            # 机器人足端刚体，用于读取足端高度和速度；顺序必须与 sensor_cfg 一致。
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
                preserve_order=True,
            ),
            # 躯干刚体，用于构造 yaw 对齐坐标系来计算足端水平速度。
            "torso_body_cfg": SceneEntityCfg("robot", body_names="base"),
            # 摆动脚接触力项 Phi(F_f, 1.0) 的高斯分母。
            "force_variance": 1.0,
            # 期望足高与当前足高误差项 Phi(h_hat_z - h_z, 0.05) 的高斯分母。
            "height_variance": 0.05,
            # 支撑脚水平速度项 Phi(v_fxy, 0.01) 的高斯分母。
            "vel_variance": 0.01,
            # 判断支撑脚实际接触的竖直接触力阈值。
            "contact_force_threshold": 1.0,
            # 期望足高小于等于该值时，视为支撑相命令。
            "height_contact_epsilon": 1.0e-4,
        },
    )
    feet_air_time_variance = RewTerm(
        func=mdp.FeetAirTimeVarianceReward,
        weight=0.0,
        params={
            # 需要统计最近腾空/接触持续时间的足端；保留顺序以保持一致性。
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
                preserve_order=True,
            ),
            # 用于计算 Var(t_air) + Var(t_contact) 的最近模式持续时间数量，即 t-2..t。
            "history_len": 3,
        },
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_on_contact,
        weight=0.0,
        params={
            # 用于检测首次触地并读取该次触地前腾空持续时间的足端。
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
                preserve_order=True,
            ),
        },
    )

    is_alive = RewTerm(
        func=mdp.is_alive,
        weight=0.0
    )
    is_terminated = RewTerm(
        func=mdp.is_terminated,
        weight=0.0
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # MDP terminations
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # 基座翻倒终止：base 倾角超过阈值即终止。
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base"), "limit_angle": 1.0},
    )

    # 只允许足端接触地面：除四个足端外，任意刚体接触力超过阈值即终止。
    illegal_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[r"^(?!.*(?:FL_foot|FR_foot|RL_foot|RR_foot)$).+"],
            ),
            "threshold": 1.0,
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""


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
