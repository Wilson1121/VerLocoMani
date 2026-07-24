# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import math

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp
from robot_lab.tasks.manager_based.locomotion.velocity.vlm_velocity_env_cfg import LocomotionVelocityRoughEnvCfg

##
# Pre-defined configs
##
# # use cloud assets
# from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip
# use local assets
from robot_lab.assets.unitree import UNITREE_Go2Arm_CFG  # isort: skip


GO2ARM_BLIND_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    curriculum=True,
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.20),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.35,
            noise_range=(0.0, 0.05),
            noise_step=0.01,
            border_width=0.25,
        ),
        "pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.125,
            slope_range=(0.0, 0.18),
            platform_width=2.0,
            border_width=0.25,
        ),
        "pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.125,
            slope_range=(0.0, 0.18),
            platform_width=2.0,
            border_width=0.25,
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.10,
            step_height_range=(0.03, 0.08),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.10,
            step_height_range=(0.03, 0.08),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
    },
)
"""Low-to-medium difficulty terrains for blind Go2Arm locomotion."""


# 继承vlm_velocity_env_cfg里的LocomotionVelocityRoughEnvCfg(速度控制机器人)类，
# 并指定实例名UnitreeGo2ArmRoughEnvCfg，与gym.register里的env_cfg_entry_point一致，并重写了部分配置
@configclass
class UnitreeGo2ArmRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    base_link_name = "base"
    foot_link_name: list[str] = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ------------------------------Sence------------------------------
        self.scene.robot = UNITREE_Go2Arm_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # The policy is blind; keep only the compact scanner used by terrain-relative height terms.
        self.scene.height_scanner = None
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name

        # Start on the easiest terrain rows and let the official curriculum advance difficulty.
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = GO2ARM_BLIND_ROUGH_TERRAINS_CFG
        self.scene.terrain.max_init_terrain_level = 1

        # ------------------------------Observations-------------------------
        # 暂时不改observation的scale、clip

        # ------------------------------Commands------------------------------
        # 机体速度命令范围。
        self.commands.base_velocity.ranges.lin_vel_x = (-0.50, 0.50)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.50, 0.50)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.50, 0.50)
        # 机体姿态与高度命令范围。
        self.commands.base_pose.ranges.roll = (-0.2, 0.2)
        self.commands.base_pose.ranges.pitch = (-0.2, 0.2)
        self.commands.base_pose.ranges.height = (0.4, 0.4)

        # ------------------------------Actions------------------------------
        self.actions.joint_pos.scale = {".*_hip_joint": 0.125, "^(?!.*_hip_joint).*": 0.25}

        # ------------------------------Events------------------------------
        # 暂时不添加event

        # ------------------------------Rewards------------------------------
        # 速度、姿态和高度跟踪奖励。
        self.rewards.track_lin_vel_xy_exp.weight = 4.0
        self.rewards.track_lin_vel_xy_exp.params["std"] = math.sqrt(0.10)
        self.rewards.track_ang_vel_z_exp.weight = 4.0
        self.rewards.track_ang_vel_z_exp.params["std"] = math.sqrt(0.10)
        self.rewards.track_base_orientation_exp.weight = 2.0
        self.rewards.track_base_orientation_exp.params["std"] = math.sqrt(0.10)
        self.rewards.track_base_height_exp.weight = 2.0
        self.rewards.track_base_height_exp.params["std"] = math.sqrt(0.10)
        self.rewards.base_lin_vel_z_exp.weight = 0.5
        self.rewards.base_lin_vel_z_exp.params["std"] = math.sqrt(0.2)
        self.rewards.base_ang_vel_xy_exp.weight = 1.0
        self.rewards.base_ang_vel_xy_exp.params["std"] = math.sqrt(0.2)

        # 关节与动作正则项。
        self.rewards.joint_torques_exp.weight = 0.00001
        self.rewards.joint_torques_exp.params["std"] = math.sqrt(40.0)
        self.rewards.joint_vel_exp.weight = 0.0001
        self.rewards.joint_vel_exp.params["std"] = math.sqrt(4.0)
        self.rewards.action_rate_exp.weight = 0.001
        self.rewards.action_rate_exp.params["std"] = math.sqrt(0.1)

        # 保留原接触时序奖励配置，但停用其训练权重。
        self.rewards.feet_contact.weight = 0.0
        self.rewards.feet_contact.params["force_variance"] = 1.0
        self.rewards.feet_contact.params["height_variance"] = 0.05
        self.rewards.feet_contact.params["vel_variance"] = 0.01
        self.rewards.feet_contact.params["contact_force_threshold"] = 1.0
        self.rewards.feet_contact.params["height_contact_epsilon"] = 1.0e-4
        # 保留旧步态奖励配置，但停用以改用 IsaacLab rough 的弱步态先验。
        self.rewards.feet_air_time_variance.weight = 0.0
        self.rewards.feet_air_time.weight = 0.0

        # IsaacLab Unitree Go2 rough uses only a weak foot-air-time gait prior.
        self.rewards.isaaclab_feet_air_time.weight = 0.2
        self.rewards.isaaclab_feet_air_time.params["threshold"] = 0.5
        self.rewards.isaaclab_feet_air_time.params["command_threshold"] = 0.1
        self.rewards.isaaclab_feet_air_time.params["sensor_cfg"].body_names = self.foot_link_name
        self.rewards.feet_stumble.weight = -0.2
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = self.foot_link_name
        self.rewards.feet_slide.weight = -0.05
        self.rewards.feet_slide.params["sensor_cfg"].body_names = self.foot_link_name
        self.rewards.feet_slide.params["asset_cfg"].body_names = self.foot_link_name
        self.rewards.feet_height_body.weight = -2.0
        self.rewards.feet_height_body.params["target_height"] = -0.20
        self.rewards.feet_height_body.params["tanh_mult"] = 2.0
        self.rewards.feet_height_body.params["asset_cfg"].body_names = self.foot_link_name

        # 存活与终止奖励。
        # self.rewards.is_alive.weight = 0.1
        # self.rewards.is_terminated.weight = -10.0

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "UnitreeGo2ArmRoughEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------


        # ------------------------------Curriculums------------------------------
        self.curriculum.terrain_levels.func = mdp.terrain_levels_vel_with_metrics
