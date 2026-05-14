# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import math

from isaaclab.utils import configclass

from robot_lab.tasks.manager_based.locomotion.velocity.vlm_velocity_env_cfg import LocomotionVelocityRoughEnvCfg

##
# Pre-defined configs
##
# # use cloud assets
# from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip
# use local assets
from robot_lab.assets.unitree import UNITREE_Go2Arm_CFG  # isort: skip

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
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name

        # ------------------------------Observations-------------------------
        # 暂时不改observation的scale、clip

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

        # 接触时序奖励。
        self.rewards.feet_contact.weight = 1.0
        self.rewards.feet_contact.params["force_variance"] = 1.0
        self.rewards.feet_contact.params["height_variance"] = 0.05
        self.rewards.feet_contact.params["vel_variance"] = 0.01
        self.rewards.feet_contact.params["contact_force_threshold"] = 1.0
        self.rewards.feet_contact.params["height_contact_epsilon"] = 1.0e-4

        self.rewards.feet_air_time_variance.weight = -1.0

        self.rewards.feet_air_time.weight = 0.25

        # 存活与终止奖励。
        # self.rewards.is_alive.weight = 0.1
        # self.rewards.is_terminated.weight = -10.0

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "UnitreeGo2ArmRoughEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------


        # ------------------------------Curriculums------------------------------


        # ------------------------------Commands------------------------------
        # 机体速度命令范围。
        self.commands.base_velocity.ranges.lin_vel_x = (-0.25, 0.25)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.25, 0.25)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.25, 0.25)

        # 机体姿态与高度命令范围。
        self.commands.base_pose.ranges.roll = (-0.0, 0.0)
        self.commands.base_pose.ranges.pitch = (-0.0, 0.0)
        self.commands.base_pose.ranges.height = (0.4, 0.4)
