# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

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


        # ------------------------------Actions------------------------------


        # ------------------------------Events------------------------------
        

        # ------------------------------Rewards------------------------------
        

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "UnitreeGo2ArmRoughEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------


        # ------------------------------Curriculums------------------------------


        # ------------------------------Commands------------------------------
