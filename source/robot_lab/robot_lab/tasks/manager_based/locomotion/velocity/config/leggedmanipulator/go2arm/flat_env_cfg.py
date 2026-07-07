# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from .rough_env_cfg import UnitreeGo2ArmRoughEnvCfg


@configclass
class UnitreeGo2ArmFlatEnvCfg(UnitreeGo2ArmRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # override rewards
        self.rewards.track_base_height_exp.params["sensor_cfg"] = None
        self.commands.base_pose.height_sensor_name = None

        # no height scan
        self.scene.height_scanner = None
        self.scene.height_scanner_base = None

        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "UnitreeGo2ArmFlatEnvCfg":
            self.disable_zero_weight_rewards()
