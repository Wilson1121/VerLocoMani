# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions import JointAction, JointActionCfg
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class CommandArmPolicyLegJointPositionAction(JointAction):
    """Apply policy outputs to leg joints and command targets directly to arm joints."""

    cfg: CommandArmPolicyLegJointPositionActionCfg
    """Configuration for the mixed action term."""

    def __init__(self, cfg: CommandArmPolicyLegJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self._filtered_leg_actions = torch.zeros_like(self._raw_actions)
        if cfg.low_pass_cutoff_hz is None:
            self._low_pass_alpha = 1.0
        else:
            if cfg.low_pass_cutoff_hz <= 0.0:
                raise ValueError("The leg action low-pass cutoff frequency must be positive.")
            nyquist_frequency = 0.5 / env.step_dt
            if cfg.low_pass_cutoff_hz >= nyquist_frequency:
                raise ValueError(
                    f"The leg action low-pass cutoff frequency ({cfg.low_pass_cutoff_hz} Hz) must be lower than "
                    f"the policy-rate Nyquist frequency ({nyquist_frequency} Hz)."
                )
            self._low_pass_alpha = 1.0 - math.exp(-2.0 * math.pi * cfg.low_pass_cutoff_hz * env.step_dt)

        if cfg.use_default_offset:
            self._offset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()

        self._arm_joint_ids, self._arm_joint_names = self._asset.find_joints(
            self.cfg.arm_joint_names, preserve_order=True
        )
        leg_joint_ids = set(self._joint_ids)
        arm_joint_ids = set(self._arm_joint_ids)
        if leg_joint_ids & arm_joint_ids:
            raise ValueError("joint_names and arm_joint_names must not overlap.")

        self.arm_command_name = cfg.arm_command_name
        self.command_manager = env.command_manager

        self._arm_raw_actions = torch.zeros(self.num_envs, len(self._arm_joint_ids), device=self.device)
        self._arm_processed_actions = torch.zeros_like(self._arm_raw_actions)

    @property
    def arm_raw_actions(self) -> torch.Tensor:
        """Raw arm command targets from the command manager."""
        return self._arm_raw_actions

    @property
    def arm_processed_actions(self) -> torch.Tensor:
        """Processed arm joint targets written directly to simulation."""
        return self._arm_processed_actions

    def process_actions(self, actions: torch.Tensor):
        """Process policy leg actions and fetch arm targets from the command manager."""
        self._filtered_leg_actions.lerp_(actions, self._low_pass_alpha)
        super().process_actions(self._filtered_leg_actions)
        self._raw_actions[:] = actions

        self._arm_raw_actions[:] = self.command_manager.get_command(self.arm_command_name)
        self._arm_processed_actions[:] = self._arm_raw_actions

    def apply_actions(self):
        """Apply leg and arm joint position targets."""
        self._asset.set_joint_position_target(self.processed_actions, joint_ids=self._joint_ids)
        self._asset.set_joint_position_target(self.arm_processed_actions, joint_ids=self._arm_joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset the policy-action filter to the default-pose action."""
        if env_ids is None:
            env_ids = slice(None)
        self._filtered_leg_actions[env_ids] = 0.0
        super().reset(env_ids)


@configclass
class CommandArmPolicyLegJointPositionActionCfg(JointActionCfg):
    """Action term configuration for command-driven arm joints and policy-driven leg joints."""

    class_type: type[ActionTerm] = CommandArmPolicyLegJointPositionAction

    arm_joint_names: list[str] = MISSING
    """Ordered arm joint names driven directly by the command manager."""

    arm_command_name: str = MISSING
    """Name of the arm command term providing absolute joint position targets."""

    use_default_offset: bool = True
    """Whether to use default joint positions as offsets for the policy-controlled leg joints."""

    low_pass_cutoff_hz: float | None = None
    """Cutoff frequency of the first-order leg-action low-pass filter in Hz.

    If set to None, filtering is disabled. The filter acts only on the policy-controlled leg
    position actions; arm command targets and wheel velocity actions are unaffected.
    """
