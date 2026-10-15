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
        self._arm_gravity_actions = torch.zeros_like(self._arm_raw_actions)

        self.enable_gravity_compensation = cfg.enable_gravity_compensation
        if cfg.gravity_compensation_scale < 0.0:
            raise ValueError(
                "Gravity compensation scale must be non-negative, but received "
                f"{cfg.gravity_compensation_scale}."
            )
        self.gravity_compensation_scale = cfg.gravity_compensation_scale

        # PhysX generalized forces include the six floating-base degrees of
        # freedom before the articulation joints. Fixed-base assets do not have
        # this prefix. Keep this mapping explicit and validate it at runtime.
        self._base_dof_offset = 0 if self._asset.is_fixed_base else 6
        self._arm_gravity_dof_ids = torch.tensor(
            [joint_id + self._base_dof_offset for joint_id in self._arm_joint_ids],
            dtype=torch.long,
            device=self.device,
        )

    @property
    def arm_raw_actions(self) -> torch.Tensor:
        """Raw arm command targets from the command manager."""
        return self._arm_raw_actions

    @property
    def arm_processed_actions(self) -> torch.Tensor:
        """Processed arm joint targets written directly to simulation."""
        return self._arm_processed_actions

    @property
    def arm_gravity_actions(self) -> torch.Tensor:
        """Current arm gravity-compensation effort targets."""
        return self._arm_gravity_actions

    def process_actions(self, actions: torch.Tensor):
        """Process policy leg actions and fetch arm targets from the command manager."""
        self._filtered_leg_actions.lerp_(actions, self._low_pass_alpha)
        super().process_actions(self._filtered_leg_actions)
        self._raw_actions[:] = actions

        self._arm_raw_actions[:] = self.command_manager.get_command(self.arm_command_name)
        self._arm_processed_actions[:] = self._arm_raw_actions

    def _update_arm_gravity_actions(self) -> None:
        """Read PhysX generalized gravity forces and select the arm joints."""
        rigid_props = self._asset.cfg.spawn.rigid_props
        gravity_disabled = rigid_props is not None and rigid_props.disable_gravity
        if not self.enable_gravity_compensation or gravity_disabled:
            self._arm_gravity_actions.zero_()
            return

        gravity_forces = self._asset.root_physx_view.get_gravity_compensation_forces()
        expected_num_dofs = self._asset.num_joints + self._base_dof_offset
        if gravity_forces.ndim != 2 or gravity_forces.shape != (self.num_envs, expected_num_dofs):
            raise RuntimeError(
                "Unexpected PhysX gravity-compensation tensor shape. Expected "
                f"({self.num_envs}, {expected_num_dofs}) for "
                f"{'fixed' if self._asset.is_fixed_base else 'floating'}-base asset, "
                f"but received {tuple(gravity_forces.shape)}."
            )

        self._arm_gravity_actions.copy_(
            gravity_forces.index_select(dim=1, index=self._arm_gravity_dof_ids)
        )
        self._arm_gravity_actions.mul_(self.gravity_compensation_scale)

    def apply_actions(self):
        """Apply position targets and current-state arm gravity compensation."""
        self._update_arm_gravity_actions()
        self._asset.set_joint_position_target(self.processed_actions, joint_ids=self._joint_ids)
        self._asset.set_joint_position_target(self.arm_processed_actions, joint_ids=self._arm_joint_ids)
        self._asset.set_joint_effort_target(self._arm_gravity_actions, joint_ids=self._arm_joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset the policy-action filter to the default-pose action."""
        if env_ids is None:
            env_ids = slice(None)
        self._filtered_leg_actions[env_ids] = 0.0
        super().reset(env_ids)
        self._arm_gravity_actions[env_ids] = 0.0


@configclass
class CommandArmPolicyLegJointPositionActionCfg(JointActionCfg):
    """Action term configuration for command-driven arm joints and policy-driven leg joints."""

    class_type: type[ActionTerm] = CommandArmPolicyLegJointPositionAction

    arm_joint_names: list[str] = MISSING
    """Ordered arm joint names driven directly by the command manager."""

    arm_command_name: str = MISSING
    """Name of the arm command term providing absolute joint position targets."""

    enable_gravity_compensation: bool = False
    """Whether to apply PhysX gravity-compensation forces to the arm joints."""

    gravity_compensation_scale: float = 1.0
    """Scale applied to the PhysX arm gravity-compensation forces."""

    use_default_offset: bool = True
    """Whether to use default joint positions as offsets for the policy-controlled leg joints."""

    low_pass_cutoff_hz: float | None = None
    """Cutoff frequency of the first-order leg-action low-pass filter in Hz.

    If set to None, filtering is disabled. The filter acts only on the policy-controlled leg
    position actions; arm command targets and wheel velocity actions are unaffected.
    """
