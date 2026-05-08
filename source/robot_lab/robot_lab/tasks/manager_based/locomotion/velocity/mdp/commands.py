# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp

from .utils import is_robot_on_terrain

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


class UniformThresholdVelocityCommand(mdp.UniformVelocityCommand):
    """Command generator that generates a velocity command in SE(2) from uniform distribution with threshold.

    This command generator automatically detects "pits" terrain and applies restrictions:
    - For pit terrains: only allow forward movement (no lateral or rotational movement)
    """

    cfg: mdp.UniformThresholdVelocityCommandCfg  # type: ignore
    """The configuration of the command generator."""

    def __init__(self, cfg: mdp.UniformThresholdVelocityCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.
        """
        super().__init__(cfg, env)
        # Track which robots were on pit terrain in the previous step
        self.was_on_pit = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample velocity commands with threshold."""
        super()._resample_command(env_ids)
        # set small commands to zero
        self.vel_command_b[env_ids, :2] *= (torch.norm(self.vel_command_b[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _update_command(self):
        """Update commands and apply terrain-aware restrictions in real-time.

        This function:
        1. Calls parent's update to handle heading and standing envs
        2. Checks which robots are currently on pit terrain
        3. For robots leaving pits: resamples their commands
        4. For robots on pits: restricts to forward-only movement and sets heading to 0
        """
        # First, call parent's update command
        super()._update_command()

        # Check which robots are currently on pit terrain (real-time check every step)
        on_pits = is_robot_on_terrain(self._env, "pits")

        # Find robots that just left pit terrain (need to resample)
        left_pit_mask = self.was_on_pit & ~on_pits
        if left_pit_mask.any():
            left_pit_env_ids = torch.where(left_pit_mask)[0]
            # Resample commands for robots that left pits
            self._resample_command(left_pit_env_ids)

        # For robots currently on pits: restrict to forward-only movement with min/max speed
        if on_pits.any():
            pit_env_ids = torch.where(on_pits)[0]
            # Force forward-only movement with min and max speed limits
            self.vel_command_b[pit_env_ids, 0] = torch.clamp(
                torch.abs(self.vel_command_b[pit_env_ids, 0]), min=0.3, max=0.6
            )
            self.vel_command_b[pit_env_ids, 1] = 0.0  # no lateral movement
            self.vel_command_b[pit_env_ids, 2] = 0.0  # no yaw rotation
            # Set heading to 0 for pit robots
            if self.cfg.heading_command:
                self.heading_target[pit_env_ids] = 0.0

        # Update tracking state
        self.was_on_pit = on_pits


@configclass
class UniformThresholdVelocityCommandCfg(mdp.UniformVelocityCommandCfg):
    """Configuration for the uniform threshold velocity command generator."""

    class_type: type = UniformThresholdVelocityCommand


class BasePoseCommand(CommandTerm):
    """Command term that samples torso roll, pitch, and height targets without trajectory caching."""

    cfg: BasePoseCommandCfg
    """Configuration for the base pose command."""

    def __init__(self, cfg: BasePoseCommandCfg, env: ManagerBasedEnv):
        """Initialize the command term."""
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        if cfg.torso_body_name not in self.robot.body_names:
            raise ValueError(
                f"Body name '{cfg.torso_body_name}' not found in robot body names: {self.robot.body_names}"
            )
        self.torso_body_id = self.robot.body_names.index(cfg.torso_body_name)

        self.torso_roll_pitch_height_command = torch.zeros(self.num_envs, 3, device=self.device)
        self.torso_projected_gravity_goal = torch.zeros(self.num_envs, 3, device=self.device)
        self._gravity_vec = self.robot.data.GRAVITY_VEC_W.clone()
        self._x_axis = torch.tensor([1.0, 0.0, 0.0], device=self.device)
        self._y_axis = torch.tensor([0.0, 1.0, 0.0], device=self.device)

        self.metrics["torso_roll_pitch_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["torso_height_error"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        return f"BasePoseCommand:\n\tCommand dimension: {tuple(self.command.shape[1:])}\n"

    @property
    def command(self) -> torch.Tensor:
        """Current torso roll, pitch, and height commands. Shape is (num_envs, 3)."""
        return self.torso_roll_pitch_height_command

    def _update_metrics(self):
        """Update torso command tracking metrics."""
        self.metrics["torso_roll_pitch_error"] = torch.norm(
            self.torso_projected_gravity_goal[:, :2] - self.robot.data.projected_gravity_b[:, :2],
            dim=1,
        )
        self.metrics["torso_height_error"] = torch.abs(
            self.torso_roll_pitch_height_command[:, 2] - self.robot.data.body_pos_w[:, self.torso_body_id, 2]
        )

    def _resample_command(self, env_ids: Sequence[int]):
        """Sample roll, pitch, and height targets directly from uniform ranges."""
        num_envs = len(env_ids)
        self.torso_roll_pitch_height_command[env_ids, 0] = math_utils.sample_uniform(
            *self.cfg.ranges.roll, (num_envs,), device=self.device
        )
        self.torso_roll_pitch_height_command[env_ids, 1] = math_utils.sample_uniform(
            *self.cfg.ranges.pitch, (num_envs,), device=self.device
        )
        self.torso_roll_pitch_height_command[env_ids, 2] = math_utils.sample_uniform(
            *self.cfg.ranges.height, (num_envs,), device=self.device
        )

        quat_roll = math_utils.quat_from_angle_axis(self.torso_roll_pitch_height_command[env_ids, 0], self._x_axis)
        quat_pitch = math_utils.quat_from_angle_axis(self.torso_roll_pitch_height_command[env_ids, 1], self._y_axis)
        desired_base_quat = math_utils.quat_mul(quat_roll, quat_pitch)
        self.torso_projected_gravity_goal[env_ids] = math_utils.quat_rotate_inverse(
            desired_base_quat, self._gravity_vec[env_ids]
        )

    def _update_command(self):
        """Keep the sampled command constant until the next resampling event."""
        pass


class ArmJointTrajectoryCommand(CommandTerm):
    """Command term that interpolates arm joints from start to goal, then holds the goal."""

    cfg: ArmJointTrajectoryCommandCfg
    """Configuration for the arm joint trajectory command."""

    def __init__(self, cfg: ArmJointTrajectoryCommandCfg, env: ManagerBasedEnv):
        """Initialize the command term."""
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.arm_joint_ids = self.robot.find_joints(self.cfg.joint_names, preserve_order=True)[0]

        self.arm_joint_start = torch.zeros(self.num_envs, len(self.arm_joint_ids), device=self.device)
        self.arm_joint_goal = torch.zeros(self.num_envs, len(self.arm_joint_ids), device=self.device)
        self.arm_joint_sub_goal = torch.zeros(self.num_envs, len(self.arm_joint_ids), device=self.device)

        # Use the parsed URDF soft joint limits as the uniform sampling bounds.
        self.lower_bound = self.robot.data.soft_joint_pos_limits[:, self.arm_joint_ids, 0]
        self.upper_bound = self.robot.data.soft_joint_pos_limits[:, self.arm_joint_ids, 1]

        self.step_dt = env.step_dt
        self.timer = torch.zeros(self.num_envs, device=self.device)
        self.traj_timesteps = self._sample_segment_timesteps(self.cfg.trajectory_time)
        self.hold_timesteps = self._sample_segment_timesteps(self.cfg.hold_time)

        self.metrics["arm_joint_error"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "ArmJointTrajectoryCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """Current interpolated joint target. Shape is (num_envs, num_joints)."""
        return self.arm_joint_sub_goal

    def _sample_segment_timesteps(self, time_range: tuple[float, float], num_envs: int | None = None) -> torch.Tensor:
        """Sample per-env segment durations and convert them to discrete timesteps."""
        batch = self.num_envs if num_envs is None else num_envs
        timesteps = (
            math_utils.sample_uniform(time_range[0], time_range[1], (batch,), device=self.device) / self.step_dt
        ).int()
        return torch.clamp(timesteps, min=1)

    def _update_metrics(self):
        """Update the joint tracking error against the current interpolated goal."""
        self.metrics["arm_joint_error"] = torch.norm(
            self.robot.data.joint_pos[:, self.arm_joint_ids] - self.arm_joint_sub_goal,
            dim=1,
        )

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample a new start-goal segment for the selected environments."""
        if len(env_ids) == 0:
            return

        self.arm_joint_start[env_ids] = torch.clamp(
            self.robot.data.joint_pos[env_ids][:, self.arm_joint_ids],
            self.lower_bound[env_ids],
            self.upper_bound[env_ids],
        )
        self.arm_joint_sub_goal[env_ids] = self.arm_joint_start[env_ids]
        self.arm_joint_goal[env_ids] = math_utils.sample_uniform(
            self.lower_bound[env_ids],
            self.upper_bound[env_ids],
            (len(env_ids), len(self.arm_joint_ids)),
            self.device,
        )

        self.traj_timesteps[env_ids] = self._sample_segment_timesteps(self.cfg.trajectory_time, len(env_ids))
        self.hold_timesteps[env_ids] = self._sample_segment_timesteps(self.cfg.hold_time, len(env_ids))
        self.timer[env_ids] = 0.0

    def _update_command(self):
        """Advance the current interpolation segment and resample completed environments."""
        self.timer += 1.0

        reaching = self.timer <= self.traj_timesteps
        holding = torch.logical_and(
            self.traj_timesteps < self.timer,
            self.timer <= self.traj_timesteps + self.hold_timesteps,
        )
        reset = self.timer > self.traj_timesteps + self.hold_timesteps

        reaching_ids = reaching.nonzero(as_tuple=False).squeeze(-1)
        holding_ids = holding.nonzero(as_tuple=False).squeeze(-1)
        reset_ids = reset.nonzero(as_tuple=False).squeeze(-1)

        if len(reaching_ids) > 0:
            alpha = (self.timer[reaching_ids] / self.traj_timesteps[reaching_ids]).reshape(-1, 1)
            self.arm_joint_sub_goal[reaching_ids] = torch.lerp(
                self.arm_joint_start[reaching_ids],
                self.arm_joint_goal[reaching_ids],
                alpha,
            )

        if len(holding_ids) > 0:
            self.arm_joint_sub_goal[holding_ids] = self.arm_joint_goal[holding_ids].clone()

        if len(reset_ids) > 0:
            self._resample(reset_ids)


class DiscreteCommandController(CommandTerm):
    """
    Command generator that assigns discrete commands to environments.

    Commands are stored as a list of predefined integers.
    The controller maps these commands by their indices (e.g., index 0 -> 10, index 1 -> 20).
    """

    cfg: DiscreteCommandControllerCfg
    """Configuration for the command controller."""

    def __init__(self, cfg: DiscreteCommandControllerCfg, env: ManagerBasedEnv):
        """
        Initialize the command controller.

        Args:
            cfg: The configuration of the command controller.
            env: The environment object.
        """
        # Initialize the base class
        super().__init__(cfg, env)

        # Validate that available_commands is non-empty
        if not self.cfg.available_commands:
            raise ValueError("The available_commands list cannot be empty.")

        # Ensure all elements are integers
        if not all(isinstance(cmd, int) for cmd in self.cfg.available_commands):
            raise ValueError("All elements in available_commands must be integers.")

        # Store the available commands
        self.available_commands = self.cfg.available_commands

        # Create buffers to store the command
        # -- command buffer: stores discrete action indices for each environment
        self.command_buffer = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        # -- current_commands: stores a snapshot of the current commands (as integers)
        self.current_commands = [self.available_commands[0]] * self.num_envs  # Default to the first command

    def __str__(self) -> str:
        """Return a string representation of the command controller."""
        return (
            "DiscreteCommandController:\n"
            f"\tNumber of environments: {self.num_envs}\n"
            f"\tAvailable commands: {self.available_commands}\n"
        )

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """Return the current command buffer. Shape is (num_envs, 1)."""
        return self.command_buffer

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        """Update metrics for the command controller."""
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample commands for the given environments."""
        sampled_indices = torch.randint(
            len(self.available_commands), (len(env_ids),), dtype=torch.int32, device=self.device
        )
        sampled_commands = torch.tensor(
            [self.available_commands[idx.item()] for idx in sampled_indices], dtype=torch.int32, device=self.device
        )
        self.command_buffer[env_ids] = sampled_commands

    def _update_command(self):
        """Update and store the current commands."""
        self.current_commands = self.command_buffer.tolist()


@configclass
class DiscreteCommandControllerCfg(CommandTermCfg):
    """Configuration for the discrete command controller."""

    class_type: type = DiscreteCommandController

    available_commands: list[int] = []
    """
    List of available discrete commands, where each element is an integer.
    Example: [10, 20, 30, 40, 50]
    """


@configclass
class BasePoseCommandCfg(CommandTermCfg):
    """Configuration for the torso base pose command."""

    class_type: type = BasePoseCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    torso_body_name: str = MISSING
    """Name of the torso body to associate with the command."""

    @configclass
    class Ranges:
        """Uniform distribution ranges for torso pose commands."""

        roll: tuple[float, float] = MISSING
        pitch: tuple[float, float] = MISSING
        height: tuple[float, float] = MISSING

    ranges: Ranges = MISSING


@configclass
class ArmJointTrajectoryCommandCfg(CommandTermCfg):
    """Configuration for the interpolated arm joint trajectory command."""

    class_type: type = ArmJointTrajectoryCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    trajectory_time: tuple[float, float] = MISSING
    """Interpolation duration in seconds."""

    hold_time: tuple[float, float] = MISSING
    """Goal hold duration in seconds."""

    joint_names: list[str] = MISSING
    """Ordered joint names to command."""


class DesiredFeetSwingHeightCommand(CommandTerm):
    """Desired feet swing height command for [FL, FR, RL, RR]."""

    cfg: DesiredFeetSwingHeightCommandCfg
    """Configuration for the feet swing height command."""

    def __init__(self, cfg: DesiredFeetSwingHeightCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        if len(cfg.phase_offsets) != 4:
            raise ValueError("phase_offsets must have 4 elements in [FL, FR, RL, RR] order.")

        self._dt = float(env.cfg.decimation * env.cfg.sim.dt)
        self._phase = torch.zeros(self.num_envs, device=self.device)
        self._max_height = torch.zeros(self.num_envs, device=self.device)
        self._command = torch.zeros(self.num_envs, 4, device=self.device)
        self._phase_offsets = torch.tensor(cfg.phase_offsets, device=self.device).view(1, 4)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        extras = super().reset(env_ids)
        if env_ids is None:
            self._phase = torch.rand(self.num_envs, device=self.device)
            return extras
        if isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.tolist()
        self._phase[env_ids] = torch.rand(len(env_ids), device=self.device)
        return extras

    @property
    def command(self) -> torch.Tensor:
        """Current desired swing heights for [FL, FR, RL, RR]."""
        return self._command

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        if self.cfg.height_range is None:
            self._max_height[env_ids] = float(self.cfg.max_height)
            return
        self._max_height[env_ids] = math_utils.sample_uniform(
            self.cfg.height_range[0],
            self.cfg.height_range[1],
            (len(env_ids),),
            device=self.device,
        )

    def _update_command(self):
        self._phase = torch.remainder(self._phase + self._dt * float(self.cfg.gait_frequency), 1.0)
        phase = torch.remainder(self._phase.unsqueeze(-1) + self._phase_offsets, 1.0)
        heights = self._max_height.unsqueeze(-1) * torch.sin(2.0 * math.pi * phase)
        if self.cfg.clip_to_positive:
            heights = torch.clamp(heights, min=0.0)
        self._command = heights


@configclass
class DesiredFeetSwingHeightCommandCfg(CommandTermCfg):
    """Configuration for desired feet swing height command."""

    class_type: type = DesiredFeetSwingHeightCommand

    max_height: float = 0.12
    """Fixed swing height in meters when ``height_range`` is not provided."""

    height_range: tuple[float, float] | None = None
    """Optional uniform sampling range for the maximum swing height."""

    gait_frequency: float = 1.5
    """Gait cycle frequency in Hz."""

    phase_offsets: tuple[float, float, float, float] = (0.0, 0.5, 0.5, 0.0)
    """Phase offsets for [FL, FR, RL, RR]."""

    clip_to_positive: bool = True
    """Whether to clamp swing heights to non-negative values."""
