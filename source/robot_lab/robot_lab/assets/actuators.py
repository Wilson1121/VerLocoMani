# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Custom actuator models used by RobotLab assets."""

from __future__ import annotations

import torch

from isaaclab.actuators import DelayedPDActuator, DelayedPDActuatorCfg, IdealPDActuator
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions


class FeedForwardDelayedPDActuator(DelayedPDActuator):
    """Delayed PD actuator that applies feed-forward effort without delay.

    Position and velocity targets are delayed to model communication and command
    latency. The effort target is evaluated immediately, which is required for
    state-dependent feed-forward terms such as gravity compensation.
    """

    cfg: FeedForwardDelayedPDActuatorCfg

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        """Compute delayed PD torques while keeping effort feed-forward current."""
        control_action.joint_positions = self.positions_delay_buffer.compute(control_action.joint_positions)
        control_action.joint_velocities = self.velocities_delay_buffer.compute(control_action.joint_velocities)

        # Bypass DelayedPDActuator.compute() so that joint_efforts is not passed
        # through the effort delay buffer. IdealPDActuator still combines and
        # clips PD and feed-forward torque according to the actuator limits.
        return IdealPDActuator.compute(self, control_action, joint_pos, joint_vel)


@configclass
class FeedForwardDelayedPDActuatorCfg(DelayedPDActuatorCfg):
    """Configuration for delayed PD control with non-delayed feed-forward effort."""

    class_type: type = FeedForwardDelayedPDActuator
