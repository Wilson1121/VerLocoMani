# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Train the Go2WArm flat policy through four checkpoint-linked arm-motion stages."""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path


TASK_NAME = "RobotLab-Isaac-Velocity-Flat-UnitreeGo2WArm-v0"
BASE_EXPERIMENT_NAME = "unitree_go2w_arm_flat"
ARM_MOTION_STAGES = (0, 1, 2, 3)
ARM_MOTION_RANGES = (0.0, 0.3, 0.6, 0.9)
MODEL_PATTERN = re.compile(r"model_(\d+)\.pt")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRAIN_SCRIPT = Path(__file__).with_name("train.py")
LOG_BASE_DIR = PROJECT_ROOT / "logs" / "rsl_rl"


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run four sequential 3000-iteration Go2WArm training stages."
    )
    parser.add_argument("--task", default=TASK_NAME, help="Registered Isaac Lab task name.")
    parser.add_argument("--iterations-per-stage", type=int, default=3000)
    parser.add_argument(
        "--start-stage",
        type=int,
        choices=ARM_MOTION_STAGES,
        default=0,
        help="First stage to run. Stages after zero require an existing batch and initial run.",
    )
    parser.add_argument(
        "--batch-name",
        default=None,
        help="Shared directory name for all stages. A timestamped name is used by default.",
    )
    parser.add_argument(
        "--initial-run",
        default=None,
        help="Existing run directory inside the batch used to initialize a nonzero start stage.",
    )
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--show-gui",
        action="store_true",
        help="Launch training with the simulator GUI instead of --headless.",
    )
    args, train_args = parser.parse_known_args()
    return args, train_args


def _validate_args(args: argparse.Namespace, train_args: list[str]) -> None:
    if args.iterations_per_stage < 1:
        raise ValueError(f"iterations-per-stage must be positive, got {args.iterations_per_stage}.")
    if args.batch_name is not None and re.fullmatch(r"[A-Za-z0-9_.-]+", args.batch_name) is None:
        raise ValueError("batch-name may only contain letters, numbers, '_', '-', and '.'.")
    if args.initial_run is not None and re.fullmatch(r"[A-Za-z0-9_.-]+", args.initial_run) is None:
        raise ValueError("initial-run may only contain letters, numbers, '_', '-', and '.'.")
    if args.start_stage == 0 and args.initial_run is not None:
        raise ValueError("initial-run is only valid when start-stage is greater than zero.")
    if args.start_stage > 0:
        if args.batch_name is None:
            raise ValueError("batch-name is required when start-stage is greater than zero.")
        if args.initial_run is None:
            raise ValueError("initial-run is required when start-stage is greater than zero.")
    managed_arguments = (
        "--max_iterations",
        "--experiment_name",
        "--run_name",
        "--resume",
        "--load_run",
        "--checkpoint",
    )
    managed_hydra_overrides = (
        "env.arm_motion_stage=",
        "env.commands.arm_joint_trajectory.fixed_default=",
        "env.commands.arm_joint_trajectory.init_range=",
    )
    for argument in train_args:
        if argument.startswith(managed_hydra_overrides):
            raise ValueError(f"{argument} is managed automatically by the staged launcher.")
        if any(argument == name or argument.startswith(f"{name}=") for name in managed_arguments):
            raise ValueError(f"{argument} is managed automatically by the staged launcher.")


def _latest_checkpoint(run_dir: Path) -> Path:
    checkpoints: list[tuple[int, Path]] = []
    for checkpoint in run_dir.glob("model_*.pt"):
        match = MODEL_PATTERN.fullmatch(checkpoint.name)
        if match is not None:
            checkpoints.append((int(match.group(1)), checkpoint))
    if not checkpoints:
        raise RuntimeError(f"No model checkpoint was written in {run_dir}.")
    return max(checkpoints, key=lambda item: item[0])[1]


def _find_new_stage_run(batch_dir: Path, existing_runs: set[Path], stage: int) -> Path:
    stage_suffix = f"_arm_stage_{stage}"
    candidates = [
        run_dir
        for run_dir in batch_dir.iterdir()
        if run_dir.is_dir() and run_dir not in existing_runs and run_dir.name.endswith(stage_suffix)
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one new stage-{stage} run directory in {batch_dir}, "
            f"found {len(candidates)}."
        )
    return candidates[0]


def _build_train_command(
    args: argparse.Namespace,
    train_args: list[str],
    experiment_name: str,
    stage: int,
    previous_run: Path | None,
    previous_checkpoint: Path | None,
) -> list[str]:
    arm_motion_range = ARM_MOTION_RANGES[stage]
    fixed_default = stage == 0
    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--task",
        args.task,
        "--max_iterations",
        str(args.iterations_per_stage),
        "--experiment_name",
        experiment_name,
        "--run_name",
        f"arm_stage_{stage}",
    ]
    if not args.show_gui:
        command.append("--headless")
    if args.num_envs is not None:
        command.extend(["--num_envs", str(args.num_envs)])
    if args.device is not None:
        command.extend(["--device", args.device])
    if args.seed is not None:
        command.extend(["--seed", str(args.seed)])
    if previous_run is not None and previous_checkpoint is not None:
        command.extend(
            [
                "--resume",
                "--load_run",
                previous_run.name,
                "--checkpoint",
                previous_checkpoint.name,
            ]
        )
    command.extend(train_args)
    command.extend(
        [
            f"env.commands.arm_joint_trajectory.fixed_default={str(fixed_default).lower()}",
            f"env.commands.arm_joint_trajectory.init_range={arm_motion_range}",
        ]
    )
    return command


def main() -> None:
    args, train_args = _parse_args()
    _validate_args(args, train_args)

    batch_name = args.batch_name or datetime.now().strftime("arm_staged_%Y-%m-%d_%H-%M-%S")
    experiment_name = f"{BASE_EXPERIMENT_NAME}/{batch_name}"
    batch_dir = LOG_BASE_DIR / BASE_EXPERIMENT_NAME / batch_name
    if args.start_stage == 0 and batch_dir.exists():
        raise FileExistsError(
            f"Staged training directory already exists: {batch_dir}. "
            "Choose a different --batch-name."
        )
    if args.start_stage > 0 and not batch_dir.is_dir():
        raise FileNotFoundError(f"Staged training directory does not exist: {batch_dir}.")

    if args.start_stage == 0:
        previous_run = None
        previous_checkpoint = None
    else:
        previous_run = batch_dir / args.initial_run
        if not previous_run.is_dir():
            raise FileNotFoundError(f"Initial run directory does not exist: {previous_run}.")
        previous_checkpoint = _latest_checkpoint(previous_run)
        print(f"[INFO] Initial checkpoint: {previous_checkpoint}", flush=True)

    for stage in ARM_MOTION_STAGES[args.start_stage :]:
        existing_runs = set(batch_dir.iterdir()) if batch_dir.exists() else set()
        command = _build_train_command(
            args,
            train_args,
            experiment_name,
            stage,
            previous_run,
            previous_checkpoint,
        )
        print(f"\n[INFO] Starting arm motion stage {stage}:")
        print(shlex.join(command), flush=True)
        try:
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        except subprocess.CalledProcessError as error:
            raise SystemExit(
                f"Arm motion stage {stage} failed with exit code {error.returncode}. "
                "Later stages were not started."
            ) from error

        previous_run = _find_new_stage_run(batch_dir, existing_runs, stage)
        previous_checkpoint = _latest_checkpoint(previous_run)
        print(f"[INFO] Stage {stage} checkpoint: {previous_checkpoint}", flush=True)

    print(f"\n[INFO] Four-stage training completed. Final checkpoint: {previous_checkpoint}")


if __name__ == "__main__":
    main()
