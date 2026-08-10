# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""OpenArm v2.0 bimanual physical topology for the generic Damiao whole-body adapter."""

from __future__ import annotations

from pathlib import Path
from typing import NoReturn

import can_motor_control
from can_motor_control import damiao

from dimos.hardware.whole_body.damiao.adapter import DamiaoWholeBodyAdapter
from dimos.hardware.whole_body.damiao.config import DamiaoRuntimeConfig
from dimos.hardware.whole_body.spec import MotorState
from dimos.utils.data import LfsPath
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


def _arm_motors(side: str) -> list[can_motor_control.MotorSpec]:
    return [
        can_motor_control.MotorSpec(f"openarm_{side}_joint1", damiao.MotorType.DM8009, 0x01, 0x11),
        can_motor_control.MotorSpec(f"openarm_{side}_joint2", damiao.MotorType.DM8009, 0x02, 0x12),
        can_motor_control.MotorSpec(f"openarm_{side}_joint3", damiao.MotorType.DM4340, 0x03, 0x13),
        can_motor_control.MotorSpec(f"openarm_{side}_joint4", damiao.MotorType.DM4340, 0x04, 0x14),
        can_motor_control.MotorSpec(f"openarm_{side}_joint5", damiao.MotorType.DM4310, 0x05, 0x15),
        can_motor_control.MotorSpec(f"openarm_{side}_joint6", damiao.MotorType.DM4310, 0x06, 0x16),
        can_motor_control.MotorSpec(f"openarm_{side}_joint7", damiao.MotorType.DM4310, 0x07, 0x17),
    ]


def _gripper_motor(side: str) -> can_motor_control.MotorSpec:
    return can_motor_control.MotorSpec(
        f"openarm_{side}_gripper",
        damiao.MotorType.DM4310,
        0x08,
        0x18,
    )


class OpenArmDamiaoAdapter(DamiaoWholeBodyAdapter):
    """Two OpenArm v2.0 arms with grippers, one CAN bus per arm."""

    _feedback_clamp_margin = 0.05
    _arm_position_lower = (
        -3.4907,
        -3.3161,
        -1.5708,
        0.0,
        -1.5708,
        -0.7854,
        -1.5708,
        -1.3963,
        -0.17453,
        -1.5708,
        0.0,
        -1.5708,
        -0.7854,
        -1.5708,
    )
    _arm_position_upper = (
        1.3963,
        0.17453,
        1.5708,
        2.4435,
        1.5708,
        0.7854,
        1.5708,
        3.4907,
        3.3161,
        1.5708,
        2.4435,
        1.5708,
        0.7854,
        1.5708,
    )

    arm_joints = {
        "left_arm": tuple(f"left_arm/joint{index}" for index in range(1, 8)),
        "right_arm": tuple(f"right_arm/joint{index}" for index in range(1, 8)),
    }
    gripper_joints = {
        "left_gripper": "left_arm/gripper",
        "right_gripper": "right_arm/gripper",
    }
    # can0/can1 follow USB enumeration order; remap through
    # DamiaoRuntimeConfig.bus_addresses if the rig comes up swapped.
    bus_defaults = {"left": "can1", "right": "can0"}
    gravity_joint_names = (
        *(f"openarm_left_joint{index}" for index in range(1, 8)),
        *(f"openarm_right_joint{index}" for index in range(1, 8)),
    )

    def __init__(
        self,
        address: str | Path | None = None,
        *,
        runtime_config: DamiaoRuntimeConfig | None = None,
        dof: int | None = None,
        hardware_id: str = "whole_body",
        domain_id: int = 0,
    ) -> None:
        super().__init__(
            address,
            runtime_config=runtime_config,
            dof=dof,
            hardware_id=hardware_id,
            domain_id=domain_id,
        )
        self._feedback_fault: str | None = None
        self._warned_feedback_clamps: set[tuple[str, str]] = set()

    def connect(self) -> bool:
        """Reconnect deliberately clears a previously latched feedback fault."""
        self._feedback_fault = None
        self._warned_feedback_clamps.clear()
        return super().connect()

    def activate(self) -> bool:
        """Refuse activation until a latched feedback fault is cleared by reconnect."""
        if self._feedback_fault is not None:
            logger.error(
                "OpenArm activation rejected while feedback fault is latched",
                hardware_id=self._hardware_id,
                error=self._feedback_fault,
            )
            return False
        return super().activate()

    def read_motor_states(self) -> list[MotorState]:
        """Normalize small OpenArm encoder overshoot before exposing feedback."""
        if self._feedback_fault is not None:
            raise RuntimeError(f"OpenArm feedback fault is latched: {self._feedback_fault}")

        states = super().read_motor_states()
        normalized = list(states)
        for index, (joint_name, lower, upper) in enumerate(
            zip(
                self.joint_names[: len(self._arm_position_lower)],
                self._arm_position_lower,
                self._arm_position_upper,
                strict=True,
            )
        ):
            state = states[index]
            if lower <= state.q <= upper:
                continue

            boundary = "lower" if state.q < lower else "upper"
            limit = lower if state.q < lower else upper
            overshoot = abs(state.q - limit)
            if overshoot > self._feedback_clamp_margin + 1e-12:
                self._latch_feedback_fault(
                    joint_name=joint_name,
                    value=state.q,
                    lower=lower,
                    upper=upper,
                )

            warning_key = (joint_name, boundary)
            if warning_key not in self._warned_feedback_clamps:
                logger.warning(
                    "Clamping OpenArm joint feedback at position limit",
                    joint=joint_name,
                    value=state.q,
                    limit=limit,
                )
                self._warned_feedback_clamps.add(warning_key)
            normalized[index] = MotorState(q=limit, dq=state.dq, tau=state.tau)
        return normalized

    def _latch_feedback_fault(
        self,
        *,
        joint_name: str,
        value: float,
        lower: float,
        upper: float,
    ) -> NoReturn:
        fault = (
            f"{joint_name} reported {value} outside [{lower}, {upper}] by more than "
            f"{self._feedback_clamp_margin} rad"
        )
        self._feedback_fault = fault
        disabled = self.deactivate()
        # Fail closed even when the transport could not confirm motor disable.
        self._active = False
        if not disabled:
            fault = f"{fault}; motor disable failed"
            self._feedback_fault = fault
        raise RuntimeError(f"OpenArm feedback fault: {fault}")

    @property
    def gravity_model_path(self) -> Path:
        """Return the lazy bimanual gravity-compensation URDF path."""
        return LfsPath("openarm_description") / "urdf/robot/openarm_v20_bimanual.urdf"

    def _build_robot(self) -> can_motor_control.Robot:
        return (
            can_motor_control.Robot.builder()
            .add_bus(
                "left",
                can_motor_control.SocketCanBus(self.bus_address("left")),
                damiao.DamiaoCodec(),
            )
            .add_bus(
                "right",
                can_motor_control.SocketCanBus(self.bus_address("right")),
                damiao.DamiaoCodec(),
            )
            .add_arm("left_arm", bus="left", motors=_arm_motors("left"))
            .add_arm("right_arm", bus="right", motors=_arm_motors("right"))
            .add_gripper(
                "left_gripper",
                bus="left",
                motor=_gripper_motor("left"),
                opening_direction="decreasing_position",
                default_current=0.15,
            )
            .add_gripper(
                "right_gripper",
                bus="right",
                motor=_gripper_motor("right"),
                opening_direction="decreasing_position",
                default_current=0.15,
            )
            .build()
        )
