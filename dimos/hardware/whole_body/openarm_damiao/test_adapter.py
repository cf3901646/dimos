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

from collections.abc import Iterator
import runpy

import can_motor_control
import numpy as np
import pinocchio
import pytest
from pytest_mock import MockerFixture

from dimos.hardware.whole_body.damiao.adapter import DamiaoWholeBodyAdapter
from dimos.hardware.whole_body.damiao.config import DamiaoRuntimeConfig
from dimos.hardware.whole_body.openarm_damiao import adapter as adapter_module
from dimos.hardware.whole_body.openarm_damiao.adapter import OpenArmDamiaoAdapter
from dimos.hardware.whole_body.spec import MotorCommand, MotorState
from dimos.robot.manipulators.openarm.config import OPENARM_DOF, OPENARM_JOINTS


def _activate_without_hardware_feedback(adapter: DamiaoWholeBodyAdapter) -> bool:
    adapter._active = True
    return True


@pytest.fixture
def openarm_adapter(mocker: MockerFixture) -> Iterator[OpenArmDamiaoAdapter]:
    mocker.patch.object(can_motor_control, "SocketCanBus", can_motor_control.MockCanBus)
    adapter = OpenArmDamiaoAdapter(
        runtime_config=DamiaoRuntimeConfig(gravity_comp=False),
    )
    yield adapter
    adapter.disconnect()


def test_import_lazy_gravity_model_does_not_resolve_lfs(mocker: MockerFixture) -> None:
    get_data = mocker.patch("dimos.utils.data.get_data")

    runpy.run_path(adapter_module.__file__)

    get_data.assert_not_called()


def test_openarm_topology_connects_arms_and_grippers(
    openarm_adapter: OpenArmDamiaoAdapter,
) -> None:
    robot = openarm_adapter._build_robot()

    assert robot.group_names() == ["left_arm", "right_arm", "left_gripper", "right_gripper"]
    assert robot.bus_names() == ["left", "right"]
    assert isinstance(robot["left_arm"], can_motor_control.Arm)
    assert isinstance(robot["right_arm"], can_motor_control.Arm)
    assert len(robot["left_arm"]) == OPENARM_DOF
    assert len(robot["right_arm"]) == OPENARM_DOF
    assert isinstance(robot["left_gripper"], can_motor_control.Gripper)
    assert isinstance(robot["right_gripper"], can_motor_control.Gripper)
    assert openarm_adapter.connect()


def test_openarm_joint_order_matches_hardware_component(
    openarm_adapter: OpenArmDamiaoAdapter,
) -> None:
    """Commands are routed positionally: the config joint list must equal the
    adapter's declared order or motors silently receive each other's targets."""
    assert list(openarm_adapter.joint_names) == OPENARM_JOINTS


def test_openarm_clamps_small_arm_feedback_overshoot_at_single_adapter_gate(
    openarm_adapter: OpenArmDamiaoAdapter,
    mocker: MockerFixture,
) -> None:
    states = [MotorState(q=0.0, dq=0.1, tau=0.2) for _ in OPENARM_JOINTS]
    states[3] = MotorState(q=-0.0011, dq=0.3, tau=0.4)
    states[12] = MotorState(q=0.7864, dq=0.5, tau=0.6)
    states[14] = MotorState(q=0.25, dq=0.7, tau=0.8)
    states[15] = MotorState(q=0.75, dq=0.9, tau=1.0)
    mocker.patch.object(DamiaoWholeBodyAdapter, "read_motor_states", return_value=states)

    result = openarm_adapter.read_motor_states()

    assert result[3] == MotorState(q=0.0, dq=0.3, tau=0.4)
    assert result[12] == MotorState(q=0.7854, dq=0.5, tau=0.6)
    assert result[14:] == states[14:]


@pytest.mark.parametrize("overshoot", [0.05, -0.05])
def test_openarm_accepts_feedback_exactly_at_clamp_margin(
    openarm_adapter: OpenArmDamiaoAdapter,
    mocker: MockerFixture,
    overshoot: float,
) -> None:
    states = [MotorState(q=0.0) for _ in OPENARM_JOINTS]
    states[3] = MotorState(q=overshoot if overshoot < 0.0 else 2.4435 + overshoot)
    mocker.patch.object(DamiaoWholeBodyAdapter, "read_motor_states", return_value=states)

    result = openarm_adapter.read_motor_states()

    assert result[3].q == pytest.approx(0.0 if overshoot < 0.0 else 2.4435)


def test_openarm_gross_feedback_violation_disables_and_latches_adapter(
    openarm_adapter: OpenArmDamiaoAdapter,
    mocker: MockerFixture,
) -> None:
    assert openarm_adapter.connect()
    mocker.patch.object(
        DamiaoWholeBodyAdapter,
        "activate",
        autospec=True,
        side_effect=_activate_without_hardware_feedback,
    )
    assert openarm_adapter.activate()
    states = [MotorState(q=0.0) for _ in OPENARM_JOINTS]
    states[3] = MotorState(q=-0.051)
    mocker.patch.object(DamiaoWholeBodyAdapter, "read_motor_states", return_value=states)
    deactivate = mocker.patch.object(openarm_adapter, "deactivate", return_value=True)

    with pytest.raises(RuntimeError, match=r"left_arm/joint4.*-0.051.*\[0.0, 2.4435\]"):
        openarm_adapter.read_motor_states()

    deactivate.assert_called_once_with()
    assert not openarm_adapter.activate()
    assert not openarm_adapter.write_motor_commands([MotorCommand()] * len(OPENARM_JOINTS))


def test_openarm_feedback_fault_fails_closed_when_disable_fails(
    openarm_adapter: OpenArmDamiaoAdapter,
    mocker: MockerFixture,
) -> None:
    assert openarm_adapter.connect()
    mocker.patch.object(
        DamiaoWholeBodyAdapter,
        "activate",
        autospec=True,
        side_effect=_activate_without_hardware_feedback,
    )
    assert openarm_adapter.activate()
    states = [MotorState(q=0.0) for _ in OPENARM_JOINTS]
    states[3] = MotorState(q=-0.051)
    mocker.patch.object(DamiaoWholeBodyAdapter, "read_motor_states", return_value=states)
    mocker.patch.object(openarm_adapter, "deactivate", return_value=False)

    with pytest.raises(RuntimeError, match="feedback fault"):
        openarm_adapter.read_motor_states()

    assert not openarm_adapter._active
    assert not openarm_adapter.activate()


def test_openarm_reconnect_clears_feedback_fault(
    openarm_adapter: OpenArmDamiaoAdapter,
    mocker: MockerFixture,
) -> None:
    assert openarm_adapter.connect()
    mocker.patch.object(
        DamiaoWholeBodyAdapter,
        "activate",
        autospec=True,
        side_effect=_activate_without_hardware_feedback,
    )
    assert openarm_adapter.activate()
    states = [MotorState(q=0.0) for _ in OPENARM_JOINTS]
    states[3] = MotorState(q=-0.051)
    read_states = mocker.patch.object(
        DamiaoWholeBodyAdapter,
        "read_motor_states",
        side_effect=[states, [MotorState(q=0.0) for _ in OPENARM_JOINTS]],
    )
    mocker.patch.object(openarm_adapter, "deactivate", return_value=True)
    with pytest.raises(RuntimeError, match="feedback fault"):
        openarm_adapter.read_motor_states()
    openarm_adapter.disconnect()

    assert openarm_adapter.connect()
    assert openarm_adapter.activate()
    assert openarm_adapter.read_motor_states() == [MotorState(q=0.0) for _ in OPENARM_JOINTS]
    assert read_states.call_count == 2


@pytest.mark.self_hosted
def test_openarm_feedback_limits_match_urdf_joint_limits(
    openarm_adapter: OpenArmDamiaoAdapter,
) -> None:
    model = pinocchio.buildModelFromUrdf(str(openarm_adapter.gravity_model_path))

    assert openarm_adapter._arm_position_lower == pytest.approx(
        np.asarray(model.lowerPositionLimit)
    )
    assert openarm_adapter._arm_position_upper == pytest.approx(
        np.asarray(model.upperPositionLimit)
    )
