# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import HardwareCapability, get_hardware_profile


def test_a6_avoids_fla_gdn_prefill() -> None:
    capability = HardwareCapability.FLA_GDN_PREFILL
    assert not get_hardware_profile(AscendDeviceType.A6).supports(capability)
    assert get_hardware_profile(AscendDeviceType.A5).supports(capability)
