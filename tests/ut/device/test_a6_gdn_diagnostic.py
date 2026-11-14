# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm_ascend import envs
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import HardwareCapability, get_hardware_profile


def test_a6_avoids_fla_gdn_prefill() -> None:
    capability = HardwareCapability.FLA_GDN_PREFILL
    assert not get_hardware_profile(AscendDeviceType.A6).supports(capability)
    assert get_hardware_profile(AscendDeviceType.A5).supports(capability)


def test_dummy_gdn_skip_requires_explicit_opt_in(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_ASCEND_DUMMY_SKIP_GDN", raising=False)
    assert not envs.VLLM_ASCEND_DUMMY_SKIP_GDN

    monkeypatch.setenv("VLLM_ASCEND_DUMMY_SKIP_GDN", "1")
    assert envs.VLLM_ASCEND_DUMMY_SKIP_GDN
