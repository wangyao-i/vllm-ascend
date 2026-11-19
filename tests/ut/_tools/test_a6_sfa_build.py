# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Check host-registration selection with CMake, without compiling NPU code."""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class TestA6SFABuild(unittest.TestCase):
    def run_entry(self, units, entry):
        cmake = shutil.which("cmake")
        if cmake is None:
            self.skipTest("cmake is required for the build-selection check")
        repo = Path(__file__).resolve().parents[3]
        source = repo / "csrc/attention/sparse_flash_attention"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "check.cmake"
            path.write_text(
                "cmake_minimum_required(VERSION 3.16)\n"
                f'set(ASCEND_COMPUTE_UNIT "{units}")\n'
                f'set(CMAKE_CURRENT_SOURCE_DIR "{source.as_posix()}")\n'
                "set(BUILD_OPEN_PROJECT OFF)\n"
                "function(add_subdirectory directory)\n"
                '  message(STATUS "SELECTED:${directory}")\n'
                "endfunction()\n"
                "function(add_op_to_compiled_list)\n"
                '  message(STATUS "REGISTERED:SFA")\n'
                "endfunction()\n"
                "function(add_ops_compile_options)\nendfunction()\n"
                "function(add_modules_sources)\nendfunction()\n"
                f'include("{(source / entry).as_posix()}")\n',
                encoding="utf-8",
            )
            return subprocess.run([cmake, "-P", str(path)], capture_output=True, text=True, timeout=30)

    def test_a6_skips_host_registration_at_both_entries(self):
        for units in ["ascend960dt", "ascend960pr", "ascend960dt;ascend960pr"]:
            for entry in ["CMakeLists.txt", "op_host/CMakeLists.txt"]:
                with self.subTest(units=units, entry=entry):
                    result = self.run_entry(units, entry)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertNotIn("SELECTED:", result.stdout)
                    self.assertNotIn("REGISTERED:", result.stdout)

    def test_other_socs_keep_host_registration(self):
        for units in ["ascend950", "ascend910b", "ascend910_93"]:
            for entry in ["CMakeLists.txt", "op_host/CMakeLists.txt"]:
                with self.subTest(units=units, entry=entry):
                    result = self.run_entry(units, entry)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    expected = "SELECTED:op_host" if entry == "CMakeLists.txt" else "REGISTERED:SFA"
                    self.assertIn(expected, result.stdout)

    def test_mixed_soc_packages_require_separate_builds(self):
        for entry in ["CMakeLists.txt", "op_host/CMakeLists.txt"]:
            with self.subTest(entry=entry):
                result = self.run_entry("ascend950;ascend960dt", entry)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("separate custom-OPP packages", result.stderr)


if __name__ == "__main__":
    unittest.main()
