cmake_minimum_required(VERSION 3.16)

# A6-ASC-FUSION: the ASC toolchain registers the ASC language (bisheng as the
# compiler driver) so the kernel sources below can build for dav cores the
# legacy ascendc.cmake route cannot describe. It must run before project(),
# because CMake only accepts languages that were registered beforehand.
#
# find_package(ASC) loads CANN's asc-config.cmake, which already aborts with a
# clear message when ASCEND_HOME_PATH is unset or missing. What it does not
# report is a package that resolved without registering the language, so check
# the module's own loaded flag before handing ASC to project().
find_package(ASC)
if(NOT _ASC_MODULE_LOADED)
    message(FATAL_ERROR
        "find_package(ASC) did not load CANN's asc-config.cmake, so the ASC "
        "language is not registered and the kernel sources cannot be "
        "compiled. Ensure ASCEND_HOME_PATH points at a CANN install that ships "
        "ascendc_kernel_cmake (CANN 9.2.0+ for this tree) and source both "
        "ascend-toolkit/set_env.sh and driver/bin/setenv.bash.")
endif()

project(vllm_ascend_C ASC CXX)

# CMakeDetermineASCCompiler.cmake locates bisheng while project() resolves the
# language; surface it so a toolchain without a usable compiler fails here
# rather than deep inside the kernel build.
if(NOT CMAKE_ASC_COMPILER)
    message(FATAL_ERROR
        "The ASC language was registered but no ASC compiler was found. "
        "CMakeDetermineASCCompiler.cmake looks for bisheng under "
        "ASCEND_CANN_PACKAGE_LINUX_PATH/ccec_compiler/bin, PATH and "
        "ASCEND_HOME_PATH.")
endif()
message(STATUS "ASC compiler: ${CMAKE_ASC_COMPILER}")

# include(CheckCXXcompilerFlag)
# check_cxx_compiler_flag("-std=c++17", COMPILER_SUPPORTS_CXX17)
set(CMAKE_CXX_STANDARD 17)

include(${CMAKE_CURRENT_LIST_DIR}/cmake/utils.cmake)

# Suppress potential warnings about unused manually-specified variables
set(ignoreMe "${VLLM_PYTHON_PATH}")

set(PYTHON_SUPPORTED_VERSIONS "3.9" "3.10" "3.11" "3.12")

find_package(pybind11 REQUIRED)

append_cmake_prefix_path("torch" "torch.utils.cmake_prefix_path")
set(VLLM_ASCEND_INSTALL_PATH "${CMAKE_INSTALL_PREFIX}")

find_package(Torch REQUIRED)

run_python(TORCH_VERSION
  "import torch; print(torch.__version__)" "Failed to locate torch path")
# check torch version is 2.10.0
if(NOT ${TORCH_VERSION} VERSION_EQUAL "2.10.0")
  message(FATAL_ERROR "Expected PyTorch version 2.10.0, but found ${TORCH_VERSION}")
endif()

set(RUN_MODE "npu" CACHE STRING "cpu/sim/npu")
set(SOC_VERSION ${SOC_VERSION})
message(STATUS "Detected SOC version: ${SOC_VERSION}")

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRINGS "Build type Release/Debug (default Release)" FORCE)
endif()

if (CMAKE_INSTALL_PREFIX STREQUAL /usr/local)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/out" CACHE STRINGS "path to install()")
endif()

# A6-ASC-FUSION: the legacy ascendc.cmake discovery is gone; the kernel target
# below is built by the ASC toolchain that find_package(ASC) set up above.
# The legacy route cannot serve A6 for two reasons. First, its host_config.cmake
# lowercases SOC_VERSION and then membership-tests it against the camelCase
# ascend960DT_list, so every ascend960DT_* value falls through to the
# FATAL_ERROR branch. Second, even when matched it maps A6 onto BUILD_MODE
# 310r6, while A6 pairs 64 cube:64 vector cores with cube_vector_combine=fuse
# and 310r6 does not describe that pairing.
set(ASCEND_CANN_PACKAGE_PATH ${ASCEND_HOME_PATH})

file(GLOB KERNEL_FILES
${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/*.cpp)

set(VLLM_ASCEND_CUSTOM_OP
    ${KERNEL_FILES}
)

# A6-ASC-FUSION: map SOC_VERSION to the dav core the ASC compiler expects.
# Every pattern must be a real prefix of an entry in CANN's SoC lists
# (host_config.cmake): ascend960DT_list, ascend950_list, ascend910b_list - which
# also carries the ascend910_93xx variants - and ascend910_list. MATCHES is an
# unanchored regex, so a pattern that is a prefix of an earlier branch is
# unreachable; keep the branches ordered from most to least specific.
# A2/A3/A5 therefore keep the same dav cores the legacy BUILD_MODE mapping gave
# them, and only A6 adds a new one.
if(SOC_VERSION MATCHES "ascend960")
    set(VLLM_ASCEND_ASC_NPU_ARCH "dav-9201")
elseif(SOC_VERSION MATCHES "ascend950")
    set(VLLM_ASCEND_ASC_NPU_ARCH "dav-c310")
elseif(SOC_VERSION MATCHES "ascend910_93|ascend910b")
    set(VLLM_ASCEND_ASC_NPU_ARCH "dav-c220")
elseif(SOC_VERSION MATCHES "ascend910")
    set(VLLM_ASCEND_ASC_NPU_ARCH "dav-c100")
endif()

if(SOC_VERSION MATCHES "ascend310p.*|ascend950")
    message(STATUS "Hardware ${SOC_VERSION} detected: skip vllm_ascend_kernels compile")
else()
    # A6-ASC-FUSION: mark the kernel sources as the ASC language, then build
    # them as an ordinary CMake target compiled for the target dav core.
    set_source_files_properties(
        ${VLLM_ASCEND_CUSTOM_OP}
        PROPERTIES LANGUAGE ASC
    )
    add_library(vllm_ascend_kernels SHARED
        ${VLLM_ASCEND_CUSTOM_OP}
    )
    if(DEFINED VLLM_ASCEND_ASC_NPU_ARCH)
        message(STATUS "Building AscendC kernels for ${VLLM_ASCEND_ASC_NPU_ARCH} (SOC ${SOC_VERSION})")
        target_compile_options(vllm_ascend_kernels PRIVATE
            $<$<COMPILE_LANGUAGE:ASC>: --npu-arch=${VLLM_ASCEND_ASC_NPU_ARCH}>
        )
    endif()
endif()

message("TORCH_NPU_PATH is ${TORCH_NPU_PATH}")

if(SOC_VERSION MATCHES "ascend310p.*")
    file(GLOB VLLM_ASCEND_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/aclnn_torch_adapter/*.cpp)
else()
    file(GLOB VLLM_ASCEND_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/aclnn_torch_adapter/*.cpp)
endif()

# Sparse KV cache offloading is currently supported on Atlas A3 only.
if(SOC_VERSION MATCHES "ascend910_93.*")
  find_package(OpenMP REQUIRED)
endif()

include_directories(
  ${pybind11_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_PATH}
  ${TORCH_INCLUDE_DIRS}
  # Prefer the selected CANN toolkit headers over the ACL snapshot bundled in
  # some torch_npu wheels. Feature checks below use ASCEND_HOME_PATH as well.
  ${ASCEND_HOME_PATH}/include
  ${TORCH_NPU_PATH}/include
)

set(
  INCLUDES
  ${TORCH_INCLUDE_DIRS}
  ${TORCH_NPU_INCLUDE_DIRS}
  ${ASCEND_HOME_PATH}/include
)

pybind11_add_module(vllm_ascend_C ${VLLM_ASCEND_SRC})
if(SOC_VERSION MATCHES "ascend910_93.*")
  target_compile_definitions(vllm_ascend_C PRIVATE VLLM_ASCEND_ENABLE_SPARSE_KV_OFFLOAD)
  target_link_libraries(vllm_ascend_C PRIVATE OpenMP::OpenMP_CXX)
endif()

# Detect aclrtMemcpyBatchAsync availability (CANN 8.5+)
# Can be overridden via VLLM_ASCEND_ENABLE_BATCH_MEMCPY env var (registered
# in vllm_ascend/envs.py, forwarded by setup.py as a CMake variable):
#   VLLM_ASCEND_ENABLE_BATCH_MEMCPY=1  -> force enable
#   VLLM_ASCEND_ENABLE_BATCH_MEMCPY=0  -> force disable
#   unset                               -> auto-detect from CANN headers
include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_INCLUDES ${ASCEND_HOME_PATH}/include)
set(CMAKE_REQUIRED_LIBRARIES ascendcl)
set(CMAKE_REQUIRED_LINK_OPTIONS "-L${ASCEND_HOME_PATH}/lib64")

# Keep builds on older CANN versions working; endpoint selection must fail
# explicitly if the installed CANN lacks the user-to-physical conversion.
check_cxx_source_compiles("
  #include <acl/acl_rt.h>
  int main() {
    int32_t physical_id = 0;
    return aclrtGetPhyDevIdByUserDevId(0, &physical_id);
  }
" HAVE_ACLRT_GET_PHY_DEV_ID_BY_USER_DEV_ID)
if(HAVE_ACLRT_GET_PHY_DEV_ID_BY_USER_DEV_ID)
  target_compile_definitions(vllm_ascend_C PRIVATE CANN_DEVICE_ID_MAPPING)
endif()

if(DEFINED VLLM_ASCEND_ENABLE_BATCH_MEMCPY)
  if("${VLLM_ASCEND_ENABLE_BATCH_MEMCPY}" STREQUAL "1")
    message(STATUS "aclrtMemcpyBatchAsync: force enabled via VLLM_ASCEND_ENABLE_BATCH_MEMCPY=1")
    target_compile_definitions(vllm_ascend_C PRIVATE CANN_MEMCPY_BATCH_ASYNC)
  else()
    message(STATUS "aclrtMemcpyBatchAsync: force disabled via VLLM_ASCEND_ENABLE_BATCH_MEMCPY=0")
  endif()
else()
  # Test the full code pattern we actually use, including struct member access.
  # This ensures the macro is only defined when the API is fully compatible.
  check_cxx_source_compiles("
    #include <acl/acl_rt.h>
    int main() {
      aclrtMemLocation loc = {};
      loc.type = ACL_MEM_LOCATION_TYPE_HOST;
      loc.id = 0;
      aclrtMemcpyBatchAttr attr = {};
      attr.srcLoc = loc;
      attr.dstLoc = loc;
      (void)aclrtMemcpyBatchAsync;
      return 0;
    }
  " HAVE_ACLRT_MEMCPY_BATCH_ASYNC)
  if(HAVE_ACLRT_MEMCPY_BATCH_ASYNC)
    message(STATUS "aclrtMemcpyBatchAsync: detected in CANN headers, enabling batch memcpy path")
    target_compile_definitions(vllm_ascend_C PRIVATE CANN_MEMCPY_BATCH_ASYNC)
  else()
    message(STATUS "aclrtMemcpyBatchAsync: not found in CANN headers, using fallback aclrtMemcpyAsync loop")
  endif()
endif()

# ECCPUConnector stores encoder-cache data in a shared mmap. Newer CANN releases
# expose aclrtHostRegisterV2(..., ACL_HOST_REG_PINNED), which turns those mmap
# pages into real page-locked host memory. Keep the custom op available on older
# CANN releases as well, but make it fail with an actionable error at runtime
# instead of silently claiming that pageable memory is pinned.
check_cxx_source_compiles("
  #include <acl/acl_rt.h>
  int main() {
    void* ptr = nullptr;
    (void)aclrtHostRegisterV2(ptr, 0, ACL_HOST_REG_PINNED);
    (void)aclrtHostUnregister;
    return 0;
  }
" HAVE_ACLRT_HOST_REGISTER_V2_PINNED)
if(HAVE_ACLRT_HOST_REGISTER_V2_PINNED)
  message(STATUS "aclrtHostRegisterV2: pinned host registration available")
  target_compile_definitions(vllm_ascend_C PRIVATE CANN_HOST_REGISTER_V2_PINNED)
else()
  message(STATUS "aclrtHostRegisterV2: pinned host registration unavailable; ECCPUConnector will be rejected")
endif()

if(SOC_VERSION MATCHES "ascend310p.*")
    target_compile_definitions(vllm_ascend_C PRIVATE -DASCEND_PLATFORM_310P)
endif()

if(NOT (SOC_VERSION MATCHES "ascend310p.*|ascend950"))
    target_compile_definitions(vllm_ascend_C PRIVATE -DVLLM_ENABLE_ATB_AND_DIRECT_KERNELS)
endif()

target_link_directories(
  vllm_ascend_C
  PRIVATE
  ${TORCH_LIBRARY_DIRS}
  ${TORCH_NPU_PATH}/lib/
  ${ASCEND_HOME_PATH}/lib64
)

set(VLLM_ASCEND_C_COMMON_LIBS
  ${TORCH_LIBRARIES}
  torch_npu
  ascendcl
  tiling_api
  register
  platform
  ascendalog
  dl
  opapi
)

if(SOC_VERSION MATCHES "ascend310p.*|ascend950")
  target_link_libraries(
    vllm_ascend_C
    PUBLIC
    ${VLLM_ASCEND_C_COMMON_LIBS}
  )
else()
  target_link_libraries(
    vllm_ascend_C
    PUBLIC
    vllm_ascend_kernels
    ${VLLM_ASCEND_C_COMMON_LIBS}
  )
endif()

target_link_options(vllm_ascend_C PRIVATE "-Wl,-rpath,$ORIGIN:$ORIGIN/lib:$ORIGIN/_cann_ops_custom/vendors/custom_transformer/op_api/lib")

if(SOC_VERSION MATCHES "ascend310p.*|ascend950")
  install(TARGETS vllm_ascend_C DESTINATION ${VLLM_ASCEND_INSTALL_PATH})
else()
  install(TARGETS vllm_ascend_C vllm_ascend_kernels DESTINATION ${VLLM_ASCEND_INSTALL_PATH})
endif()
