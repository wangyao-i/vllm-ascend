# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# A6 uses CANN 9.3 SparseFlashAttention. Its schema and host registrations
# must come from the SDK, including when building --ops=ALL.
if("ascend960dt" IN_LIST ASCEND_COMPUTE_UNIT OR "ascend960pr" IN_LIST ASCEND_COMPUTE_UNIT)
    set(SFA_NON_A6_UNITS ${ASCEND_COMPUTE_UNIT})
    list(REMOVE_ITEM SFA_NON_A6_UNITS ascend960dt ascend960pr)
    if(SFA_NON_A6_UNITS)
        message(FATAL_ERROR "Build A6 and other SoCs in separate custom-OPP packages.")
    endif()
    message(STATUS "Using SDK SparseFlashAttention for A6")
    return()
endif()

file(GLOB CURRENT_DIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)
if(NOT ENABLE_TEST AND NOT BENCHMARK)
    list(REMOVE_ITEM CURRENT_DIRS tests)
endif()
foreach(SUB_DIR ${CURRENT_DIRS})
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${SUB_DIR}/CMakeLists.txt")
        add_subdirectory(${SUB_DIR})
    endif()
endforeach()
