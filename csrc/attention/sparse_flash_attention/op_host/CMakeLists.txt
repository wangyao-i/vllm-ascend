# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# Dependencies may add op_host directly, bypassing the operator CMakeLists.
if("ascend960dt" IN_LIST ASCEND_COMPUTE_UNIT OR "ascend960pr" IN_LIST ASCEND_COMPUTE_UNIT)
    set(SFA_NON_A6_UNITS ${ASCEND_COMPUTE_UNIT})
    list(REMOVE_ITEM SFA_NON_A6_UNITS ascend960dt ascend960pr)
    if(SFA_NON_A6_UNITS)
        message(FATAL_ERROR "Build A6 and other SoCs in separate custom-OPP packages.")
    endif()
    return()
endif()

add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    set(sparse_flash_attention_depends attention/common CACHE INTERNAL "Dependencies for sparse_flash_attention")
    target_sources(op_host_aclnnInner PRIVATE
        sparse_flash_attention_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME SparseFlashAttention
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
            -mllvm -cce-vf-remove-membar=false
            -mllvm -cce-aicore-hoist-movemask=false
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE sparse_flash_attention ACLNNTYPE aclnn_inner)
endif()
