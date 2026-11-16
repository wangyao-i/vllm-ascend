/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_lightning_indexer_v2_metadata.h
 * \brief
 */

#ifndef QUANT_LIGHTNING_INDEXER_V2_METADATA_H
#define QUANT_LIGHTNING_INDEXER_V2_METADATA_H

#include <cstdint>

namespace optiling {

// Constants
// A6-ASC-FUSION: Ascend960DT(968B7) 实测 AIC=64 / AIV=64
// (torch.npu.get_device_properties -> cube_core_num=64, vector_core_num=64)。
// 原值 36/72 是 A5 规格；A6 走 in-tree 路线会以 64 核调用本算子，触发
//   AclNN_Parameter_Error(EZ1001): The maximum supported AIC num is 80, but got 64
// 64*8(Qli) + 64*8(Qld) = 1024 = QLI_V2_METADATA_TOTAL_SIZE，正好装满，说明 64/64 是本布局的满配。
// hidden: 这两个符号与 CANN 核心库 libopapi.so 同名(其值 80)，不隐藏会被动态链接期抢占，
// 出现"判定用 36、日志打印 80"的自相矛盾输出，排查会被带偏。
inline constexpr uint32_t AIC_CORE_MAX_NUM __attribute__((visibility("hidden"))) = 64;
inline constexpr uint32_t AIV_CORE_MAX_NUM __attribute__((visibility("hidden"))) = 64;
constexpr uint32_t QLI_V2_METADATA_TOTAL_SIZE = 1024;
using QLI_V2_METADATA_T = int32_t;

inline constexpr uint32_t QLI_V2_METADATA_SIZE = 8;
inline constexpr uint32_t QLD_V2_METADATA_SIZE = 8;

// LI Metadata Index Definitions
inline constexpr uint32_t QLI_V2_CORE_ENABLE_INDEX = 0;
inline constexpr uint32_t QLI_V2_BN2_START_INDEX = 1;
inline constexpr uint32_t QLI_V2_M_START_INDEX = 2;
inline constexpr uint32_t QLI_V2_S2_START_INDEX = 3;
inline constexpr uint32_t QLI_V2_BN2_END_INDEX = 4;
inline constexpr uint32_t QLI_V2_M_END_INDEX = 5;
inline constexpr uint32_t QLI_V2_S2_END_INDEX = 6;
inline constexpr uint32_t QLI_V2_FIRST_QLD_V2_DATA_WORKSPACE_IDX_INDEX = 7;

// LD Metadata Index Definitions
inline constexpr uint32_t QLD_V2_CORE_ENABLE_INDEX = 0;
inline constexpr uint32_t QLD_V2_BN2_IDX_INDEX = 1;
inline constexpr uint32_t QLD_V2_M_IDX_INDEX = 2;
inline constexpr uint32_t QLD_V2_WORKSPACE_IDX_INDEX = 3;
inline constexpr uint32_t QLD_V2_WORKSPACE_NUM_INDEX = 4;
inline constexpr uint32_t QLD_V2_M_START_INDEX = 5;
inline constexpr uint32_t QLD_V2_M_NUM_INDEX = 6;

/**
 * @brief 获取属性的绝对索引
 * @param coreIdx 核索引
 * @param metaIdx 元数据索引
 * @param isAIV 是否为AIV数据，默认为false
 * @return 返回属性的绝对索引
 */
#ifdef __CCE_AICORE__
__aicore__ inline uint32_t GetAttrAbsIndex(uint32_t coreIdx, uint32_t metaIdx, bool isAIV = false)
{
    if (isAIV) {
        return QLI_V2_METADATA_SIZE * AIC_CORE_MAX_NUM + QLD_V2_METADATA_SIZE * coreIdx + metaIdx;
    } else {
        return QLI_V2_METADATA_SIZE * coreIdx + metaIdx;
    }
}
#endif

namespace detail {
struct QliV2Metadata {
    uint32_t qliV2Metadata[AIC_CORE_MAX_NUM][QLI_V2_METADATA_SIZE];
    uint32_t qldV2Metadata[AIV_CORE_MAX_NUM][QLD_V2_METADATA_SIZE];
};
}; // namespace detail

static_assert(QLI_V2_METADATA_TOTAL_SIZE * sizeof(QLI_V2_METADATA_T) >= sizeof(detail::QliV2Metadata));
}; // namespace optiling

#endif // QUANT_LIGHTNING_INDEXER_V2_METADATA_H
