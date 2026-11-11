/**
 * Compatibility shim for CANN 9.2.0 predefined error-report macros.
 *
 * These macros interpolate their arguments into
 *
 *     const std::vector<const char*> msgvalue = {opName, paramNames, ...};
 *
 * so every argument must be const char*. Tiling code here passes std::string and
 * cannot bind. Include this header after the CANN headers: it redefines the
 * affected macros with the CANN bodies unchanged except that each argument goes
 * through VC(), which returns const char* unchanged and stages std::string text
 * in a per-thread ring so the pointer stays valid for the whole expansion. The
 * error codes reported to REPORT_PREDEFINED_ERR_MSG() are unchanged.
 *
 * Macros whose CANN bodies already copy their arguments into std::string locals
 * (OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON, ..._SHAPEDIM_WITH_REASON) accept
 * std::string natively and are left alone.
 */

#ifndef VLLM_ASCEND_OPS_LOG_COMPAT_H
#define VLLM_ASCEND_OPS_LOG_COMPAT_H

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <string>

namespace vllm_ascend_log_compat {

/**
 * Return @p value unchanged.
 *
 * @param value A C string owned by the caller.
 * @returns The same pointer.
 */
inline const char *VC(const char *value) { return value; }

/**
 * Expose @p value as a C string that outlives one macro expansion.
 *
 * Text is copied into a per-thread ring of 24 slots of 2 KiB, which is far more than
 * a single four-argument macro can consume. @p value is truncated to fit a slot.
 *
 * @param value Text to expose as a C string.
 * @returns Pointer valid until that slot is reused by a later call on this thread.
 */
inline const char *VC(const std::string &value)
{
    constexpr size_t kSlots = 24;
    constexpr size_t kSlotSize = 2048;
    static thread_local std::array<std::array<char, kSlotSize>, kSlots> slots{};
    static thread_local size_t next = 0;
    auto &slot = slots[next];
    next = (next + 1) % kSlots;
    const size_t n = std::min(value.size(), kSlotSize - 1);
    std::memcpy(slot.data(), value.data(), n);
    slot[n] = '\0';
    return slot.data();
}

}  // namespace vllm_ascend_log_compat

#define VLLM_ASCEND_VC(x) ::vllm_ascend_log_compat::VC(x)

#ifdef OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON
#undef OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON
#define OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(opName, paramNames, incorrectShapes, reason)                    \
    do {                                                                                                      \
        const char *_vcOpName_ = VLLM_ASCEND_VC(opName);                                                       \
        const char *_vcParamNames_ = VLLM_ASCEND_VC(paramNames);                                               \
        const char *_vcIncorrectShapes_ = VLLM_ASCEND_VC(incorrectShapes);                                     \
        const char *_vcReason_ = VLLM_ASCEND_VC(reason);                                                       \
        OP_LOGE_WITHOUT_REPORT(                                                                                \
            _vcOpName_, "OP %s's parameters %s have incorrect shapes %s. Reason: %s.", _vcOpName_,             \
            _vcParamNames_, _vcIncorrectShapes_, _vcReason_);                                                  \
        const std::vector<const char *> msgKey = {"op_name", "param_names", "incorrect_shapes", "reason"};     \
        const std::vector<const char *> msgvalue = {_vcOpName_, _vcParamNames_, _vcIncorrectShapes_, _vcReason_}; \
        REPORT_PREDEFINED_ERR_MSG("EZ0010", msgKey, msgvalue);                                                 \
    } while (0)
#endif

#ifdef OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON
#undef OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON
#define OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(opName, paramNames, incorrectSizes, reason)                 \
    do {                                                                                                       \
        const char *_vcOpName_ = VLLM_ASCEND_VC(opName);                                                        \
        const char *_vcParamNames_ = VLLM_ASCEND_VC(paramNames);                                                \
        const char *_vcIncorrectSizes_ = VLLM_ASCEND_VC(incorrectSizes);                                        \
        const char *_vcReason_ = VLLM_ASCEND_VC(reason);                                                        \
        OP_LOGE_WITHOUT_REPORT(                                                                                 \
            _vcOpName_, "OP %s's parameters %s have incorrect shape sizes %s. Reason: %s.", _vcOpName_,          \
            _vcParamNames_, _vcIncorrectSizes_, _vcReason_);                                                    \
        const std::vector<const char *> msgKey = {"op_name", "param_names", "incorrect_sizes", "reason"};       \
        const std::vector<const char *> msgvalue = {_vcOpName_, _vcParamNames_, _vcIncorrectSizes_, _vcReason_}; \
        REPORT_PREDEFINED_ERR_MSG("EZ0015", msgKey, msgvalue);                                                  \
    } while (0)
#endif

#ifdef OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON
#undef OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON
#define OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(opName, paramNames, incorrectDtypes, reason)                    \
    do {                                                                                                       \
        const char *_vcOpName_ = VLLM_ASCEND_VC(opName);                                                        \
        const char *_vcParamNames_ = VLLM_ASCEND_VC(paramNames);                                                \
        const char *_vcIncorrectDtypes_ = VLLM_ASCEND_VC(incorrectDtypes);                                      \
        const char *_vcReason_ = VLLM_ASCEND_VC(reason);                                                        \
        OP_LOGE_WITHOUT_REPORT(                                                                                 \
            _vcOpName_, "OP %s's parameters %s have incorrect dtypes %s. Reason: %s.", _vcOpName_,               \
            _vcParamNames_, _vcIncorrectDtypes_, _vcReason_);                                                   \
        const std::vector<const char *> msgKey = {"op_name", "param_names", "incorrect_dtypes", "reason"};      \
        const std::vector<const char *> msgvalue = {_vcOpName_, _vcParamNames_, _vcIncorrectDtypes_, _vcReason_}; \
        REPORT_PREDEFINED_ERR_MSG("EZ0020", msgKey, msgvalue);                                                  \
    } while (0)
#endif

#ifdef OP_LOGE_FOR_INVALID_VALUES_WITH_REASON
#undef OP_LOGE_FOR_INVALID_VALUES_WITH_REASON
#define OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(opName, paramNames, incorrectValues, reason)                    \
    do {                                                                                                       \
        const char *_vcOpName_ = VLLM_ASCEND_VC(opName);                                                        \
        const char *_vcParamNames_ = VLLM_ASCEND_VC(paramNames);                                                \
        const char *_vcIncorrectValues_ = VLLM_ASCEND_VC(incorrectValues);                                      \
        const char *_vcReason_ = VLLM_ASCEND_VC(reason);                                                        \
        OP_LOGE_WITHOUT_REPORT(                                                                                 \
            _vcOpName_, "OP %s's parameters %s have incorrect values %s. Reason: %s.", _vcOpName_,               \
            _vcParamNames_, _vcIncorrectValues_, _vcReason_);                                                   \
        const std::vector<const char *> msgKey = {"op_name", "param_names", "incorrect_values", "reason"};      \
        const std::vector<const char *> msgvalue = {_vcOpName_, _vcParamNames_, _vcIncorrectValues_, _vcReason_}; \
        REPORT_PREDEFINED_ERR_MSG("EZ0024", msgKey, msgvalue);                                                  \
    } while (0)
#endif

// Some call sites use the singular spelling, which other CANN releases provided.
#ifndef OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON
#define OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(opName, paramNames, incorrectSizes, reason) \
    OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(opName, paramNames, incorrectSizes, reason)
#endif

#endif  // VLLM_ASCEND_OPS_LOG_COMPAT_H
