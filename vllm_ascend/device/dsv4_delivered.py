# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
"""Delivered-package layer for the DSv4 path (real A6 only).

Every DSv4 fused op here is the delivered package's (`npu_ops_transformer` on CANN
9.2, `cann_ops_transformer` on 9.3): the in-tree `_C_ascend` kernels register on
this build, but their aclnn entry points are not installed, and the first one
called (`aclnnHcPre`) dies with "not in libopapi.so".

The shims are ported from the layer that already ran DSv4-Flash on this machine
(the dev fork's `device/dsa_cann_ops.py`), minus that fork's selection machinery:
no op-selection layer and no stand-in whitelist. Each shim documents
the contract difference it closes.
"""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch_npu
from vllm.logger import logger

_OT_PACKAGE = "npu_ops_transformer"

# Op key of the compressor in the delivered package (also the name the shim resolves).
_COMPRESSOR_OP_KEY = "compressor"

_VISION_GATING_WARNED = False


_OT_PACKAGE_PREFERRED = {
    "compressor": "cann_ops_transformer",
    "quant_compressor": "cann_ops_transformer",
}


_OT_OP_NAMES = {
    "compressor": "npu_compressor",
    "quant_compressor": "quant_compressor",
    "kv_compress_epilog": "kv_compress_epilog",
    "indexer_quant_cache": "indexer_quant_cache",
    "inplace_partial_rotary_mul": "inplace_partial_rotary_mul",
    "mhc_pre_sinkhorn": "mhc_pre_sinkhorn",
    "kv_quant_sparse_attn_sharedkv": "npu_mixed_quant_sparse_flash_mla",
    # The delivery registers both halves of the sparse-attention pair, so the metadata
    # companion is mapped too -- leaving it unmapped would put the compressed plan's tiling on
    # an undeclared path, which
    # would have left the compressed plan's metadata on the self-built/fabricated path forever.
    "kv_quant_sparse_attn_sharedkv_metadata": "npu_mixed_quant_sparse_flash_mla_metadata",
    "quant_lightning_indexer": "npu_quant_lightning_indexer",
    "quant_lightning_indexer_metadata": "npu_quant_lightning_indexer_metadata",
    "lightning_indexer": "npu_lightning_indexer",
}


_OT_OPS: dict[str, Any] = {}


_OT_PYTHON_INTERFACE = frozenset({"compressor"})


_SYNCED_FIRST_CALL: set[str] = set()


def ot_ops(package: str) -> Any:
    """The ``<package>.ops`` module, or None when the package is not installed."""
    if package not in _OT_OPS:
        try:
            mod = importlib.import_module(f"{package}.ops")
        except Exception:  # noqa: BLE001 - not installed / import failure
            mod = None
        _OT_OPS[package] = mod
    return _OT_OPS[package]


def _ot_package_for(name: str) -> str:
    """The delivered package to resolve ``name`` from.

    ``_OT_PACKAGE_PREFERRED`` names the ops whose 9.3.0 spelling is the only ABI-correct one
    *when that package exists*. On a 9.2.0 box it does not, and these ops have to keep resolving
    through ``npu_ops_transformer``; ``ot_ops`` memoizes the failed import, so asking costs one
    import attempt per process.
    """
    preferred = _OT_PACKAGE_PREFERRED.get(name)
    if preferred is not None and ot_ops(preferred) is not None:
        return preferred
    return _OT_PACKAGE


def _ot_op_in(name: str) -> Callable[..., Any] | None:
    """Resolve the delivered op our shim calls ``name``.

    ``name`` is this repository's name for the op; the delivered spelling comes from
    ``_OT_OP_NAMES`` (see above).

    Two entries can serve the same op:

    * ``torch.ops.npu_ops_transformer.<registered>`` -- the registered schema. This is what our
      shims are written against and the default.
    * ``npu_ops_transformer.ops.<name>`` -- the package's own interface: ``ops/__init__.py``
      imports each op's ``torch_extension`` module and binds that module-level function under the
      op name (vendor source: ``_discover_ops_from_dir`` + ``_load_op``). The binding uses the op
      *directory* name, so the delivered spelling and ours can differ (``npu_compressor`` in the
      schema, ``compressor`` as the function).

    ``_OT_PYTHON_INTERFACE`` names the ops where only the second works; everything else keeps the
    schema, so a package-side difference cannot quietly change what we call. An op still counts
    only when its entry actually provides a callable, otherwise it resolves to None and the
    caller degrades explicitly instead of reaching a stand-in by accident.
    """
    package = _ot_package_for(name)
    ops_mod = ot_ops(package)
    if ops_mod is None:
        return None

    registered = _OT_OP_NAMES.get(name)
    if registered is None:
        return None

    if name in _OT_PYTHON_INTERFACE:
        # The registered schema for these is not callable as documented (see the table above),
        # so the package's own python entry is the only entry -- never fall back to the schema.
        for attr in (registered, name):
            candidate = getattr(ops_mod, attr, None)
            if callable(candidate):
                return candidate
        return None

    ns = getattr(torch.ops, package, None)
    if ns is None:
        return None
    # 9.3.0 registers the un-prefixed spelling (``npu_compressor`` -> ``compressor``).
    for candidate in (registered, registered[len("npu_") :] if registered.startswith("npu_") else registered):
        try:
            return getattr(ns, candidate)
        except (AttributeError, RuntimeError):
            continue
    return None


def ot_op(name: str) -> Callable[..., Any] | None:
    """``npu_ops_transformer`` op ``name``, or None when the package does not provide it."""
    return _ot_op_in(name)


def _ot_impl(name: str) -> Callable[..., Any]:
    """The delivered implementation a shim body calls."""
    fn = ot_op(name)
    if fn is None:
        raise RuntimeError(
            f"the delivered op for '{name}' is unavailable -- no installed delivered package "
            f"provides it (default package '{_OT_PACKAGE}'; alternative packages only for "
            "_OT_PACKAGE_PREFERRED entries, spellings in _OT_OP_NAMES)."
        )
    return fn


_NAMESPACE_ORDER = ("cann_ops_transformer", "npu_ops_transformer")


def delivered_namespace():
    """``torch.ops`` namespace of the installed delivered package.

    CANN 9.3 ships it as ``cann_ops_transformer`` (ops registered without the ``npu_``
    prefix); 9.2 ships ``npu_ops_transformer`` (``npu_compressor``,
    ``npu_mixed_quant_sparse_flash_mla``, ...). Callers that use the namespace directly
    (the compressed-cache writer, the sparse-attention pair) go through this.

    "Installed" is decided by the same memoized probe ``ot_ops`` uses, so this function and the
    per-op resolver cannot disagree about whether a package exists. They can still prefer
    different packages when *both* are importable: the resolver defaults to ``_OT_PACKAGE`` and
    only opts into ``cann_ops_transformer`` for the ``_OT_PACKAGE_PREFERRED`` entries, while this
    one takes the first entry of ``_NAMESPACE_ORDER``. That divergence is unreachable on the two
    supported configurations (9.2-only, 9.3-only), so it warns rather than staying silent -- a box
    with both would otherwise call the schema and the namespace spellings of one op family out of
    different libraries.
    """
    installed = [package for package in _NAMESPACE_ORDER if ot_ops(package) is not None]
    if len(installed) > 1:
        logger.warning_once(
            "both delivered op packages are importable (%s); the namespace path uses "
            "'%s' while per-op resolution defaults to '%s'",
            ", ".join(installed),
            installed[0],
            _OT_PACKAGE,
        )
    for package in installed:
        namespace = getattr(torch.ops, package, None)
        if namespace is not None:
            return namespace
    raise RuntimeError(
        "no delivered op package is installed: neither 'cann_ops_transformer' (CANN 9.3) nor "
        "'npu_ops_transformer' (9.2) could be imported"
    )


def _sync_first_call(op_key: str) -> None:
    """Synchronize once per op so asynchronously-reported aclnn failures surface at the call.

    Skipped -- and deliberately NOT marked as done -- while a stream is being captured: an aclnn
    op may not synchronize one (runtime 107027 "Not allow to synchronize captured-stream", the very
    failure ``dsv4_compressor_metadata`` was rewritten to avoid). The
    first call outside a capture then pays the sync, so attribution survives either ordering.
    """
    if op_key in _SYNCED_FIRST_CALL or _stream_is_capturing():
        return
    _SYNCED_FIRST_CALL.add(op_key)
    sync = getattr(torch_npu.npu, "synchronize", None)
    if sync is not None:
        sync()


def _stream_is_capturing() -> bool:
    """Whether the current NPU stream is being captured (ACLGraph).

    ``torch.compiler.is_compiling()`` cannot answer this: it tracks Dynamo / ``torch.compile``
    only and returns False throughout ``torch.npu.graph`` capture. The PTA API
    is the one the fork already uses in ``worker/v2/aclgraph_utils.py``. Probed per call rather
    than bound at import: it is consulted once per op per process, and a torch_npu build without
    ``torch_npu`` may not carry it at all.
    """
    probe = getattr(torch_npu.npu, "is_current_stream_capturing", None)
    if probe is None:
        return False
    try:
        return bool(probe())
    except Exception:  # noqa: BLE001 - never let this probe break an op call
        return False


def rms_norm_gamma(weight: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Gamma in a dtype ``aclnnRmsNorm`` accepts beside ``x`` (see the dev fork's utils)."""
    if weight.dtype == torch.float32 or weight.dtype == x.dtype:
        return weight
    return weight.to(x.dtype)


@dataclass(frozen=True)
class RopeTables:
    """Partial-RoPE inputs of the compressor epilogue.

    ``head_dim`` is the rotated width (0 = no rotation) and ``sin``/``cos`` are the flattened
    ``[rows, head_dim]`` tables the fork hands over -- the same bytes the metadata op produces
    in ``[rows, 1, 1, head_dim]`` form.
    """

    sin: torch.Tensor | None = None
    cos: torch.Tensor | None = None
    head_dim: int = 0


def _compressor_norm_and_rope(
    compressed: torch.Tensor,
    norm_weight: torch.Tensor,
    rope: RopeTables,
    *,
    norm_eps: float,
) -> torch.Tensor:
    """Finish the compressor the delivered op stops short of: RMSNorm, then partial RoPE.

    The delivered ``npu_compressor`` documents five steps and names its output ``cmpKvOut =
    C_i^Comp``: matmul -> group +ape -> softmax -> Hadamard -> sum over the compression axis.
    This repo's in-tree compressor carries on with two more
    (``csrc/attention/compressor/README.md`` steps 6-7): ``RmsNorm(C^Comp) = norm_weight *
    C^Comp / RMS(C^Comp)``, then RoPE on the last ``rope_head_dim`` columns only
    (``op_kernel/arch35/compressor_block_vec.h``: "rope: 只对后RD进行rope"). The delivered
    arch35 kernel contains no norm and no rope at all -- its ``vf/`` directory has only
    ``vf_add`` / ``vf_mul`` / ``vf_softmax``, unlike the arch22 variant which still carries an
    ``RmsNormParam`` -- so for that entry the two steps belong to the caller.

    ``norm_weight`` is ``[D]`` and ``C^Comp`` is ``[..., D]``, so the norm is over the last dim,
    with eps inside the sqrt exactly as ``vf/vf_rms_norm.h`` computes it (sum of squares ->
    ``* 1/N`` -> ``+ eps`` -> ``sqrt`` -> divide -> multiply by gamma).
    """
    if compressed is None or compressed.numel() == 0 or norm_weight is None:
        return compressed

    # The delivered compressor produces C^Comp in fp32 while the checkpoint's gamma is bf16, and
    # aclnnRmsNorm has no (fp32 x, bf16 gamma) entry, so the gamma follows the rule in
    # ``rms_norm_gamma`` (shared with the Ascend RMSNorm):
    # Ascend RMSNorm: an fp32 gamma is the widest accepted form beside a 16-bit activation and
    # is kept, anything else follows the activation.
    norm_weight = rms_norm_gamma(norm_weight, compressed)
    normalized, _ = torch_npu.npu_rms_norm(compressed, norm_weight, norm_eps)
    if rope.head_dim > 0 and rope.cos is not None and rope.sin is not None:
        head_dim = normalized.shape[-1]
        # The delivered rope op requires x, cos and sin to be 4-D *together* and rotates the
        # ``partial_slice`` columns of the last dim; anything else is rejected before the
        # kernel runs (``EZ0013: ... x, cos and sin have incorrect shape dims``, see
        # inplace_rope_regbase_tiling_base.cpp:254). x is already [rows, D] here, and the fork
        # hands ``compressor`` flattened [rows, rope_head_dim] tables because the in-tree and
        # legacy entries take them that way (models/deepseek_v4/compressor.py) -- the bytes
        # are the same as the metadata op's [rows, 1, 1, dim] layout, so the unit axes are
        # restored here. ``unsqueeze`` (never ``reshape``) keeps x sharing this storage: the
        # op mutates its input in place, and that view is what carries the rotation back.
        x4 = normalized
        while x4.dim() < 4:
            x4 = x4.unsqueeze(-2)
        rows = x4.shape[0]
        inplace_partial_rotary_mul(
            x4,
            rope.cos.reshape(rows, 1, 1, rope.head_dim),
            rope.sin.reshape(rows, 1, 1, rope.head_dim),
            rotary_mode="interleave",
            partial_slice=[head_dim - rope.head_dim, head_dim],
        )
    return normalized


def compressor(
    x,
    wkv,
    wgate,
    state_cache,
    ape,
    *,
    norm_weight=None,
    rope_sin=None,
    rope_cos=None,
    state_block_table=None,
    cu_seqlens=None,
    seqused=None,
    start_pos=None,
    rope_head_dim: int = 0,
    cmp_ratio: int = 4,
    coff: int = 1,
    norm_eps: float = 1e-6,
    rotary_mode: int = 0,  # noqa: ARG001 - in-tree only
    cache_mode: int = 1,
):
    """``_C_ascend.compressor`` (18-arg) -> the delivered compressor.

    CANN sig: ``compressor(x, wkv, wgate, state_cache, ape, cmp_ratio=4, *,
    state_block_table, cu_seqlens, seqused, start_pos, coff=1, cache_mode=1)``
    returning only ``cmp_kv``. It is resolved through the package's own interface
    (``npu_ops_transformer.ops.compressor``) because the registered ``torch.ops`` schema for
    this op cannot be called as documented -- see ``_OT_PYTHON_INTERFACE``.

    ``norm_weight`` / ``rope_sin`` / ``rope_cos`` / ``rope_head_dim`` / ``norm_eps`` /
    ``rotary_mode`` are keyword-only here (the five required tensors of the delivered op stay
    positional): the CANN op has **no** norm and no RoPE input, while this repo's in-tree
    compressor folds both in. Both missing steps are applied here, after the call
    (``_compressor_norm_and_rope``), and the result is what the caller expects.
    """
    # A6 runs the delivered 14-argument op: CANN 9.2 ships its bf16/fp16 kernel under
    # ``opp/built-in/.../ascend960dt/ops_transformer/compressor`` (attrs: cmp_ratio,
    # coff, cache_mode, state_cache_stride_dim0), and the package's own interface
    # ``npu_ops_transformer.ops.compressor.npu_compressor`` is the spelling that reaches
    # it. The in-tree ``_C_ascend.compressor`` folds RMSNorm and RoPE into the op (19
    # arguments) and needs a vendor bundle that would shadow the built-in A6 kernel, so it
    # must not be used here.
    fn = _ot_impl(_COMPRESSOR_OP_KEY)
    out = fn(
        x,
        wkv,
        wgate,
        state_cache,
        ape,
        cmp_ratio,
        state_block_table=state_block_table,
        cu_seqlens=cu_seqlens,
        seqused=seqused,
        start_pos=start_pos,
        coff=coff,
        cache_mode=cache_mode,
    )
    # One synchronize after the first call, like the two cache-writing ops: aclnn failures can
    # be reported only at the next sync, so without this the worker dies with the engine's error
    # long after the call that caused it.
    _sync_first_call(_COMPRESSOR_OP_KEY)
    # ``_compressor_norm_and_rope`` rotates with rotary_mode="interleave" hardcoded, which is
    # the only layout this checkpoint uses. Reject the other spelling instead of silently
    # applying interleave to a caller that asked for half (in-tree rotary_mode=1).
    if rotary_mode not in (0, 2):
        raise RuntimeError(
            f"compressor: rotary_mode={rotary_mode} is not supported by the delivered path "
            "(0 or 2 / interleave only); the in-tree op also accepts 1 (half)."
        )
    out = _compressor_norm_and_rope(
        out,
        norm_weight,
        RopeTables(sin=rope_sin, cos=rope_cos, head_dim=rope_head_dim),
        norm_eps=norm_eps,
    )
    # The delivered compressor yields C^Comp in fp32 and the norm/rope post-processing keeps it
    # there, while the in-tree contract lists ``cmp_kv`` as FLOAT16/BFLOAT16
    # (csrc/attention/compressor/README.md) and the result goes straight into the compressed
    # cache write. Hand back the caller's dtype instead of a 4-byte-per-element tensor: the
    # cache write interprets those bytes at the cache dtype, so a wider payload is not a
    # slower answer, it is the wrong one.
    return out.to(x.dtype)


def inplace_partial_rotary_mul(x, r1, r2, *, rotary_mode="interleave", partial_slice=None):
    """Partial interleaved RoPE, mutating ``x`` in place.

    The in-tree ``_C_ascend`` op is preferred whenever this build registered it.
    Its aclnn entry carries the third attribute (``negate_sin``); the delivered
    package's entry passes only ``rotary_mode`` and ``partial_slice``, and on A6
    that shorter call lands in this repo's own tiling, which dereferences
    ``attrs->GetAttrPointer<bool>(2)`` without a null check -- so the 2-attribute
    call segfaults inside ``aclnnInplacePartialRotaryMulGetWorkspaceSize`` right
    after the shape checks pass. Callers already pass the negated sin where the
    sign must flip, so ``negate_sin=False`` reproduces their intent.
    """
    try:
        in_tree = torch.ops._C_ascend.inplace_partial_rotary_mul
    except AttributeError:  # build without the in-tree op registered
        in_tree = None
    if in_tree is not None:
        return in_tree(
            x,
            r1,
            r2,
            rotary_mode,
            list(partial_slice or [0, 0]),
            False,
        )
    fn = _ot_impl("inplace_partial_rotary_mul")
    return fn(x, r1, r2, rotary_mode=rotary_mode, partial_slice=partial_slice)


def _indexer_scale_as(scale: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    """Reshape an indexer dequant scale to the shape the delivered op checks.

    The V2 op compares ``q_descale`` against the query (``[T, N]`` for a TND
    query) and ``k_descale`` against the first three dims of the PA_BBND key
    (``[blocks, block_size, N]``), while the pipeline keeps both in the flat
    cache spelling (one trailing 1 per row). Same elements, different rank, so
    reshape when the counts agree; otherwise hand over the original and let the
    operator's own shape check report what it wanted.
    """
    want = 1
    for dim in shape:
        want *= dim
    if tuple(scale.shape) == tuple(shape) or scale.numel() != want:
        return scale
    return scale.reshape(*shape)


def quant_lightning_indexer_metadata(
    *,
    num_heads_q,
    num_heads_k,
    head_dim,
    topk,
    quant_mode,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    batch_size=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
    layout_q=None,
    layout_k=None,
    mask_mode=None,
    cmp_ratio=None,
    **_ignored,
):
    """Tiling metadata for the quantized indexer, on the delivered V2 metadata op.

    The DSv4 call site still carries the fork's ``actual_seq_lengths_query`` /
    ``actual_seq_lengths_key`` spellings and a ``device`` argument; the delivered
    schema has neither (``cu_seqlens_q``/``cu_seqlens_k`` carry the same
    information and the output device follows the optional tensors), so they are
    swallowed here instead of at the call site.
    """
    fn = _ot_impl("quant_lightning_indexer_metadata")
    return fn(
        num_heads_q=num_heads_q,
        num_heads_k=num_heads_k,
        head_dim=head_dim,
        topk=topk,
        quant_mode=quant_mode,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        cmp_residual_k=cmp_residual_k,
        batch_size=batch_size,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        layout_q=layout_q,
        layout_k=layout_k,
        mask_mode=mask_mode,
        cmp_ratio=cmp_ratio,
    )


def quant_lightning_indexer(
    *,
    query,
    key,
    weights,
    query_dequant_scale,
    key_dequant_scale,
    topk,
    quant_mode,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    block_table=None,
    output_idx_offset=None,
    metadata=None,
    max_seqlen_q=-1,
    layout_q="BSND",
    layout_k="BSND",
    mask_mode=0,
    cmp_ratio=1,
    return_value=0,
    **_ignored,
):
    """Quantized indexer top-k, on the delivered V2 op (fp8 q/k + fp32 scales on A6)."""
    fn = _ot_impl("quant_lightning_indexer")
    return fn(
        query=query,
        key=key,
        weights=weights,
        query_dequant_scale=_indexer_scale_as(query_dequant_scale, tuple(query.shape[:2])),
        key_dequant_scale=_indexer_scale_as(key_dequant_scale, tuple(key.shape[:3])),
        topk=topk,
        quant_mode=quant_mode,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        cmp_residual_k=cmp_residual_k,
        block_table=block_table,
        output_idx_offset=output_idx_offset,
        metadata=metadata,
        max_seqlen_q=max_seqlen_q,
        layout_q=layout_q,
        layout_k=layout_k,
        mask_mode=mask_mode,
        cmp_ratio=cmp_ratio,
        return_value=return_value,
    )


def mhc_pre_sinkhorn(
    x,
    hc_fn,
    hc_scale,
    hc_base,
    *,
    hc_mult: int,
    sinkhorn_iters: int,
    norm_eps: float,
    hc_eps: float,
):
    """``deepseek_v4.hc_pre_ref`` -> ``torch.ops.npu_ops_transformer.mhc_pre_sinkhorn``.

    The delivery's fused mHC pre + sinkhorn, validated on the real box during bring-up, so it is
    *the* A6 implementation of mHC-pre -- there is no fallback: this shim resolves it
    directly, so a package without it raises at resolution.

    Contract differences against this call site, all handled here:
      * the op wants 4-D ``[bs, seq_len, n, c]`` while the model passes 3-D ``[T, n, c]``
        -- one unit ``seq_len`` axis is added and stripped again (each token is its own
        length-1 sequence, which is what the fork's per-token mHC means);
      * its epsilon order is ``hc_eps, norm_eps``, the caller's is ``norm_eps, hc_eps``;
      * ``h_res`` comes back flattened ``[..., n*n]`` and is reshaped to ``[..., n, n]``
        to match ``hc_pre_ref``'s ``comb``.

    What this op returns is now settled by the vendor's own golden, not by its name
    (``ops-transformer mhc/mhc_pre_sinkhorn/tests/assets/golden.py``, the reference the op is
    validated against):

      * ``h_res`` is the **post-sinkhorn** combination matrix: the golden's ``_compute`` runs
        ``softmax`` then a column pass and ``num_iters-1`` (row, column) passes, and writes that
        final ``current`` into ``h_res``. So ``hc_post`` gets the doubly-stochastic comb it
        expects and the shim must NOT add a second sinkhorn. (The L0 header's wording,
        "Combination matrix before Sinkhorn", means the residuals' matrix, not an un-normalised
        one -- CANN's separate ``aclnnMhcPre(V2)`` really does ship an un-sinkhorned ``hRes``,
        which is what C3 used to warn about.)
      * output order is ``hin, hPost, hRes, hPre, hcBeforeNorm, invRms, sumOut, normOut``, and
        with ``need_backward=False`` (our ``outFlag=False``) the golden empties the last five --
        so the first three are the real ones and the length-8 return is expected.
      * ``hin = sum(x * h_pre)`` and ``h_post = 2*sigmoid(...)`` match ``hc_pre_ref`` exactly;
        ``num_iters`` comes from the model config (``config.hc_sinkhorn_iters``) on both paths.

    ``hc_mult`` / ``sinkhorn_iters`` / ``norm_eps`` / ``hc_eps`` are keyword-only: the caller
    names them, which keeps the shim's arity down and makes the epsilon order swap above
    explicit at the call site.
    """
    fn = _ot_impl("mhc_pre_sinkhorn")
    added_axis = x.dim() == 3
    if added_axis:
        x = x.unsqueeze(1)  # [T, n, c] -> [T, 1, n, c]
    # ``outFlag`` is the ninth parameter the schema declares (omitting it raises "missing value
    # for argument 'outFlag'"). False asks for the main outputs only: the last five come back
    # empty, leaving exactly the three this shim uses.
    out = fn(x, hc_fn, hc_scale, hc_base, hc_mult, sinkhorn_iters, hc_eps, norm_eps, outFlag=False)
    # Eight values come back; the first three are the real ones. Fewer than three means a
    # contract change on the delivery side, which must not pass silently.
    if len(out) < 3:
        raise RuntimeError(f"mhc_pre_sinkhorn returned {len(out)} values; the first three are used")
    hin, h_post, h_res = out[0], out[1], out[2]
    if added_axis:
        hin = hin.squeeze(1)  # [T, 1, c] -> [T, c]
        h_post = h_post.squeeze(1)  # [T, 1, n] -> [T, n]
        h_res = h_res.squeeze(1)  # [T, 1, n*n] -> [T, n*n]
    comb = h_res.reshape(*h_res.shape[:-1], hc_mult, hc_mult)
    return hin, h_post, comb


def mhc_post_npu(x, residual, post, comb):
    """``deepseek_v4.hc_post_ref`` -> ``torch_npu.npu_mhc_post(x, h_res, h_out, h_post)``.

    Same arg semantics as the CANN op: x=residual streams, h_out=sub-layer output.
    """
    return torch_npu.npu_mhc_post(residual, comb, x, post)


def dsv4_compressor_metadata(
    full_compress_cos: torch.Tensor,
    full_compress_sin: torch.Tensor,
    query_start_loc: torch.Tensor,
    start_pos: torch.Tensor,
    block_table: torch.Tensor,
    *,
    kv_block_size: int,
    slot_mapping_format: int,
    compress_ratio: int,
    num_compressed_tokens: int,
    num_reqs_actual: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The compressed plan's row metadata, in tensor ops, for builds without the in-tree op.

    This build's ``_C_ascend`` has no DSv4 compressor-metadata op, yet the compressed cache write needs to
    know WHERE each compressed row goes: the slot mapping this returns is what
    ``kv_compress_epilog`` writes the main compressed cache at and what
    ``indexer_quant_cache`` writes the indexer caches at. It therefore has to be real --
    an ``arange`` placeholder puts rows in the wrong pages, invisibly where values are
    (all values are 0) and silently wrong everywhere else.

    This is a faithful port of the in-tree op, not a reinvention. Spec:
    ``csrc/attention/compressor_metadata/op_kernel/compressor_metadata.h`` plus the
    vendor's own python reference and hand-written expectations in
    ``tests/e2e/nightly/single_node/ops/singlecard_ops/test_compressor_metadata.py``.
    In the kernel's terms:

    * a request contributes ``(start + seq) // cmp_ratio - start // cmp_ratio`` rows when
      ``start_pos >= 0`` and ``seq > 0``; ``prefix`` is their exclusive scan;
    * row ``r`` belongs to the request whose prefix interval contains it, and its
      compressed position is ``start // cmp_ratio + (r - prefix[req])``;
    * that position addresses block table entry ``pos // kv_block_size`` at offset
      ``pos % kv_block_size``; FLAT stores ``block_id * kv_block_size + offset`` while
      BLOCK_OFFSET stores the pair ``(block_id, offset)``;
    * the row carries the RoPE row of the ORIGINAL group start, ``rope[pos * cmp_ratio]``
      (every ``cmp_ratio``-th row of the full table);
    * invalid rows -- past ``prefix[-1]``/``num_compressed_tokens``, block table entry out
      of range or negative, or no RoPE row -- get ``cos=1, sin=0, slot=-1`` (BLOCK_OFFSET
      also writes ``kv_block_size - 1`` in column 1, as the kernel's WriteInvalidTile does).

    Output shapes match the real op's meta (``torch_binding_meta.cpp``): ``cos``/``sin`` are
    ``[rows, 1, 1, rope_dim]`` with the rope dtype, ``slot`` is int32 and either ``[rows]``
    or ``[rows, 2]``.

    Everything is computed with tensor ops on the device the inputs live on -- no host
    round trip and no boolean-mask indexing (see the note in the body: on A6 that becomes
    aclnnNonzeroV2, which cannot run inside a captured stream), so this can run per layer per
    forward inside an ACLGraph without a sync.

    Known difference from the kernel: the RoPE bound is checked per row here, while the
    kernel checks it once per tile (tiles are additionally cut at block boundaries). They
    can only differ for the rows between the last existing RoPE row and the end of that
    tile, i.e. at the tail of a maximum-length sequence; this port keeps any row whose own
    RoPE row exists.

    The scalar arguments are keyword-only and carry the in-tree schema's own names
    (``kv_block_size``, ``slot_mapping_format``, ``compress_ratio``, ``num_compressed_tokens``,
    ``num_reqs_actual``), so a call site that reaches either implementation names them the same
    way.
    """
    rows = int(num_compressed_tokens)
    reqs = int(num_reqs_actual)
    ratio = int(compress_ratio)
    block_size = int(kv_block_size)
    device = full_compress_cos.device

    # ``flatten(1)`` and not ``reshape(shape[0], -1)``: the ``-1`` is ambiguous for an empty RoPE
    # table, which would raise HERE instead of reaching the ``rope_rows == 0`` guard below.
    rope_cos = full_compress_cos.flatten(1)
    rope_sin = full_compress_sin.flatten(1)
    rope_rows, rope_dim = rope_cos.shape

    out_cos = torch.ones((rows, 1, 1, rope_dim), dtype=rope_cos.dtype, device=device)
    out_sin = torch.zeros((rows, 1, 1, rope_dim), dtype=rope_sin.dtype, device=device)
    from vllm_ascend.attention.dsa_attn_kv_plan import DSA_COMPRESSOR_SLOT_MAPPING_FLAT

    flat = int(slot_mapping_format) == DSA_COMPRESSOR_SLOT_MAPPING_FLAT
    if flat:
        slot = torch.full((rows,), -1, dtype=torch.int32, device=device)
    else:
        slot = torch.full((rows, 2), -1, dtype=torch.int32, device=device)
        slot[:, 1] = block_size - 1
    if rows <= 0 or reqs <= 0:
        return out_cos, out_sin, slot

    start = start_pos[:reqs].to(torch.int64)
    cu_seqlens = query_start_loc[: reqs + 1].to(torch.int64)
    seq_len = cu_seqlens[1:] - cu_seqlens[:reqs]
    counted = (start >= 0) & (seq_len > 0)
    per_req = torch.where(counted, (start + seq_len) // ratio - start // ratio, torch.zeros_like(seq_len))
    prefix = torch.zeros(reqs + 1, dtype=torch.int64, device=device)
    prefix[1:] = torch.cumsum(per_req, dim=0)

    row = torch.arange(rows, dtype=torch.int64, device=device)
    # The kernel's FindRequest counts the requests whose prefix end is <= row; rows past the
    # last prefix end are invalid and are masked out below, hence the clamp.
    req = torch.searchsorted(prefix[1:], row, right=True).clamp_(max=reqs - 1)
    compressed_pos = start[req] // ratio + (row - prefix[req])
    block_id_offset = torch.div(compressed_pos, block_size, rounding_mode="floor")
    slot_offset = compressed_pos - block_id_offset * block_size
    rope_pos = compressed_pos * ratio

    stride = int(block_table.shape[1])
    in_table = block_id_offset < stride
    block_id = block_table[req, block_id_offset.clamp(max=stride - 1)].to(torch.int64)
    keep = (row < prefix[reqs]) & in_table & (block_id >= 0) & (rope_pos < rope_rows)

    if rope_rows == 0:
        return out_cos, out_sin, slot

    # No boolean-mask indexing here on purpose: `row[keep]` compiles to aclnnNonzeroV2, which
    # synchronizes the host to learn how many elements were selected -- illegal inside a
    # captured stream (ACLGraph: "Not allow to synchronize captured-stream", runtime 107027,
    # opType NonZero). index_select / torch.where give the same values with no
    # host round trip, so this can run per layer per forward inside an ACLGraph.
    safe_rope = rope_pos.clamp(min=0, max=rope_rows - 1)
    gathered_cos = rope_cos.index_select(0, safe_rope)
    gathered_sin = rope_sin.index_select(0, safe_rope)
    keep_2d = keep.unsqueeze(-1)
    out_cos[:, 0, 0] = torch.where(keep_2d, gathered_cos, torch.ones_like(gathered_cos))
    out_sin[:, 0, 0] = torch.where(keep_2d, gathered_sin, torch.zeros_like(gathered_sin))
    if flat:
        flat_slot = (block_id * block_size + slot_offset).to(torch.int32)
        slot = torch.where(keep, flat_slot, torch.full_like(flat_slot, -1))
    else:
        slot[:, 0] = torch.where(keep, block_id.to(torch.int32), torch.full_like(block_id, -1, dtype=torch.int32))
        slot[:, 1] = torch.where(
            keep, slot_offset.to(torch.int32), torch.full_like(slot_offset, block_size - 1, dtype=torch.int32)
        )
    return out_cos, out_sin, slot


_NPU_GATING_HASH_OK: bool | None = None


_GATING_HASH_ARG_NAMES = ("input_ids", "tid2eid")


def torch_npu_op(name: str) -> Callable[..., Any] | None:
    """Return ``torch_npu.<name>``, or None when this build does not bind it.

    PTA is the long-term home for the aclnn ops the delivery package does not ship.
    """
    return getattr(torch_npu, name, None)


def _npu_gating_schema_args() -> set[str] | None:
    """Argument names of ``npu::npu_moe_gating_top_k`` from the registered torch schema.

    The schema is the authoritative source: ``torch_npu``'s python wrapper is an
    ``OpCommand``-style ``(*args, **kwargs)`` function, so its signature says nothing about the
    op (that trap cost us a serve run -- see ``_npu_gating_accepts_hash``).
    """
    try:
        packet = getattr(getattr(torch.ops, "npu", None), "npu_moe_gating_top_k", None)
        schemas = getattr(packet, "_schemas", None)
        if schemas:
            names: set[str] = set()
            for schema in schemas.values():
                names |= {arg.name for arg in schema.arguments}
            return names
        # The schema dump lives behind torch's private ``_C``; reached through getattr because
        # there is no public API that lists every registered schema.
        all_schemas = getattr(getattr(torch, "_C", None), "_jit_get_all_schemas", None)
        if all_schemas is not None:
            for schema in all_schemas():
                if str(schema.name) == "npu::npu_moe_gating_top_k":
                    return {arg.name for arg in schema.arguments}
    except Exception:  # noqa: BLE001 - schema introspection is best-effort
        return None
    return None


def _npu_gating_accepts_hash(fn: Callable[..., Any]) -> bool:
    """Whether ``torch_npu.npu_moe_gating_top_k`` accepts ``input_ids`` / ``tid2eid``.

    Three sources, because the obvious one is a trap: the wrapper is an ``OpCommand``-style
    function declared ``(*args, **kwargs)``, so ``inspect.signature`` reports exactly those two
    names -- reading it that way made A6 refuse a build whose registered schema *does* carry
    ``input_ids`` / ``tid2eid`` (measured on the box, 2026-09-22:

    ``npu::npu_moe_gating_top_k(Tensor x, int k, *, Tensor? bias=None, Tensor? input_ids=None,
    Tensor? tid2eid=None, ...)``). Order:

      1. the registered torch schema (authoritative);
      2. ``inspect.signature``, ignoring the bare ``*args`` / ``**kwargs`` names;
      3. the wrapper's own docstring, which torch_npu generates from the schema and which shows
         the prototype ``npu_moe_gating_top_k(x, k, bias=None, input_ids=None, tid2eid=None, ...)``.

    If none of them is conclusive, assume yes: passing the arguments then fails loudly inside the
    op instead of silently dropping the hash re-placement. Probed once -- none of these inputs
    can change within a process.
    """
    global _NPU_GATING_HASH_OK
    if _NPU_GATING_HASH_OK is None:
        names = _npu_gating_schema_args()
        if names is None:
            try:
                params = set(inspect.signature(fn).parameters)
            except (TypeError, ValueError):
                params = set()
            names = params - {"args", "kwargs"} or None
        if names is None:
            doc = getattr(fn, "__doc__", "") or ""
            if all(argument in doc for argument in _GATING_HASH_ARG_NAMES):
                names = set(_GATING_HASH_ARG_NAMES)
        _NPU_GATING_HASH_OK = True if names is None else any(argument in names for argument in _GATING_HASH_ARG_NAMES)
    return _NPU_GATING_HASH_OK


def _op_schema_has_term(op: Callable[..., Any], term: str) -> bool:
    """Whether the registered schema of ``op`` mentions ``term``.

    The shims that forward a keyword the delivered op does not declare get a TypeError at
    call time; probing the schema first turns that into a message that names the missing
    argument. ``_schemas`` is the only reliable view -- the wrappers here are declared
    ``(*args, **kwargs)``, so ``inspect.signature`` reports nothing useful (same trap as
    ``_npu_gating_accepts_hash`` documents).
    """
    schemas = getattr(op, "_schemas", None)
    if not schemas:
        return False
    return any(term in str(schema) for schema in schemas.values())


def moe_gating_top_k_hash_npu(
    x: torch.Tensor,
    *,
    k: int,
    bias: torch.Tensor | None = None,
    input_ids: torch.Tensor | None = None,
    tid2eid: torch.Tensor | None = None,
    k_group: int = 1,
    group_count: int = 1,
    routed_scaling_factor: float = 1.0,
    eps: float = 1e-20,
    group_select_mode: int = 1,
    renorm: int = 0,
    norm_type: int = 2,
    out_flag: bool = False,
    bias_vl: torch.Tensor | None = None,
    image_sentinel_lo: int = 0,
    image_sentinel_count: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``torch.ops._C_ascend.moe_gating_top_k_hash`` -> ``torch_npu.npu_moe_gating_top_k``
    carrying the hash-placement arguments (``input_ids`` / ``tid2eid``).

    Same A6 kernel as the plain gating op (``A5DeviceAdaptor.moe_gating_top_k``);
    the hash variant only adds the token-id -> expert re-placement.

    ``x`` and ``k`` are passed positionally to match torch_npu's documented prototype
    (``npu_moe_gating_top_k(x, k, bias=None, input_ids=None, tid2eid=None, ...)``); everything
    after them is keyword-only in the registered schema.

    A build that really does not take those arguments is refused loudly instead of falling back
    to a pure-torch top-k: that stand-in IGNORES the hash placement, i.e. it routes tokens to
    different experts than the fused op -- a silently wrong result is worse than a failed start.
    (The refusal must be based on the registered schema, not on ``inspect.signature`` of this
    ``OpCommand`` wrapper -- see :func:`_npu_gating_accepts_hash`.)
    """
    # The in-tree op also carries the vision preset (``bias_vl``); torch_npu's wrapper has no
    # such argument, so the shim can only honour it by delegating to the in-tree op.
    if bias_vl is not None:
        # The delivered gating op has no ``bias_vl`` argument. Warning and continuing routed
        # image tokens without the vision bias: wrong experts, no error, and the caller could
        # not tell. Take the in-tree op when this build registered it (that one does accept
        # ``bias_vl``), otherwise refuse -- never substitute a top-k that ignores it.
        # A6 keeps MOE_GATING_TOP_K_HASH_VISION (its in-tree op declares ``bias_vl``), so A6
        # with a vision checkpoint reaches this branch too -- the in-tree delegation below is
        # the primary path, not an A5-only fallback.
        ascend_ns = getattr(torch.ops, "_C_ascend", None)
        in_tree_hash = getattr(ascend_ns, "moe_gating_top_k_hash", None) if ascend_ns is not None else None
        # The in-tree op declares ``bias_vl`` (csrc autogen proto has it as OPTIONAL_INPUT),
        # but the registered schema is what a call actually has to satisfy. Probe it before
        # delegating: handing a keyword the schema does not declare raises TypeError, which is
        # the same "silent wrong answer became an opaque crash" trade this branch exists to
        # avoid -- with a message that says which argument is missing instead.
        if in_tree_hash is not None and not _op_schema_has_term(in_tree_hash, "bias_vl"):
            in_tree_hash = None
        if in_tree_hash is None:
            raise RuntimeError(
                "bias_vl requires the in-tree moe_gating_top_k_hash, which this build does not "
                "register: the delivered gating op has no vision bias input, and routing image "
                "tokens without it silently changes the expert selection. Withhold "
                "MOE_GATING_TOP_K_HASH_VISION so the router uses its vision-aware fallback."
            )
        topk_weights, topk_ids, vision_out = in_tree_hash(
            x,
            k,
            bias=bias,
            input_ids=input_ids,
            tid2eid=tid2eid,
            k_group=k_group,
            group_count=group_count,
            routed_scaling_factor=routed_scaling_factor,
            eps=eps,
            group_select_mode=group_select_mode,
            renorm=renorm,
            norm_type=norm_type,
            out_flag=out_flag,
            bias_vl=bias_vl,
            image_sentinel_lo=image_sentinel_lo,
            image_sentinel_count=image_sentinel_count,
        )
        return topk_weights, topk_ids.to(torch.int32), vision_out

    fn = torch_npu_op("npu_moe_gating_top_k")
    if fn is None:
        raise RuntimeError(
            "torch_npu.npu_moe_gating_top_k is not available in this process; the A6 hash gating op cannot be called."
        )
    if (input_ids is not None or tid2eid is not None) and not _npu_gating_accepts_hash(fn):
        raise RuntimeError(
            "torch_npu.npu_moe_gating_top_k does not expose the hash-placement arguments "
            "(input_ids / tid2eid) in this build: neither the registered schema, nor the "
            "wrapper docstring, nor its signature lists them. Refusing instead of substituting a "
            "top-k that ignores hash placement (it would route tokens to different experts). "
            "Upgrade the torch_npu build."
        )

    topk_weights, topk_ids, out = fn(
        x,
        k,
        bias=bias,
        input_ids=input_ids,
        tid2eid=tid2eid,
        k_group=k_group,
        group_count=group_count,
        group_select_mode=group_select_mode,
        renorm=renorm,
        norm_type=norm_type,
        routed_scaling_factor=routed_scaling_factor,
        eps=eps,
    )
    return topk_weights, topk_ids.to(torch.int32), out
