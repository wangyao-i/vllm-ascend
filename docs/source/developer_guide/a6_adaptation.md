# A6 operator and framework alignment

This note tracks the A6 bring-up differences for Qwen3.5-35B and DeepSeek-V4-Flash.
The starting point is `br_poc_a6_1106` at `28e0d9db728b6b1b7563d901e9d79494dc39870b`,
which includes the community `v0.32.0rc1` tag. Source inspection was performed on
2026-11-17. Runtime failures described by existing bring-up comments still need
reproduction on a pinned NPU stack.

## Integration boundary

Prefer the community model, cache, scheduling and distributed implementations.
Put A6 tiling, kernel selection, workspace and cross-core synchronization changes
in the operator implementation. A matching entry-point core ratio also requires
the controller to handle the corresponding vector work and synchronization.

The existing hardware profile and device adaptor provide the dispatch boundary.
The capabilities below describe the bring-up software stack. Each enabled path
still needs its model, shape, dtype and runtime preconditions.

## Changes in this alignment commit

| Area | Correction | Qualification still required |
| --- | --- | --- |
| Community DSA dispatch | Add the in-tree QLI compute and metadata methods to `BaseDeviceAdaptor`, inherited by A5 | Native A2/A3/A5 DSA coverage |
| A6 QLI dispatch | Require the delivered FP8 operator and its paired metadata interface | FP8 outputs, metadata ABI and graph replay |
| Compressor and mHC | Scope delivered implementations to A6 capabilities and restore the community calls for other profiles | Native regression coverage and A6 model parity |
| Qwen normalization and MRoPE | Enable the existing fused capabilities on A2/A3/A5, retain A6 exclusions | Fused-path regression coverage |
| GDN prefill | Give the native PTA probe its own capability, independent of external FLA availability | Native and FLA backends qualified separately |
| Diagnostics | Remove QLI tensor-to-host debug prints. Retain compressor first-call synchronization with its capture guard | Graph capture, replay and model performance |
| A5 build | Restore shared-KV attention and metadata entries accidentally disabled with A6 TODO comments | A5 custom-op build |
| K2q CSR | Add DT/PR registration to histogram, row-prefix, tile-prefix and scatter stages | Full five-stage pipeline and boundary cases |
| AddRmsNormBias | Select regbase tiling for DAV_9201/DAV_9202 to match the selected apt kernel | Target CANN build, tiling keys, dtype and output parity |
| GDN FwdH | Select the existing `Ascend960DT` architecture tag for 9201 in the kernel and two epilogue policies | Workspace, scheduling and cross-core flags |
| Operator definitions | Remove duplicate DT/PR registrations in FwdH, FwdO and AttnRes | Target CANN build |

The A6 AddRmsNormBias and monolithic PTA GDN prefill paths remain disabled during qualification.
The selected chunk pipeline combines Triton preparation with the in-tree AscendC
FwdH/FwdO operators, so FwdH architecture changes affect the active prefill path.
The FwdH tag correction addresses one source inconsistency. A complete port must
also establish controller and synchronization correctness. PR/9202 FwdH support
requires a separately supported architecture policy.

## Framework bypasses retained for A6

| Path | Reason recorded by the bring-up branch | Condition for restoring the community path |
| --- | --- | --- |
| QLI delivered package | In-tree A6 v2 dispatch reaches the INT8 arch22 implementation, while the A6 adaptor prepares FP8 tensors | Port FP8 compute and metadata together and preserve the community ABI |
| Compressor delivered kernel plus Python RMSNorm/RoPE | In-tree launch reportedly hangs at synchronization, and the A6 build excludes this operator. The delivered output ends before norm/RoPE | Implement the fused community output and cache contract, including dtype, rotary mode and continuous state updates |
| Delivered mHC pre | In-tree pre reportedly fails to return on the bring-up stack | Qualify in-tree pre output order, epsilon order, sinkhorn, shapes and precision |
| PTA mHC post | Existing A6 validation reportedly used this pairing with delivered pre. The branch also reports that in-tree post can return | Compare both post implementations with the same pre outputs, then restore independently if equivalent |
| GDN chunk prefill | Existing comments report monolithic PTA fused timeouts. The fallback uses Triton and AscendC FwdH/FwdO | Qualify chunk boundaries, final state, variable lengths, padding and repeated graph replay |
| Gemma RMSNorm and add-RMSNorm-bias decomposition | Gemma target binary availability is unresolved, and custom add-RMSNorm-bias had inconsistent tiling dispatch | Verify loaded symbols, target binaries and custom-op results on the pinned stack |
| Split RMSNorm/MRoPE | Existing comments report a vector-core exception in fused Triton MRoPE | Reproduce and fix the fused kernel with 1D/3D positions, supported dtypes and graph modes |
| TopKTopP fallback | Existing stack reportedly lacks the required target binary | Verify the exact PTA API, sampling semantics and target binary |
| FP4 NZ conversion exception | Existing comments report a packed-FP4 C0 mismatch | Establish the GMM weight ABI and validate tensor/list inputs, conversion and reload |
| ACL DVPP compatibility shim | Python attribute compatibility with the packaged runtime | Qualify the dependency version that removes the need for the shim |

The mHC pre wrapper retains three ABI adaptations. It adds a sequence dimension
to `[T,n,c]`, passes `(hc_eps, norm_eps)` to the delivered interface, and reshapes
the flattened residual matrix to `[T,n,n]`. The native path retains its original
`(norm_eps, hc_eps)` argument order.

The compressor wrapper retains its output dtype and cache-update behavior.
Moving its RMSNorm/RoPE work into an operator must preserve those contracts.
It also retains the existing first-call synchronization to attribute asynchronous
kernel failures. Capture defers that synchronization until the first call outside
capture. Changes to this diagnostic behavior require target-stack evidence.
Existing `A6_IN_TREE_COMPRESSOR` and `A6_IN_TREE_INDEXER` diagnostic switches
remain available for controlled comparison. Their centralization and lifecycle
need cleanup before an upstream submission.

## Outstanding framework issues

- The delivered resolver eagerly imports two package families and suppresses
  their import failures. Package selection, import diagnostics and prewarming
  should use a qualified dependency contract.
- Compressor loads a wildcard-matched extension from the torch extension cache.
  That path does not establish the Python, torch, CANN or SoC build identity.
  Replace it with the installed package interface after qualifying that interface
  on the target stack. The current path is retained to preserve the existing
  bring-up deployment.
- A6 inherits `BaseDeviceAdaptor`, whose fused GMM-SiTU operation is unsupported.
  Quantization modes that select it require an operator port and a matching
  adaptor implementation. Model coverage must include the selected quantization.
- QLI metadata contains a global 64-core assumption. A5 uses a different core
  budget, and metadata offsets depend on that budget. Update the producer and
  consumer together or restore the original ABI for each supported stack.

## Outstanding operator work

| Operator | Source gap | Next operator task |
| --- | --- | --- |
| GMM-SiTU | Host build excludes non-950 targets and the compute body is guarded by 3510. Internal control uses two AIV roles | Implement the A6 controller, work partition and synchronization |
| QLI v2 | Added FP8 keys lack a complete A6 FP8 execution branch | Port the arch92 compute path with its metadata ABI |
| Sparse and quantized sparse flash attention | Target registration coexists with older arch22 and 1:2 execution paths | Port target-specific tiling, execution and cross-core flags |
| QLI metadata | Fixed core counts affect allocation and offsets | Use runtime core budgets with a version-matched producer/consumer layout |
| GDN FwdH/FwdO | Architecture tags and entry ratio cover part of the migration | Audit head slots, workspace, vector scheduling and flag ownership |
| HcPre | Host and kernel dispatch still select legacy behavior for A6 | Port the complete arch92 pre/sinkhorn path behind the community ABI |
| Compressor | A6 build exclusion and legacy controller remain | Port the full fused contract, including local norm/RoPE semantics |
| FLA recurrent V310 | FP16 and sequence-length constraints differ from general Qwen BF16/MTP usage | Qualify each required dtype and sequence length with the installed package |

Pure-vector operators also need per-operator dispatch review. Moe top-k hash,
HcPost and rotary contain legacy host or architecture selectors. Their target
registration alone does not establish hash-mode inputs or tail correctness.

## FLA package dependency

`fla_npu` is imported by the GDN module for causal convolution and recurrent GDN.
Disabling fused prefill still requires these imports. Existing Docker/CI setup
installs the `flash-linear-attention-npu-a2`, `-a3` or `-a5` distribution at
`26.9.1`. The Python import name is `fla_npu`.

The inspected A6-dev README uses `FLA_NPU_SOC=ascend960dt`, and its example build
selects `chunk_fwd_h,chunk_gated_delta_rule_fwd_prepare`. That example does not
establish coverage of the convolution, recurrent and complete prefill paths
required by Qwen. Record the installed wheel, module path, kernel set and build
identity before choosing the backend. PR1134 is a separate candidate revision.

## Adaptation order by model and operator provider

Use the A5 model and cache contracts as the baseline. Select an implementation
in the device adaptor or existing cache plan. A6 CANN binaries, Python bindings
and custom kernels can be qualified independently behind those contracts.
`torch_npu` is the Python PTA interface. `aclnn` is a C operator interface and can
also be used by FLA and `_C_ascend` wrappers. Check the loaded library and target
kernel to identify which implementation executes.

| Provider | Active path and adaptation order |
| --- | --- |
| `fla_npu` | Qwen convolution and recurrent GDN are first priority. Preserve Python signatures, state layout, accepted-token semantics and BF16/MTP coverage. Use PR1134 as a recurrent candidate and qualify causal convolution separately. Treat package fused prefill as a later backend choice. |
| Triton Ascend | Qwen chunk prefill uses L2Norm, cumsum, scaled KKT, triangular solve and W/U preparation. Qualify these alongside the AscendC consumers using the actual compiler. Check launch geometry and numerical parity per stage. Investigate fused MRoPE separately from the working decomposed path. |
| `_C_ascend` | Qualify Qwen FwdH and FwdO together. For DSv4, prioritize matching FP8 QLI compute/metadata and its cache writers, then restore fused compressor and mHC pre contracts. Qualify legacy SFA/QSFA according to the selected model and cache plan. Port GMM-SiTU when the selected quantization invokes it. |
| PTA and delivered CANN packages | Validate the A5-compatible math and layout for each called quantization, matmul, scatter and norm interface on the target stack. Keep delivered QLI, DSv4 KV epilog/attention and compressor/mHC alternatives behind their existing boundaries until replacements pass parity. Track installed package, operator schema, resolved library and SoC binary together. |

DSv4 attention KV and indexer KV are separate. The FP8 attention-KV plan selects
delivered `kv_compress_epilog` and mixed-quant sparse MLA with matching metadata.
The FP8 indexer uses QLI compute/metadata, its own cache writer and scale layout.
The BF16 attention-KV plan uses sparse flash MLA with its metadata and a PTA
scatter. Preserve each producer/consumer pair and verify cache bytes, padding,
slot mapping and replay before changing either path. Restore the A5 QLI 36/72
metadata contract or qualify a complete versioned producer/consumer migration
before sharing a new layout across device families.

## Validation and handoff

The existing hardware-profile matrix tracks declared capabilities. Native imports,
CANN compilation and model execution require the target NPU environment.

For each native check, record SoC, actual AIC/AIV counts, driver, CANN, torch,
torch_npu, Triton, FLA and delivered package versions, commit and build hashes.
Use the user's actual Qwen3.5-35B and DSv4-Flash weights, quantization and launch
configuration. Compare one implementation change at a time with fixed inputs.

Required qualification includes operator output/state parity, tails and empty
requests, GDN chunk boundaries, prefill/decode, continuous cache updates, eager
and repeated graph replay. Measure performance separately with identical
workloads. Multi-card TP/EP coverage must match the intended deployment.

The release-note fragment for this change is `release_notes/fragments/55.yaml`.

## Pinned reference sources

- [FLA A6-dev at 1a2baa0](https://github.com/flashserve/flash-linear-attention-npu/tree/1a2baa070a02f5f4a5a6847a9248cd807540db60)
- [FLA PR1134](https://github.com/flashserve/flash-linear-attention-npu/pull/1134), inspected head `d78cc1d88853bb6afe333972734fabdfb00270c3`
- [CANN ops-transformer at bd8de49](https://gitcode.com/cann/ops-transformer/tree/bd8de49304dd9a32f475ab44bcacd00229e92349)
- [CANN ops-nn at 5b1b131](https://gitcode.com/cann/ops-nn/tree/5b1b131b1edc492c95d7ba60c534ce74dedeb8d6)
