---
category: sop
conditions:
  fault_class:
  - runtime_error
  - hang
  - aicore_exception
  - hccl_timeout
  framework: vllm-ascend
domain_path: vllm-ascend/runtime/cannbot-runbooks
evidence_strength: software_contract
id: unit-vllm-ascend-cann-error-code-and-hang-crash-triage-sop
inherits_from:
- unit-vllm-ascend-npu-runtime-and-graph-invariants
provenance:
  evidence_refs:
  - https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920beta2/maintenref/troubleshooting/docs/zh/troubleshooting/error-codes/Driver-Errors/EL0004-Memory_Allocation_Failure.md
  - https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920beta2/maintenref/troubleshooting/docs/zh/error_codes/EI0006-Communication_Error_Get_Socket.md
  - https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920beta2/maintenref/troubleshooting/docs/zh/error_codes/EI0002-Communication_Error_Timeout.md
  - https://gitcode.com/cann/oam-tools
  - https://github.com/vllm-project/vllm-ascend/issues/4051
  - https://github.com/vllm-project/vllm-ascend/pull/3005
  - https://github.com/vllm-project/vllm-ascend/pull/3158
  experiment_status: not_run_source_only
  source_digest: 76476793563a667b5d66b04c526d97113ba6d24e5a9d2e52a7a4b75ae380cc2c
  source_locator: public/cannbot-skills/ops/cann-error-code-standard/references/development-standard.md
  source_type: cannbot_skill_and_runbook_distillation
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: CANN error-code, AICore exception, HCCL hang and OOM triage SOP for vllm-ascend
triggers:
  error_signatures:
  - EZ9999 AclNN_Inner_Error with no semantic cause in the message
  - EL0004 Memory_Allocation_Failure or 507001 event timeout during a hang
  - EI0006 Get_Socket_Timeout or EI0002 Notify register timeout
  - ErrCode=507015 aicore exception or ErrCode=507018 aicpu exception
  - ErrCode=507035 vector core exception
  - PrintCoreInfo reports aic error mask, PC start and IFU/MTE/VEC/CUBE status
  - rank zero HCCL timeout while another rank has the first real error
  - service hangs after a graph capture or an idle DP step
  keywords:
  - CANN error code
  - EZ9999
  - EL0004
  - EI0006
  - EI0002
  - '507015'
  - '507018'
  - '507035'
  - '507001'
  - AICore
  - AICPU
  - vector core
  - plog
  - aic error mask
  - PC start
  - ASCEND_LAUNCH_BLOCKING
  - msaicerr
  - asys
  - hccn_tool
  - HCCL timeout
  - OOM
  - memory_summary
---

# CANN error-code, AICore exception, HCCL hang and OOM triage SOP for vllm-ascend

## Applicability and diagnostic stance

Use this SOP when a `vllm-ascend` process exits with a CANN or runtime error, hangs in a collective, or reports an AICore/vector-core exception. It covers the first ten minutes of triage: decode the code, find the first causal ERROR, separate AICore from HCCL from memory, and collect a bundle that another engineer can reproduce.

The SOP deliberately does not treat the last error as the cause. CANN collectives are asynchronous and the reported stack can be inaccurate; the rank that times out is often not the rank that failed.

## Procedure

1. **Decode the six-character error code and locate the earliest root-cause ERROR line across all ranks.**

   A CANN code is `[Level][Module][Code]`: the leading `E` is the error level, the second letter is the reporting module, and the four digits are the module-local code. Read the module first; do not rank severity by the four digits.

   | Code | Reporting domain | What it does and does not prove |
   | --- | --- | --- |
   | `EZ9999` | Operator / AclNN (Opiling host, tiling or inner error) | A generic internal error. It may wrap a real kernel, tiling or parameter failure; it is not itself a root cause. |
   | `EL0004` | Driver | Memory allocation failure. Confirm device versus host memory before changing a batch size. |
   | `EL0006` | Driver | Insufficient resources. In graph capture this often means SQ/CQ stream resources, not HBM. |
   | `EI0006` | HCCL | Get-socket timeout. It can be a missing peer, a rank that exited, or a physical link drop. |
   | `EI0002` | HCCL | Notify-register or collective wait timeout. It is usually a peer-participation or peer-failure symptom. |
   | `507015` | Runtime / RTS | AICore exception; the AI Core kernel faulted. |
   | `507018` | Runtime / RTS | AICPU exception; the AICPU kernel faulted. |
   | `507035` | Runtime / RTS | Vector-core exception; often an address, alignment or DataCopyPad contract violation. |
   | `507001` | Runtime / RTS | Event timeout. Treat it as a secondary stall/timeout symptom until an allocation or device fault is confirmed. |

   Search the plog, not only stdout. The usual path is `$HOME/ascend/log/debug/plog/plog-<pid>_*.log`; older or containerized layouts may use `$HOME/ascend/log/plog/`. Search every rank and sort by timestamp:

   ```bash
   LOG_ROOT="${ASCEND_PROCESS_LOG_PATH:-$HOME/ascend/log}"
   grep -RInE "\[ERROR\]|ErrCode=|error code|EZ[0-9]{4}|EI[0-9]{4}|EL[0-9]{4}" "$LOG_ROOT" \
     | sort -t: -k1,1 -k2,2n | head -80
   ```

   Read the first `[ERROR]` that names a concrete module, file and line. A later `EI0006` on rank zero is not the first error if rank seven already logged `EL0004`.

   **Invariant:** one root cause per incident; the first causal ERROR is earlier than, and not a consequence of, the rank-zero timeout; the decoded module matches the component that owns the failing operation.

   **Verify:** the chosen line has a PID, a source file and line, and a code whose module matches the failing operation (operator, HCCL, driver or runtime). If every candidate is a timeout, go to step 3 before changing code.

2. **For `507015`, `507018`, `507035` or `EZ9999`, extract the AI Core state and reproduce the exact operator call.**

   A `PrintCoreInfo` line carries the evidence needed to distinguish an address fault from an illegal instruction or a bad input:

   ```text
   PrintCoreInfo:An error occurs on the device(...), the error is aicore error,
   core id is ..., error code = ..., dump info: pc start: 0x..., current: 0x...,
   ifu error info: ..., mte error info: ..., vec error info: ...,
   cube error info: ..., aic error mask: ..., para base: ...
   ```

   `error code = 0` is not "no error". Read `aic error mask`, `pc start`, `current`, and the `IFU`, `MTE2`, `MTE3`, `VEC` and `CUBE` status words. A non-zero `mte` or `vec` status with a PC inside a copy or vector instruction points to an address/stride problem; a `cube` status with an invalid-input string points to NaN/Inf or a bad L0A/L0B operand.

   Re-run the smallest failing case with synchronous launch so the Python stack names the exact call:

   ```bash
   export ASCEND_LAUNCH_BLOCKING=1
   python3 <single_case.py> 2>&1 | tee "$HOME/aicore_repro.log"
   ```

   Then run the two CANN collectors:

   ```bash
   # msaicerr: run outside the -p directory, with separate -p and -out paths.
   # Set ASCEND_TOOLKIT_HOME to the CANN toolkit root, or use its actual install path.
   cd "${ASCEND_TOOLKIT_HOME:?set to the CANN toolkit root}/tools/msaicerr"
   python3 msaicerr.py -p "$HOME/aic_err_info" -out "$HOME/msaicerr_out" -dev 0

   # asys: parse the AI Core error and, if needed, collect the live stack.
   asys analyze -r=aicore_error --path "$HOME/aic_err_info"
   asys collect -r=stacktrace --output="$HOME/dfx_info"
   ```

   `msaicerr` analyzes only the first AI Core error in a bundle and does not support every fused operator. If the fault is inside a SuperKernel or a fused chain, disable fusion and rerun so the failing operator becomes a standalone call. `asys` is available only in the Ascend EP form; on an RC image, use `msaicerr` and the plog path instead.

   **Invariant:** the failing operator, shape, stride, dtype and seed are recorded; the reproducer calls the same kernel variant as the service (not a fallback); the AICore status words are read before the operator is changed.

   **Verify:** the single-operator case reproduces the same code and the same `pc start`/status on the same device; after the fix it passes with `ASCEND_LAUNCH_BLOCKING=1`, then again with `ASCEND_LAUNCH_BLOCKING=0`. A fix that only passes in eager is not a graph fix.

3. **For `EI0006` or `EI0002`, compare every rank's last collective and separate rank skew from a physical link fault.**

   Collect, for every rank, the last launched collective and the process stack at the hang:

   ```bash
   grep -RInE "Hccl(AllReduce|AllGather|AllToAll|Broadcast|ReduceScatter)|current working operator" "$LOG_ROOT" | tail -40
   for pid in $(pgrep -f "vllm|mindie"); do
     echo "===== $pid ====="
     cat "/proc/$pid/stack" 2>/dev/null || true
   done
   ```

   Then check the physical side:

   ```bash
   hccn_tool -i <device-id> -link -g
   hccn_tool -i <device-id> -stat -g
   hccn_tool -i <device-id> -tls -g
   ```

   Interpret the combination:

   - **All ranks entered the same collective, one link is down:** physical or configuration fault; fix the link, the rank table or the transport selection.
   - **One rank never entered the collective, peers wait:** rank skew or an early return on that rank. Check `num_scheduled_tokens == 0`, an idle DP branch, a missing dummy run, or an exception before the collective.
   - **One rank logged `EL0004` or an AICore fault earlier:** the timeout is secondary. Go to step 2 or step 4; do not chase the network.

   `hccn_tool -i ... -tls -g` returning normal does not exonerate rank skew. TLS health only says the link can be established; it does not say every rank called the same collective.

   **Invariant:** the diagnosis names either a rank that did not participate or a link that is physically down; the two are not conflated; the first non-collective error is checked before the timeout.

   **Verify:** with `ASCEND_LAUNCH_BLOCKING=1`, the failing collective reports synchronously on the rank that actually failed; after the repair, all ranks enter and leave the same sequence and exit zero. Remove launch blocking before measuring performance.

4. **For `EL0004`, `507001` or a suspected OOM, classify the memory domain and the pool before changing a batch knob.**

   `507001` is a runtime event-timeout code. It often appears when an allocation failure or device fault stalls the stream, so it is a secondary symptom unless an allocation failure is independently confirmed.

   Collect the device and host view:

   ```bash
   npu-smi info
   python3 - <<'PY'
   import torch, torch_npu
   print(torch.npu.memory_summary())
   print("allocated", torch.npu.memory_allocated() / 2**30, "GiB")
   print("reserved ", torch.npu.memory_reserved() / 2**30, "GiB")
   print("max allocated", torch.npu.max_memory_allocated() / 2**30, "GiB")
   PY
   free -g
   ```

   Then separate the domains:

   - **Device HBM OOM:** `EL0004`, a device allocation failure, or `npu-smi` showing the card full. Reduce `max_num_batched_tokens`, `max_num_seqs`, the KV cache fraction, or the capture pool.
   - **Host OOM:** `EH0010` or a host allocator failure. Reduce host-side metadata, process count or pinned memory; freeing NPU HBM will not help.
   - **SQ/CQ or workspace exhaustion:** `Alloc sq cq fail`, `207005`, `EL0006`. This is a stream-resource problem; reduce capture gears or communication groups, not the KV cache.

   For an ACLGraph service, record the capture footprint separately from the steady-state footprint. A capture-time OOM with a lower steady-state footprint is a pool or reservation problem, not a model-too-large problem. `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` can reduce fragmentation; it does not create capacity.

   **Invariant:** the failing domain (device, host, stream/workspace) is identified before a knob is changed; a secondary `507001` is not treated as proof of memory corruption; the fix is applied to the correct pool.

   **Verify:** after the change, 100 steady decode steps show a memory plateau, no `EL0004`, and no new `Alloc sq cq fail`; the same trace passes with the original batch size if the root cause was fragmentation rather than capacity.

5. **Collect a minimal reproducible bundle and verify the fix under normal asynchronous execution.**

   A bundle must contain the smallest case that still fails, not the whole service log. Include:

   - the exact CANN, driver, torch_npu, vLLM and `vllm-ascend` revisions, SoC and die count;
   - the parallel topology, graph mode, dtype/quantization and KV-cache mode;
   - the failing command, environment variables (with secrets removed), seed and input shapes/strides;
   - the first `[ERROR]` line, the full `PrintCoreInfo`, the relevant plog slice and the `msaicerr` `info.txt`;
   - the single-layer or single-operator reproducer script and its expected versus actual tensor.

   ```bash
   asys collect --tar=True --output="$HOME/dfx_info"
   # Keep the msaicerr output directory separate from the input directory.
   ls "$HOME/msaicerr_out" "$HOME/dfx_info"
   ```

   Then verify the fix in the real execution mode:

   ```bash
   unset ASCEND_LAUNCH_BLOCKING
   export ASCEND_LAUNCH_BLOCKING=0
   # 10 warmup steps, then 100 measured steps on the same trace
   python3 <service_smoke.py>
   ```

   **Invariant:** the bundle reproduces the pre-fix failure; the fix passes the same case after launch blocking is removed; the failed logs are preserved rather than overwritten by a clean retry.

   **Verify:** the pre-fix bundle fails on the same code and line; the post-fix run completes 100 steps with no error, no timeout and parity against the reference; the bundle contains no credential, internal address or user-home path.

## Boundaries and escalation

The error-code catalog proves public semantics, not that the target build registered or used a code. A code can be a wrapper: `EZ9999` in particular is a generic internal code and must not be promoted to a root cause. `msaicerr` does not support every fused operator and analyzes only the first AI Core error; a SuperKernel fault may require disabling fusion before it can be analyzed. The `hccn_tool` fields and the plog layout vary by CANN version, so the first causal line and the tool's own output remain authoritative.

Escalate to the kernel owner when the same single-operator reproducer fails with the same PC and status after the input contract, tiling and workspace have been checked. Escalate to the collective owner when all ranks enter the same collective and a physical link is confirmed down. Escalate to the platform owner when `npu-smi` reports a device health fault rather than an application allocation failure.
