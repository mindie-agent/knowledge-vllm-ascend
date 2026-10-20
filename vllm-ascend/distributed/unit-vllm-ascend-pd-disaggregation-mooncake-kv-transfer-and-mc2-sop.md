---
category: sop
conditions:
  framework: vllm-ascend
  serving_mode:
  - pd_disaggregated
  - multi_node_ep
  soc:
  - Ascend910B
  - Ascend910_93
  - Ascend950
domain_path: vllm-ascend/distributed
evidence_strength: reproduced_empirical
id: unit-vllm-ascend-pd-disaggregation-mooncake-kv-transfer-and-mc2-sop
inherits_from:
- unit-vllm-ascend-npu-runtime-and-graph-invariants
- unit-vllm-ascend-hccl-and-dp-collective-invariants
provenance:
  evidence_refs:
  - https://github.com/vllm-project/vllm-ascend/pull/1691
  - https://github.com/vllm-project/vllm-ascend/issues/4842
  - https://github.com/vllm-project/vllm-ascend/issues/5044
  - https://github.com/vllm-project/vllm-ascend/pull/2947
  - https://github.com/vllm-project/vllm-ascend/issues/4051
  - https://gitcode.com/cann/cann-recipes-infer/merge_requests/452
  - https://gitcode.com/cann/cann-recipes-infer/merge_requests/421
  - https://gitcode.com/cann/cann-recipes-infer/issues/160
  source_digest: e5ac79ef5eed0a027306b53757acd1bcf3084287cd0ffdce7fc115b363a41e09
  source_locator: cann/cannbot-knowledge/knowledge/model/inference/runbooks/runtime/mooncake_ascend_hixl_kv_transfer_bringup.md
  source_type: cannbot_runbook_and_transcript_distillation
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: PD disaggregation with Mooncake/LlmDataDist KV transfer and MC2 dispatch/combine bring-up SOP
triggers:
  error_signatures:
  - Mooncake Ascend backend initialize returns non-zero or reports USE_ASCEND_DIRECT missing
  - decode transfer fails with 103901 while plog shows mem type 0 but no mem type 1
  - KV block length or offset is wrong under torchair three-dimensional cache layout
  - prefill and decode have different TP sizes and decode reads permuted or stale KV heads
  - HCCL_BUFFSIZE is too SMALL or MoeDistributeDispatchV2 do tiling failed
  - EI0006 socket timeout, UB LINK ERROR or a missing rank during PD bring-up
  - decoded text is corrupted only at high concurrency because prefill released a KV block too early
  keywords:
  - PD disaggregation
  - prefill
  - decode
  - MooncakeConnector
  - MooncakeConnectorStoreV1
  - LlmDataDist
  - KVTransfer
  - HIXL
  - RoCE
  - HCCS
  - register_buffer
  - mem type
  - block_rank
  - asymmetric TP
  - num_kv_heads
  - MC2
  - Dispatch
  - Combine
  - HCCL_BUFFSIZE
  - moe_ep_group_mc2
  - fullmesh_v2
  - ranktable
  - Phy-ID
  - Ascend910_93
---

# PD disaggregation with Mooncake/LlmDataDist KV transfer and MC2 dispatch/combine bring-up SOP

## Applicability and failure boundary

Use this SOP when a `vllm-ascend` deployment separates prefill (P) and decode (D) and moves paged KV cache over a Mooncake or `LlmDataDist` connector, especially when P and D use different tensor-parallel widths or when the MoE expert path is enabled. It covers bring-up order, transport registration, KV-head re-sharding, MC2 communication-domain setup and the first end-to-end PD smoke.

It does not cover model-quality tuning, scheduler policy, or the internal arithmetic of attention and grouped matmul. A failure that is already isolated to one kernel's numerical result belongs in the kernel triage path, not here.

## Procedure

1. **Freeze the PD topology, the die-level rank table and the transport mode before starting either side.**

   Record on both P and D: `P_TP`, `D_TP`, `P_DP`, `D_DP`, `PP`, `EP`, `num_kv_heads`, `head_dim`, `block_size`, `page_size`, dtype of the KV cache, and whether ACLGraph or TorchAir is enabled. On Ascend A3 a board exposes two dies, so a "card count" is not a rank count: enumerate every die and bind one process to one die. Verify that the rank table and the process-local device selection agree.

   ```bash
   npu-smi info
   npu-smi info -t topo
   # ranktable: inspect the logical rank -> device_id / Phy-ID mapping
   python3 - <<'PY'
   import json
   rt = json.load(open("<ranktable.json>"))
   for server in rt.get("server_list", []):
       for dev in server.get("device", []):
           print(server.get("server_ip"), dev.get("device_id"), dev.get("rank_id"))
   PY
   ```

   Set the data-plane interface and transport selection explicitly. `HCCL_INTRA_ROCE_ENABLE=0` selects the intra-node HCCS path; `HCCL_INTRA_ROCE_ENABLE=1` selects the intra-node RoCE path. Do not leave it to the default if the cluster mixes HCCS and RoCE, and do not set different values inside one communication domain.

   ```bash
   export HCCL_IF_IP="<data-plane-ip>"
   export GLOO_SOCKET_IFNAME="<data-plane-nic>"
   export HCCL_INTRA_ROCE_ENABLE=0        # 1 only when the domain is deliberately RoCE
   export HCCL_CONNECT_TIMEOUT=600        # diagnostic headroom, not a fix
   ```

   **Invariant:** one logical rank maps to exactly one die; every rank's `torch.npu.set_device(local_rank)` resolves to the same die recorded in the rank table; `HCCL_IF_IP`, `GLOO_SOCKET_IFNAME` and `HCCL_INTRA_ROCE_ENABLE` are identical for all ranks in one domain.

   **Verify:** `npu-smi info` shows the expected `Phy-ID` per rank and no rank is bound to the wrong die; `python3 -c "import torch, torch_npu; print(torch.npu.current_device())"` agrees with the intended local rank; a four-rank `all_reduce` on the selected transport completes before any model is loaded.

2. **Build and register the KV transport before constructing the connector.**

   For the Mooncake Ascend direct backend, build with the Ascend-direct flag, put the resulting libraries on `LD_LIBRARY_PATH`, and construct the engine only after the process has selected its NPU. The engine reads the physical device from the current Ascend context, so a missing `set_device` silently binds the wrong die.

   ```bash
   cmake .. -DUSE_ASCEND_DIRECT=ON && make -j"$(nproc)" && make install
   export LD_LIBRARY_PATH="<mooncake-build>/lib:$LD_LIBRARY_PATH"
   ```

   ```python
   import torch
   import torch_npu
   from mooncake.engine import TransferEngine

   torch_npu.set_device(local_rank)          # must precede engine construction
   engine = TransferEngine()
   ret = engine.initialize(hostname, "P2PHANDSHAKE", "ascend", device_name)
   if ret != 0:
       raise RuntimeError(f"Mooncake Ascend initialize failed: {ret}")
   session_id = f"{hostname}:{int(engine.get_rpc_port())}"
   ```

   `initialize` takes four strict positional arguments. Named arguments are rejected by the pybind signature. For HCCS IPC, every registered base address must be 2 MiB aligned; over-allocate and narrow rather than assuming `torch.empty` alignment. The RoCE path selected by `HCCL_INTRA_ROCE_ENABLE=1` does not require that alignment, but keeping the aligned allocation is harmless.

   For the vLLM `MooncakeConnectorStoreV1` path, a store-only deployment does not implicitly register NPU device memory. Add the explicit switch on both P and D:

   ```yaml
   kv_connector: MooncakeConnectorStoreV1
   kv_role: kv_producer        # kv_consumer on D
   kv_connector_extra_config:
     register_buffer: true
   ```

   For `LlmDataDist`, use the connector's producer and consumer roles and the same rank-table and HCCL environment; the transport must be registered on both sides before the first block is pulled or pushed.

   **Invariant:** the engine is constructed after `set_device`; `initialize` returns zero; `register_kv_caches` derives block rank from the live cache rank, `block_rank = len(first_kv_cache.shape) - 1`, and computes `block_len = product(shape[1:]) * dtype_size`. Never hard-code `block_rank = 3`; eager KV is four-dimensional while a TorchAir KV cache is a flattened three-dimensional tensor.

   **Verify:** the P and D plogs contain `mem type:1` (NPU HBM registration) in addition to `mem type:0` (host registration); decode does not report transfer error `103901`; a single-block write-then-read returns identical bytes; a 2 MiB alignment assertion passes for every registered base address.

3. **Re-shard KV blocks and KV heads when `P_TP != D_TP`, and hold the prefill block until every decode rank acknowledges it.**

   KV ownership is by attention head, not by token. Compute the canonical head interval for every rank:

   ```python
   # Case 1: enough KV heads to shard. Require an exact division or an explicit pad.
   if num_kv_heads >= P_TP:
       assert num_kv_heads % P_TP == 0
       owned_heads_p = num_kv_heads // P_TP   # canonical ownership interval
       local_heads_p = owned_heads_p          # actual local cache width
   else:
       # Case 2: fewer KV heads than TP ranks. Each rank stores the full replicated
       # cache, but ownership is one canonical head: keep one copy and do not sum replicas.
       owned_heads_p = 1
       local_heads_p = num_kv_heads

   # The destination uses the same rule with D_TP.
   # Canonical global head for local head h on rank r (non-replicated case):
   #   global_h = r * owned_heads_p + h
   # D rank d receives the intersection of its own head interval with each P rank's interval.
   ```

   When `P_TP > D_TP` and `num_kv_heads >= P_TP`, one D rank needs the KV shards of several P ranks and must concatenate them in canonical head order. When `P_TP < D_TP` and `num_kv_heads >= D_TP`, one P shard is split across several D ranks by head; do not split by token, block or sequence position. When `num_kv_heads < P_TP` or `num_kv_heads < D_TP` (common for MLA with a latent KV head), several ranks hold the same head: deduplicate by canonical head id and transfer the block once. Pad KV heads only if the model and attention operator explicitly support padding, and never silently truncate `num_kv_heads`.

   The transfer unit is one paged block. Reconstruct the destination row with the request's `block_table` and `slot_mapping`; a block may be physically non-contiguous across the page pool. Under TorchAir, the cache is `[num_block, block_size, num_kv_heads * head_dim]`; under eager it is `[num_block, block_size, num_kv_heads, head_dim]`. Both must produce the same `block_len`.

   ```python
   import math

   shape = tuple(first_kv_cache.shape)
   # connector-side dimension count; block_len is the byte length actually transferred
   block_rank = len(shape) - 1
   block_len = math.prod(shape[1:]) * first_kv_cache.element_size()
   assert block_len == block_size * local_heads_p * head_dim * first_kv_cache.element_size()
   ```

   In a pull-based path, prefill must not release the request's blocks when `prefill` finishes. Delay release until all decode TP ranks have acknowledged receipt. Count distinct `tp_rank` values in a set; a duplicate acknowledgement from the same rank must be ignored, otherwise the barrier can be satisfied early.

   ```python
   delay_free = len(computed_block_ids) > 0
   # D -> P completion event carries (request_id, tp_rank, tp_size)
   if tp_rank in done_receiving_counts[request_id]:
       return False
   done_receiving_counts[request_id].add(tp_rank)
   if len(done_receiving_counts[request_id]) == decode_tp_size:
       done_receiving_counts.pop(request_id)
       return True
   return False
   ```

   **Invariant:** the union of owned canonical head intervals covers `num_kv_heads` exactly once after re-sharding; in the replicated case, multiple ranks map to the same canonical interval and it is counted once; a block is released only after the distinct decode-rank count reaches `D_TP`; no P rank frees a block while any D rank can still read it.

   **Verify:** with a prompt longer than `block_size`, dump the first and last block's K and V on P and D and compare after dequantization; run one case with page offset zero and one with a non-zero storage offset; run both a sharded case (`num_kv_heads >= 8`, `P_TP=8, D_TP=4` and `P_TP=4, D_TP=8`) and a replicated case (`num_kv_heads < TP`, such as an MLA latent head); at high concurrency, decoded tokens must not change when P is artificially slowed or when a second request reuses the same block pool.

4. **Initialize the MC2 dispatch/combine domain and size its HCCL buffer from the workload.**

   MC2 `npu_moe_distribute_dispatch_v2` / `npu_moe_distribute_combine_v2` must run on a dedicated communication domain, typically `moe_ep_group_mc2`. Do not attach it to `moe_ep_group`, the TP group, or a CP group. The dedicated domain also confines the enlarged buffer to the decode MC2 path.

   ```bash
   # illustrative; the exact launch variable is framework-specific
   export HCCL_BUFFSIZE="<computed-mb>"
   export HCCL_OP_EXPANSION_MODE=AIV     # optional latency optimization; re-check determinism for evaluation
   ```

   Compute the requirement from the operator's own formula rather than copying a number:

   ```text
   NEEDED_HCCL_BUFFSIZE =
     (maxBs * tokenNeedSizeDispatch * ep_worldsize * localMoeExpertNum
      + maxBs * tokenNeedSizeCombine * (k + sharedExpertNum)) * 2
   ```

   All ranks must use the same value, rounded up with headroom. The required value grows with EP world size, experts per rank and maximum batch. A value copied from a different model is not evidence.

   On Ascend A2, MC2 dispatch/combine v2 is capped at 24 experts per rank in the unlayered mode. If `experts_per_rank` exceeds that, do not "fix" it by raising the buffer: fall back to eager or disable `enable_moe_mc2_dispatch` and use the prefill double-routing path. Also do not enable `comm_alg="fullmesh_v2"` for a MoE with zero or identity experts; keep the dedicated domain but omit `fullmesh_v2` and pass `copy_expert_num`.

   Under TorchAir with `replace_allreduce=True`, split `mc2_mask` by TP rank whenever `tp_size > 1`, outside the `not replace_allreduce` branch, so the mask and the split token tensor keep the same leading dimension.

   **Invariant:** MC2 owns one non-shared HCCL domain; all ranks use one `HCCL_BUFFSIZE`; `experts_per_rank <= 24` on A2; `fullmesh_v2` is not combined with zero experts; `mc2_mask.shape[0]` equals the split token count for every TP rank.

   **Verify:** the first `profile_run` / dummy forward completes without `MoeDistributeDispatchV2 do tiling failed` or `HCCL_BUFFSIZE is too SMALL`; `npu-smi info` shows no unexpected memory growth after the dummy forward; a two-rank dispatch/combine returns `expert_token_num` whose sum equals `num_tokens * top_k`; the same run succeeds in eager before it is captured in a graph.

5. **Run a staged PD smoke and use the first failing layer to choose the repair step.**

   Start with one P and one D, prefix caching off, concurrency one, a prompt that fits one page, and a short output. Then raise prompt length beyond `block_size`, then enable the production TP/DP shape, then raise concurrency.

   ```bash
   # 1P1D smoke; keep the same trace for the non-PD baseline
   curl -s http://<prefill-host>:<port>/v1/completions \
     -H 'Content-Type: application/json' \
     -d '{"model":"<model>","prompt":"<short-prompt>","max_tokens":16,"temperature":0}'
   ```

   For each stage record: first-token equality with the non-PD baseline, TTFT, TPOT, `output_tokens/s`, the P and D plogs, `npu-smi info`, and the last collective on each rank. Stop at the first stage that fails; do not raise concurrency on a failing stage.

   Failure routing:

   - `103901`, or `mem type:1` absent: return to step 2 and fix device-memory registration or engine construction order.
   - KV shape, block offset or head permutation error, or decoded text corrupt only under concurrency: return to step 3 and re-check the canonical head interval, `block_len`, `block_table` and the ACK barrier.
   - `HCCL_BUFFSIZE is too SMALL`, `MoeDistributeDispatchV2 do tiling failed`, or a mask shape mismatch: return to step 4.
   - `EI0006`, `UB LINK ERROR`, or one rank missing from the collective: return to step 1, then step 4. Check the die-level rank table, `HCCL_IF_IP`, `HCCL_INTRA_ROCE_ENABLE` and the last collective per rank before changing any model code.

   **Invariant:** a stage is accepted only when the same trace passes on repeated launches, P and D agree on the request and block identity, and no rank reports a first error that is masked by another rank's timeout.

   **Verify:** the 1P1D short-prompt run returns the same first token as non-PD; the long-prompt run returns a coherent continuation and the KV transfer log shows registered device memory; the production-shape run completes a 4P4D or larger smoke with the planned concurrency; after a clean stop and restart, the same trace passes again. If any gate fails, preserve the P and D logs and roll back to the non-PD path rather than tuning around a corrupted transfer.

## Boundaries and rollback

The 2 MiB alignment requirement is specific to the HIXL/HCCS IPC registration path; the RoCE path does not require it. The A2 expert ceiling and the fullmesh_v2 zero-expert exclusion are platform and algorithm boundaries, not general MoE rules. A PD deployment may legitimately choose a push-based layerwise schedule instead of the pull-and-ACK path; in that design the release happens after each layer is pushed, and the ACK barrier described here is not used.

Rollback is to disable the KV connector and run the same P or D instance without disaggregation, or to switch the connector backend to the previously validated one. Keep the failed logs; do not delete them to make a retry look clean.
