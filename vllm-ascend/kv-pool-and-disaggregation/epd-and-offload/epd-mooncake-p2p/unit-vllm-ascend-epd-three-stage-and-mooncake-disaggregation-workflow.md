---
category: workflow
conditions:
  component: epd_and_mooncake_disaggregation
  framework: vllm-ascend
domain_path: vllm-ascend/kv-pool-and-disaggregation/epd-and-offload/epd-mooncake-p2p
evidence_strength: reproduced_empirical
id: unit-vllm-ascend-epd-three-stage-and-mooncake-disaggregation-workflow
inherits_from:
- unit-vllm-ascend-hccl-and-dp-collective-invariants
- unit-vllm-ascend-mm-encoder-attention-and-vit-aclgraph-sop
provenance:
  evidence_refs:
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:docs/source/user_guide/feature_guide/epd_disaggregation.md
  - vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:docs/source/developer_guide/Design_Documents/disaggregated_prefill.md
  source_digest: 9fdd55a9be00fbeaa65eeb236337b3c1b23054e1862187d5455858aaa5838a2e
  source_locator: vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py
  source_type: architecture_and_runtime_guide
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L3
title: Three-stage Encoder-Prefill-Decode (EPD) multimodal disaggregation and MooncakeLayerwise
  P2P workflow
triggers:
  error_signatures:
  - multimodal vision encoding blocks language prefill TTFT or cannot scale independently
    of LLM TP
  - EPD cross-process encoder cache misses because --mm-processor-cache-gb is non-zero
  - MooncakeLayerwiseConnector metaserver reverse prefill trigger fails or stalls
    on heterogeneous P/D TP
  keywords:
  - epd_disaggregation
  - ECConnector
  - ExampleConnector
  - MooncakeLayerwiseConnector
  - mooncake_layerwise_connector
  - mooncake_connector
  - disaggregated_encoder
  - metaserver
---
# Three-stage Encoder-Prefill-Decode (EPD) multimodal disaggregation and MooncakeLayerwise P2P workflow

## Trigger and scope

Use this workflow when deploying three-stage Encoder-Prefill-Decode (`1E1PD` or `1E1P1D`) multimodal disaggregation (`docs/source/user_guide/feature_guide/epd_disaggregation.md`) or two-stage P/D disaggregation with `MooncakeLayerwiseConnector` / `MooncakeConnectorV1` (`vllm_ascend/distributed/kv_transfer/kv_p2p/`).

## Stage 1 — Decouple Vision Encoder (E) from Language Prefill/Decode (PD) via `ECConnector`

1. **Separate compute-bound ViT encoding from LLM prefill/decode (`epd_disaggregation.md`):**
   - Run lightweight Vision Encoder instances (`E`) independently from heavy LLM Prefill (`P`) and Decode (`D`) fleets, transferring vision embeddings via `ECConnector` (`vllm/distributed/ec_transfer`, e.g., `ExampleConnector` over shared storage/cache).
   - Pass `--mm-processor-cache-gb 0` on all instances when using cross-process `ECConnector` caching so in-process worker caches do not bypass or conflict with the shared encoder cache.
2. **Route mixed text-only vs multimodal traffic at the EPD Load Balancing Proxy:**
   - The EPD proxy inspects incoming requests: text-only requests bypass `E` and route directly to `P`/`D`, while multimodal requests dispatch to the least-loaded `E` instance first, then forward the EC cache handle to `P`/`D`.

## Stage 2 — Pipeline Prefill-to-Decode KV push with `MooncakeLayerwiseConnector`

1. **Understand the reverse-triggered layerwise P2P flow (`mooncake_layerwise_connector.py`):**
   - Unlike bulk pull-based PD transfer, under `MooncakeLayerwiseConnector` (`load_balance_proxy_layerwise_server_example.py`), the request enters the **Decoder** instance first to pre-allocate target HBM KV blocks, and the Decoder triggers remote prefill in reverse via `/v1/metaserver`.
   - As the Prefill node completes each transformer layer $l$, its worker immediately pushes layer $l$'s KV blocks over RoCE/HCCS RDMA into the Decoder's pre-allocated HBM buffers, overlapping network transmission of layer $l$ with prefill compute of layer $l+1$.
2. **Align heterogeneous P/D parallel topologies (`prefill` vs `decode` `tp_size` / `pp_size` / `dp_size`):**
   - Explicitly declare both `"prefill": {"dp_size": ..., "tp_size": ..., "pp_size": ...}` and `"decode": {"dp_size": ..., "tp_size": ...}` inside `kv_connector_extra_config`, and ensure each instance binds a unique `kv_port` base.

## Stage 3 — Validate end-to-end EPD / Layerwise P2P health and TTFT/ITL

1. **Verify zero-stall handoff:**
   - Send mixed text and image requests through the proxy, confirm that text requests skip `E`, multimodal requests hit `ECConnector`, and `MooncakeLayerwiseConnector` completes all layer pushes without RDMA timeout.
