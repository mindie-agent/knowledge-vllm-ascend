---
blocks:
- block_id: 1319da3aa4895ce4be6215501a43d17e76d86b09e281f0e15bbe3cf97f006529
  indexed: true
  sha256: c3278703f6c75d93dd63abc26bda0469617e22fb620ed32e1a813c523fc155e2
  source_range:
    body_end: 1987
    body_start: 0
    published_commit: 9aab5e1e2f1298f992b1b41e403c2b1e0232d12c
    published_path: cases/cbc41d955cd8abf495b4129bbdff15fbf3550476a8d1312bf3841cf75033773a.md
  summary: A small public-API check script comparing torch_npu.npu_rms_norm against a CPU float32 RMSNorm reference was written to /tmp on a remote container and run with python3. On torch 2.10.0+cpu / torch_npu 2.10.0.post4.dev20260715, device Ascend910_9362, both FP16 and BF16 cases (x=[3,256], w=[256], epsilon=1e-6) returned finite outputs with allclose=True under tolerances rtol=atol=2e-3 (FP16) and 2e-2 (BF16); exit code 0. Two CANN/allocator warnings were printed.
  title: torch_npu npu_rms_norm FP16/BF16 small-input check vs CPU reference on Ascend910_9362
entry:
  conditions:
    torch: 2.10.0+cpu
    torch_npu: 2.10.0.post4.dev20260715
  domain: vllm-ascend
  entry_id: cbc41d955cd8abf495b4129bbdff15fbf3550476a8d1312bf3841cf75033773a
  kind: experience
  material_digest: 80bdd52a60183114c6631e8ce923d1e0f06123b5b97b721540527984fc625d88
  revision: 91fed1a1b2eb62f91a2956283b7d4f11857221da12fc920e245f107413b95933
  schema: mindie-entry/3
  summary: A small public-API check script comparing torch_npu.npu_rms_norm against a CPU float32 RMSNorm reference was written to /tmp on a remote container and run with python3. On torch 2.10.0+cpu / torch_npu 2.10.0.post4.dev20260715, device Ascend910_9362, both FP16 and BF16 cases (x=[3,256], w=[256], epsilon=1e-6) returned finite outputs with allclose=True under tolerances rtol=atol=2e-3 (FP16) and 2e-2 (BF16); exit code 0. Two CANN/allocator warnings were printed.
  title: torch_npu npu_rms_norm FP16/BF16 small-input check vs CPU reference on Ascend910_9362
navigation: A small public-API check script comparing torch_npu.npu_rms_norm against a CPU float32 RMSNorm reference was written to /tmp on a remote container and run with python3. On torch 2.10.0+cpu / torch_npu 2.10.0.post4.dev20260715, device Ascend910_9362, both FP16 and BF16 cases (x=[3,256], w=[256], epsilon=1e-6) returned finite outputs with allclose=True under tolerances rtol=atol=2e-3 (FP16) and 2e-2 (BF16); exit code 0. Two CANN/allocator warnings were printed.
schema: mindie-material-task/1
status: complete
task_id: cbc41d955cd8abf495b4129bbdff15fbf3550476a8d1312bf3841cf75033773a
---

# torch_npu npu_rms_norm FP16/BF16 small-input check vs CPU reference on Ascend910_9362

A small public-API check script comparing torch_npu.npu_rms_norm against a CPU float32 RMSNorm reference was written to /tmp on a remote container and run with python3. On torch 2.10.0+cpu / torch_npu 2.10.0.post4.dev20260715, device Ascend910_9362, both FP16 and BF16 cases (x=[3,256], w=[256], epsilon=1e-6) returned finite outputs with allclose=True under tolerances rtol=atol=2e-3 (FP16) and 2e-2 (BF16); exit code 0. Two CANN/allocator warnings were printed.

## Materials

- [torch_npu npu_rms_norm FP16/BF16 small-input check vs CPU reference on Ascend910_9362](blocks/1319da3aa4895ce4be6215501a43d17e76d86b09e281f0e15bbe3cf97f006529.md): A small public-API check script comparing torch_npu.npu_rms_norm against a CPU float32 RMSNorm reference was written to /tmp on a remote container and run with python3. On torch 2.10.0+cpu / torch_npu 2.10.0.post4.dev20260715, device Ascend910_9362, both FP16 and BF16 cases (x=[3,256], w=[256], epsilon=1e-6) returned finite outputs with allclose=True under tolerances rtol=atol=2e-3 (FP16) and 2e-2 (BF16); exit code 0. Two CANN/allocator warnings were printed.
