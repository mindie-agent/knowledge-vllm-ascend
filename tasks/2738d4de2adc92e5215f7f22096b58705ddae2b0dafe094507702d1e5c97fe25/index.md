---
blocks:
- block_id: a906f207ded68fe82be78c9f8167aab87a64476a53db84a2b29e80656659c92f
  indexed: true
  sha256: d80ec12a5c76372dfaf66dee46bd1de43c312a873c1989c7f0b6a4fdd0b0caac
  source_range:
    body_end: 6284
    body_start: 0
    published_commit: 9aab5e1e2f1298f992b1b41e403c2b1e0232d12c
    published_path: cases/2738d4de2adc92e5215f7f22096b58705ddae2b0dafe094507702d1e5c97fe25.md
  summary: The report records a Qwen3-0.6B BF16 run with two 32-token outputs, 19.868 s model loading, and 1.193 s generation. Runtime fields name NPUPlatform and device_config=npu. It also records an earlier startup exit 1 with a torch_npu fork reinitialization error; a later script deferred NPU initialization to the vLLM worker.
  title: vLLM-Ascend Qwen3-0.6B NPU preflight and inference attempts
entry:
  conditions:
    torch: 2.10.0+cpu
    torch_npu: 2.10.0.post4.dev20260715
    vllm: 0.27.1+empty
    vllm-ascend: 0.19.1rc2.dev1835
  domain: vllm-ascend
  entry_id: 2738d4de2adc92e5215f7f22096b58705ddae2b0dafe094507702d1e5c97fe25
  kind: experience
  material_digest: d9bd4840c64d61bc593603f267224ead7ec1b24e804460c49f7e11064680dbfc
  revision: 6158f905e9478ef6d0b88181b2e1b6cd4063a20c3f9b8b0dc915147e5d6bc85f
  schema: mindie-entry/3
  summary: The report records a Qwen3-0.6B BF16 run with two 32-token outputs, 19.868 s model loading, and 1.193 s generation. Runtime fields name NPUPlatform and device_config=npu. It also records an earlier startup exit 1 with a torch_npu fork reinitialization error; a later script deferred NPU initialization to the vLLM worker.
  title: vLLM-Ascend Qwen3-0.6B NPU preflight and inference attempts
navigation: The report records a Qwen3-0.6B BF16 run with two 32-token outputs, 19.868 s model loading, and 1.193 s generation. Runtime fields name NPUPlatform and device_config=npu. It also records an earlier startup exit 1 with a torch_npu fork reinitialization error; a later script deferred NPU initialization to the vLLM worker.
schema: mindie-material-task/1
status: complete
task_id: 2738d4de2adc92e5215f7f22096b58705ddae2b0dafe094507702d1e5c97fe25
---

# vLLM-Ascend Qwen3-0.6B NPU preflight and inference attempts

The report records a Qwen3-0.6B BF16 run with two 32-token outputs, 19.868 s model loading, and 1.193 s generation. Runtime fields name NPUPlatform and device_config=npu. It also records an earlier startup exit 1 with a torch_npu fork reinitialization error; a later script deferred NPU initialization to the vLLM worker.

## Materials

- [vLLM-Ascend Qwen3-0.6B NPU preflight and inference attempts](blocks/a906f207ded68fe82be78c9f8167aab87a64476a53db84a2b29e80656659c92f.md): The report records a Qwen3-0.6B BF16 run with two 32-token outputs, 19.868 s model loading, and 1.193 s generation. Runtime fields name NPUPlatform and device_config=npu. It also records an earlier startup exit 1 with a torch_npu fork reinitialization error; a later script deferred NPU initialization to the vLLM worker.
