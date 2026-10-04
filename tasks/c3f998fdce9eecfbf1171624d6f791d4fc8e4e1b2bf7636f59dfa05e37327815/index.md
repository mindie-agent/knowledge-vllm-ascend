---
blocks:
- block_id: 89bafb72f7ba1dc374e81c1f4c302ab4f20d7fbb1edc0afe568356190bc89f99
  indexed: true
  sha256: 3ab4baa8e56d7ce2f61e54813b70ccdf5fcef8aadee2356d947dc6ae095211b8
  source_range:
    body_end: 2503
    body_start: 0
    published_commit: 9aab5e1e2f1298f992b1b41e403c2b1e0232d12c
    published_path: cases/c3f998fdce9eecfbf1171624d6f791d4fc8e4e1b2bf7636f59dfa05e37327815.md
  summary: A bounded one-logical-device check of torch_npu.npu_rms_norm for shape [7, 2048], seed 20260921, and epsilon 1e-6 passed for FP16 and BF16 when the CPU reference first rounded inputs to the tested dtype and then computed in FP32. With initial ASCEND_RT_VISIBLE_DEVICES=8, the script normalized visibility to logical device 0 before importing torch/torch_npu. The result applies only to this small operator check; physical-device mapping, multi-device behavior, serving, throughput, and performance remain unverified.
  title: Bounded Ascend torch_npu RMSNorm validation with dtype-rounded FP16/BF16 references and logical-device normalization
entry:
  conditions:
    torch_npu_version: 2.10.0.post2
    torch_version: 2.10.0+cpu
  domain: vllm-ascend
  entry_id: c3f998fdce9eecfbf1171624d6f791d4fc8e4e1b2bf7636f59dfa05e37327815
  kind: experience
  material_digest: 28715992ff2b57c8410c6f1fdabe1495f2b8233042055e65416e1b9a7ec07285
  revision: 74f5e961426a9e46fc6f21e7b3f3acd2478724ed0d81a42807c127d3cb8e1c80
  schema: mindie-entry/3
  summary: A bounded one-logical-device check of torch_npu.npu_rms_norm for shape [7, 2048], seed 20260921, and epsilon 1e-6 passed for FP16 and BF16 when the CPU reference first rounded inputs to the tested dtype and then computed in FP32. With initial ASCEND_RT_VISIBLE_DEVICES=8, the script normalized visibility to logical device 0 before importing torch/torch_npu. The result applies only to this small operator check; physical-device mapping, multi-device behavior, serving, throughput, and performance remain unverified.
  title: Bounded Ascend torch_npu RMSNorm validation with dtype-rounded FP16/BF16 references and logical-device normalization
navigation: A bounded one-logical-device check of torch_npu.npu_rms_norm for shape [7, 2048], seed 20260921, and epsilon 1e-6 passed for FP16 and BF16 when the CPU reference first rounded inputs to the tested dtype and then computed in FP32. With initial ASCEND_RT_VISIBLE_DEVICES=8, the script normalized visibility to logical device 0 before importing torch/torch_npu. The result applies only to this small operator check; physical-device mapping, multi-device behavior, serving, throughput, and performance remain unverified.
schema: mindie-material-task/1
status: complete
task_id: c3f998fdce9eecfbf1171624d6f791d4fc8e4e1b2bf7636f59dfa05e37327815
---

# Bounded Ascend torch_npu RMSNorm validation with dtype-rounded FP16/BF16 references and logical-device normalization

A bounded one-logical-device check of torch_npu.npu_rms_norm for shape [7, 2048], seed 20260921, and epsilon 1e-6 passed for FP16 and BF16 when the CPU reference first rounded inputs to the tested dtype and then computed in FP32. With initial ASCEND_RT_VISIBLE_DEVICES=8, the script normalized visibility to logical device 0 before importing torch/torch_npu. The result applies only to this small operator check; physical-device mapping, multi-device behavior, serving, throughput, and performance remain unverified.

## Materials

- [Bounded Ascend torch_npu RMSNorm validation with dtype-rounded FP16/BF16 references and logical-device normalization](blocks/89bafb72f7ba1dc374e81c1f4c302ab4f20d7fbb1edc0afe568356190bc89f99.md): A bounded one-logical-device check of torch_npu.npu_rms_norm for shape [7, 2048], seed 20260921, and epsilon 1e-6 passed for FP16 and BF16 when the CPU reference first rounded inputs to the tested dtype and then computed in FP32. With initial ASCEND_RT_VISIBLE_DEVICES=8, the script normalized visibility to logical device 0 before importing torch/torch_npu. The result applies only to this small operator check; physical-device mapping, multi-device behavior, serving, throughput, and performance remain unverified.
