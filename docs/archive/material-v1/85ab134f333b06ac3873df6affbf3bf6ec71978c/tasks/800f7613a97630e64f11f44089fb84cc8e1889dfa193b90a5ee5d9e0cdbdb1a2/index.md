---
blocks:
- block_id: 4a6c0655651acc0ed3aec04a9aa080c489c4acd821b3d7f7af41ebe82f5b8473
  indexed: true
  sha256: 9c5e1b58f8ba0222ea04a2ceeb734bddeee725e4267043170be7ed3fe150adb0
  source_range:
    body_end: 4322
    body_start: 0
    published_commit: 9aab5e1e2f1298f992b1b41e403c2b1e0232d12c
    published_path: cases/800f7613a97630e64f11f44089fb84cc8e1889dfa193b90a5ee5d9e0cdbdb1a2.md
  summary: 'Small-scale layout check of torch_npu.npu_rms_norm on one logical Ascend NPU device: x[:, ::2] / weight[::2] non-contiguous path vs contiguous copies for torch.float16 and torch.bfloat16, each compared to a dtype-rounded FP32 CPU reference and the two outputs compared directly; result written to result.json and read back.'
  title: torch_npu.npu_rms_norm non-contiguous view vs contiguous copy check on NPU for float16 and bfloat16
entry:
  conditions:
    python: 3.12.13
    torch: 2.10.0+cpu
    torch_npu: 2.10.0.post2
  domain: vllm-ascend
  entry_id: 800f7613a97630e64f11f44089fb84cc8e1889dfa193b90a5ee5d9e0cdbdb1a2
  kind: experience
  material_digest: 163d7c4a0221f3c8e85ea5d412357a6b3c7f9e60f70a494d6ca75005c3e383a3
  revision: 1994ab9af780f588c5c53587330d84ec0459e16ff0d2f9f9ee07fd71cf2e7b31
  schema: mindie-entry/3
  summary: 'Small-scale layout check of torch_npu.npu_rms_norm on one logical Ascend NPU device: x[:, ::2] / weight[::2] non-contiguous path vs contiguous copies for torch.float16 and torch.bfloat16, each compared to a dtype-rounded FP32 CPU reference and the two outputs compared directly; result written to result.json and read back.'
  title: torch_npu.npu_rms_norm non-contiguous view vs contiguous copy check on NPU for float16 and bfloat16
navigation: 'Small-scale layout check of torch_npu.npu_rms_norm on one logical Ascend NPU device: x[:, ::2] / weight[::2] non-contiguous path vs contiguous copies for torch.float16 and torch.bfloat16, each compared to a dtype-rounded FP32 CPU reference and the two outputs compared directly; result written to result.json and read back.'
schema: mindie-material-task/1
status: complete
task_id: 800f7613a97630e64f11f44089fb84cc8e1889dfa193b90a5ee5d9e0cdbdb1a2
---

# torch_npu.npu_rms_norm non-contiguous view vs contiguous copy check on NPU for float16 and bfloat16

Small-scale layout check of torch_npu.npu_rms_norm on one logical Ascend NPU device: x[:, ::2] / weight[::2] non-contiguous path vs contiguous copies for torch.float16 and torch.bfloat16, each compared to a dtype-rounded FP32 CPU reference and the two outputs compared directly; result written to result.json and read back.

## Materials

- [torch_npu.npu_rms_norm non-contiguous view vs contiguous copy check on NPU for float16 and bfloat16](blocks/4a6c0655651acc0ed3aec04a9aa080c489c4acd821b3d7f7af41ebe82f5b8473.md): Small-scale layout check of torch_npu.npu_rms_norm on one logical Ascend NPU device: x[:, ::2] / weight[::2] non-contiguous path vs contiguous copies for torch.float16 and torch.bfloat16, each compared to a dtype-rounded FP32 CPU reference and the two outputs compared directly; result written to result.json and read back.
