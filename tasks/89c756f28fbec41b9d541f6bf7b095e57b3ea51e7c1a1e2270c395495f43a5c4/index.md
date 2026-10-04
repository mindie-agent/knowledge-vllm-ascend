---
blocks:
- block_id: 993f00f5d9a4a1ab93edff427d203140ae95e335be8086122cfaac42cef1928b
  indexed: true
  sha256: 19280529a14ef3f44ef871e7959e219421fc47ec77f2a3060ea107443bb632ba
  source_range:
    body_end: 3395
    body_start: 0
    published_commit: 9aab5e1e2f1298f992b1b41e403c2b1e0232d12c
    published_path: cases/89c756f28fbec41b9d541f6bf7b095e57b3ea51e7c1a1e2270c395495f43a5c4.md
  summary: A task-local, content-verified operator check passed for torch_npu.npu_rms_norm on one Ascend910_9362 device with shape [5,1536], epsilon 1e-6, and torch_npu 2.10.0.post2. The CPU reference must use inputs first rounded to the tested dtype. FP16 and BF16 outputs were finite and allclose within separate tolerances. This validates only the small single-device operator case, not vLLM serving, throughput, peak memory, or general device-ID mappings.
  title: 'vllm-ascend: validate torch_npu.npu_rms_norm on Ascend910_9362 with dtype-rounded FP16/BF16 references and ASCEND_RT_VISIBLE_DEVICES=0'
entry:
  conditions:
    torch_npu_version: 2.10.0.post2
    torch_version: 2.10.0+cpu
  domain: vllm-ascend
  entry_id: 89c756f28fbec41b9d541f6bf7b095e57b3ea51e7c1a1e2270c395495f43a5c4
  kind: experience
  material_digest: ae0efd83960e65c92e382d5b82fd8fe8c72ccf3ea30ba02b8ed9786241b4b2f0
  revision: 4496f43833416a927afdeecc755190dc5401f89fa79a6a67b8e8923d54eb62ea
  schema: mindie-entry/3
  summary: A task-local, content-verified operator check passed for torch_npu.npu_rms_norm on one Ascend910_9362 device with shape [5,1536], epsilon 1e-6, and torch_npu 2.10.0.post2. The CPU reference must use inputs first rounded to the tested dtype. FP16 and BF16 outputs were finite and allclose within separate tolerances. This validates only the small single-device operator case, not vLLM serving, throughput, peak memory, or general device-ID mappings.
  title: 'vllm-ascend: validate torch_npu.npu_rms_norm on Ascend910_9362 with dtype-rounded FP16/BF16 references and ASCEND_RT_VISIBLE_DEVICES=0'
navigation: A task-local, content-verified operator check passed for torch_npu.npu_rms_norm on one Ascend910_9362 device with shape [5,1536], epsilon 1e-6, and torch_npu 2.10.0.post2. The CPU reference must use inputs first rounded to the tested dtype. FP16 and BF16 outputs were finite and allclose within separate tolerances. This validates only the small single-device operator case, not vLLM serving, throughput, peak memory, or general device-ID mappings.
schema: mindie-material-task/1
status: complete
task_id: 89c756f28fbec41b9d541f6bf7b095e57b3ea51e7c1a1e2270c395495f43a5c4
---

# vllm-ascend: validate torch_npu.npu_rms_norm on Ascend910_9362 with dtype-rounded FP16/BF16 references and ASCEND_RT_VISIBLE_DEVICES=0

A task-local, content-verified operator check passed for torch_npu.npu_rms_norm on one Ascend910_9362 device with shape [5,1536], epsilon 1e-6, and torch_npu 2.10.0.post2. The CPU reference must use inputs first rounded to the tested dtype. FP16 and BF16 outputs were finite and allclose within separate tolerances. This validates only the small single-device operator case, not vLLM serving, throughput, peak memory, or general device-ID mappings.

## Materials

- [vllm-ascend: validate torch_npu.npu_rms_norm on Ascend910_9362 with dtype-rounded FP16/BF16 references and ASCEND_RT_VISIBLE_DEVICES=0](blocks/993f00f5d9a4a1ab93edff427d203140ae95e335be8086122cfaac42cef1928b.md): A task-local, content-verified operator check passed for torch_npu.npu_rms_norm on one Ascend910_9362 device with shape [5,1536], epsilon 1e-6, and torch_npu 2.10.0.post2. The CPU reference must use inputs first rounded to the tested dtype. FP16 and BF16 outputs were finite and allclose within separate tolerances. This validates only the small single-device operator case, not vLLM serving, throughput, peak memory, or general device-ID mappings.
