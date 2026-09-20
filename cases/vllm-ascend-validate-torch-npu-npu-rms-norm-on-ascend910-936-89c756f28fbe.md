---
conditions:
  torch_npu_version: 2.10.0.post2
  torch_version: 2.10.0+cpu
domain: vllm-ascend
entry_id: 89c756f28fbec41b9d541f6bf7b095e57b3ea51e7c1a1e2270c395495f43a5c4
kind: experience
producers:
- 7ccbf9f88de1b66a82c572fa4861119c2b85cff21988006296143e34dfed26d7
retirement_reason: ''
revision: 6af93131bf84d7e431724642936397145fd4ab960e7e56a632209c42cca04298
schema: mindie-entry/1
sources: []
status: active
summary: A task-local, content-verified operator check passed for torch_npu.npu_rms_norm on one Ascend910_9362 device with shape [5,1536], epsilon 1e-6, and torch_npu 2.10.0.post2. The CPU reference must use inputs first rounded to the tested dtype. FP16 and BF16 outputs were finite and allclose within separate tolerances. This validates only the small single-device operator case, not vLLM serving, throughput, peak memory, or general device-ID mappings.
title: 'vllm-ascend: validate torch_npu.npu_rms_norm on Ascend910_9362 with dtype-rounded FP16/BF16 references and ASCEND_RT_VISIBLE_DEVICES=0'
---

Problem and background: the task needed a held-out numerical sanity check for `torch_npu.npu_rms_norm` without depending on a model or a historical answer. The script seeded PyTorch with `20260920`, created `x32` with shape `[5,1536]` and `w32` with shape `[1536]`, and tested epsilon `1e-6` on one mapped NPU.

Observed hardware: the verified result names `Ascend910_9362`. The FP16/BF16 dtypes, visibility and device selection, numerical parameters and tolerances below describe this particular experiment, not universal requirements.

Failure or applicability issue: an assistant narrative in the task reported that a process using `ASCEND_RT_VISIBLE_DEVICES=8` saw `torch.npu.device_count()==0`. The supplied script exits with status 2 when the device count is zero, so the operator cannot be tested in that process. The included content-verified result does not contain the `8` run; this failure is therefore reported task evidence rather than independently verified result evidence, and the material does not establish why that visibility choice failed beyond no device being exposed. Do not generalize a physical device number to a required logical device number.

Correction: the verified run recorded `ASCEND_RT_VISIBLE_DEVICES=0`, `device_count=1`, and then selected logical device 0:

```python
torch.npu.set_device(0)
for dtype, tol in [(torch.float16, 0.002), (torch.bfloat16, 0.016)]:
    # Round the actual inputs before transferring them to the NPU.
    x_cpu = x32.to(dtype)
    w_cpu = w32.to(dtype)
    x = x_cpu.npu()
    w = w_cpu.npu()
    y, _ = torch_npu.npu_rms_norm(x, w, epsilon=1e-6)
    torch.npu.synchronize()
    measured = y.float().cpu()

    # Compare against the same rounded inputs, evaluated in FP32.
    xf = x_cpu.float()
    wf = w_cpu.float()
    reference = xf * torch.rsqrt((xf * xf).mean(-1, keepdim=True) + 1e-6) * wf
    delta = (measured - reference).abs()
    assert torch.isfinite(measured).all()
    assert torch.allclose(measured, reference, atol=tol, rtol=tol)
```

Why the reference construction matters: comparing the NPU result with a reference made from the original FP32 tensors would test a different input. The verified script instead casts `x32` and `w32` to FP16 or BF16 first, then converts those actual operator inputs to FP32 for the reference calculation.

Observed results from the content-verified task script and JSON result: `torch.float16` was finite with maximum absolute error `0.0019469261169433594`, `atol=rtol=0.002`, and `allclose=true`; `torch.bfloat16` was finite with maximum absolute error `0.015557289123535156`, `atol=rtol=0.016`, and `allclose=true`. The combined report set `passed=true`. Recorded elapsed times were approximately `0.30268721003085375` seconds for FP16 and `0.0024450598284602165` seconds for BF16, but the script explicitly makes no performance claim. Its conservative explicit-tensor bound was `675840` bytes; runtime contexts and allocator caches were outside that bound.

Limits: the evidence covers only shape `[5,1536]`, one logical device, the recorded software versions, and these two dtypes. It does not verify model execution, vLLM serving, throughput, latency, peak memory, multi-device behavior, other shapes, or the reported `ASCEND_RT_VISIBLE_DEVICES=8` case. The result was read from task-local artifacts with complete content metadata; it was not independently reproduced here.
