---
blocks:
- block_id: 9a58cdf2e7b677d59125629ddae48cbb56e10810097a4d1c4d2f35ba2b8f9df3
  indexed: true
  sha256: e2234485a3c3ccd2d9d1efa4e8dd3f0d613d6ac909824d8a7aaf2e03725e89c9
  source_range:
    body_end: 2792
    body_start: 0
    published_commit: 9aab5e1e2f1298f992b1b41e403c2b1e0232d12c
    published_path: cases/81149b472aa8b84bc887f15e8ce58ea631f58d34b29e0b2c2fb17749904fdbfc.md
  summary: 'Recorded a read-only remote investigation of physical NPU 8: endpoint probe, npu-smi help/version, full device list with HBM usage, mapping, topology, and per-card health. All 16 chips showed VLLMWorker_TP processes using ~57.8 GB each and HBM near capacity. A cgroup/docker attribution query for one observed PID was issued; its output is not present in this increment.'
  title: Read-only npu-smi survey of physical NPU 8 occupancy on a remote Ascend910 host
entry:
  conditions:
    npu-smi: 25.5.3
    python: 3.13.13
    remote-dev plugin: 0.9.5
  domain: vllm-ascend
  entry_id: 81149b472aa8b84bc887f15e8ce58ea631f58d34b29e0b2c2fb17749904fdbfc
  kind: experience
  material_digest: 39effbf083b5348b964effa8c9c94165ce68e27b438a43c896920133619f84aa
  revision: 1b56d55e4357907979386923ffaf1f355ef2f53fdc26f56cd826a8b1ef71a6d2
  schema: mindie-entry/3
  summary: 'Recorded a read-only remote investigation of physical NPU 8: endpoint probe, npu-smi help/version, full device list with HBM usage, mapping, topology, and per-card health. All 16 chips showed VLLMWorker_TP processes using ~57.8 GB each and HBM near capacity. A cgroup/docker attribution query for one observed PID was issued; its output is not present in this increment.'
  title: Read-only npu-smi survey of physical NPU 8 occupancy on a remote Ascend910 host
navigation: 'Recorded a read-only remote investigation of physical NPU 8: endpoint probe, npu-smi help/version, full device list with HBM usage, mapping, topology, and per-card health. All 16 chips showed VLLMWorker_TP processes using ~57.8 GB each and HBM near capacity. A cgroup/docker attribution query for one observed PID was issued; its output is not present in this increment.'
schema: mindie-material-task/1
status: complete
task_id: 81149b472aa8b84bc887f15e8ce58ea631f58d34b29e0b2c2fb17749904fdbfc
---

# Read-only npu-smi survey of physical NPU 8 occupancy on a remote Ascend910 host

Recorded a read-only remote investigation of physical NPU 8: endpoint probe, npu-smi help/version, full device list with HBM usage, mapping, topology, and per-card health. All 16 chips showed VLLMWorker_TP processes using ~57.8 GB each and HBM near capacity. A cgroup/docker attribution query for one observed PID was issued; its output is not present in this increment.

## Materials

- [Read-only npu-smi survey of physical NPU 8 occupancy on a remote Ascend910 host](blocks/9a58cdf2e7b677d59125629ddae48cbb56e10810097a4d1c4d2f35ba2b8f9df3.md): Recorded a read-only remote investigation of physical NPU 8: endpoint probe, npu-smi help/version, full device list with HBM usage, mapping, topology, and per-card health. All 16 chips showed VLLMWorker_TP processes using ~57.8 GB each and HBM near capacity. A cgroup/docker attribution query for one observed PID was issued; its output is not present in this increment.
