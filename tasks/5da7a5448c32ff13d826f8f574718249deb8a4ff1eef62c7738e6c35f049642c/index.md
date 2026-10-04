---
blocks:
- block_id: 801fd5c698c906e5753669323778e3d90afe43560a50f6cca4a73276dd396d2e
  indexed: true
  sha256: 0c2ac40bbf4332de0ed89411d2fd8563957dc9a4522867dee6df1908e7295480
  source_range:
    body_end: 1908
    body_start: 0
    published_commit: 9aab5e1e2f1298f992b1b41e403c2b1e0232d12c
    published_path: cases/5da7a5448c32ff13d826f8f574718249deb8a4ff1eef62c7738e6c35f049642c.md
  summary: 'Conversation excerpt: 当前任务已在最新的活动缓存版本中绑定，经验状态为 `capture-ready`，社区分享配置也显示为 public。现在我先用指定条目 ID 检索；只有命中后才会读取该条目的正文和证据。

    指定 ID 的纯 ID 查询没有结果；加入模型与耗时关键词后，知识库返回了同一 ID 前缀的 feed 命中（引用以该 ID 开头）。我会只打开这个命中，核实实际正文；不会采用检索摘要里夹带的其他条目内容。

    **命中情况：**用完整条目 ID 单独检索时，BM25 没有返回结果；加入 `Qwen3-0.6B` 和 `1.201 秒` 后，返回并读取了目标 ID 的完整正文。来源为 `vllm-ascend` 域的 `feed`，类型是 `experience`，`withdrawn=false`。正文共 4,655 字符，`next_offset=null`，没有后续页。

    **MindIE 引用：**

    `mindie://vllm-ascend/fb05d2bf865a620a08481eb46d36d603ef0599de5d1397ec35f12a18cb3ff761@35208d3fa1363eeb518547def70099926da65788d784487d2731cb8a21c108bb`

    **Token 数：**正文记录两条请求各生成 32 tokens，共 **64 个输出 tokens**；没有提供 prompt 输入 token 数。4,655 是正文字符数，不是模型 token 数。我没有用其他条目的 token 数补充。

    **NPU 证据：**正文记载 vLLM 选择 `NPUPlatform`；推理期间 `npu-smi` 在物理 Phy-ID 1 / NPU 0 Chip 1 观察到任务进程及约 13.4 GB 显存占用，结束后进程消失、显存回到约 2,899 MiB 基线。最终退出码为 0。首次启动因诊断调用初始'
  title: 当前任务已在最新的活动缓存版本中绑定，经验状态为 `capture-ready`，社区分享配置也显示为 public。现在我先用指定条目 ID 检索；只有命中后才会读取该条目的正文和证据。
entry:
  conditions: {}
  domain: vllm-ascend
  entry_id: 5da7a5448c32ff13d826f8f574718249deb8a4ff1eef62c7738e6c35f049642c
  kind: experience
  material_digest: 822dcdc01986403de3e858efe83171dcb81f07dfe42743e7ff6261a87f3f9d5a
  revision: 14408b977aefddd8cdea1c6a91c47b871a16f3e5eab3d075d02f24ec2a62cc18
  schema: mindie-entry/3
  summary: 'Conversation excerpt: 当前任务已在最新的活动缓存版本中绑定，经验状态为 `capture-ready`，社区分享配置也显示为 public。现在我先用指定条目 ID 检索；只有命中后才会读取该条目的正文和证据。

    指定 ID 的纯 ID 查询没有结果；加入模型与耗时关键词后，知识库返回了同一 ID 前缀的 feed 命中（引用以该 ID 开头）。我会只打开这个命中，核实实际正文；不会采用检索摘要里夹带的其他条目内容。

    **命中情况：**用完整条目 ID 单独检索时，BM25 没有返回结果；加入 `Qwen3-0.6B` 和 `1.201 秒` 后，返回并读取了目标 ID 的完整正文。来源为 `vllm-ascend` 域的 `feed`，类型是 `experience`，`withdrawn=false`。正文共 4,655 字符，`next_offset=null`，没有后续页。

    **MindIE 引用：**

    `mindie://vllm-ascend/fb05d2bf865a620a08481eb46d36d603ef0599de5d1397ec35f12a18cb3ff761@35208d3fa1363eeb518547def70099926da65788d784487d2731cb8a21c108bb`

    **Token 数：**正文记录两条请求各生成 32 tokens，共 **64 个输出 tokens**；没有提供 prompt 输入 token 数。4,655 是正文字符数，不是模型 token 数。我没有用其他条目的 token 数补充。

    **NPU 证据：**正文记载 vLLM 选择 `NPUPlatform`；推理期间 `npu-smi` 在物理 Phy-ID 1 / NPU 0 Chip 1 观察到任务进程及约 13.4 GB 显存占用，结束后进程消失、显存回到约 2,899 MiB 基线。最终退出码为 0。首次启动因诊断调用初始'
  title: 当前任务已在最新的活动缓存版本中绑定，经验状态为 `capture-ready`，社区分享配置也显示为 public。现在我先用指定条目 ID 检索；只有命中后才会读取该条目的正文和证据。
navigation: 'Conversation excerpt: 当前任务已在最新的活动缓存版本中绑定，经验状态为 `capture-ready`，社区分享配置也显示为 public。现在我先用指定条目 ID 检索；只有命中后才会读取该条目的正文和证据。

  指定 ID 的纯 ID 查询没有结果；加入模型与耗时关键词后，知识库返回了同一 ID 前缀的 feed 命中（引用以该 ID 开头）。我会只打开这个命中，核实实际正文；不会采用检索摘要里夹带的其他条目内容。

  **命中情况：**用完整条目 ID 单独检索时，BM25 没有返回结果；加入 `Qwen3-0.6B` 和 `1.201 秒` 后，返回并读取了目标 ID 的完整正文。来源为 `vllm-ascend` 域的 `feed`，类型是 `experience`，`withdrawn=false`。正文共 4,655 字符，`next_offset=null`，没有后续页。

  **MindIE 引用：**

  `mindie://vllm-ascend/fb05d2bf865a620a08481eb46d36d603ef0599de5d1397ec35f12a18cb3ff761@35208d3fa1363eeb518547def70099926da65788d784487d2731cb8a21c108bb`

  **Token 数：**正文记录两条请求各生成 32 tokens，共 **64 个输出 tokens**；没有提供 prompt 输入 token 数。4,655 是正文字符数，不是模型 token 数。我没有用其他条目的 token 数补充。

  **NPU 证据：**正文记载 vLLM 选择 `NPUPlatform`；推理期间 `npu-smi` 在物理 Phy-ID 1 / NPU 0 Chip 1 观察到任务进程及约 13.4 GB 显存占用，结束后进程消失、显存回到约 2,899 MiB 基线。最终退出码为 0。首次启动因诊断调用初始'
schema: mindie-material-task/1
status: complete
task_id: 5da7a5448c32ff13d826f8f574718249deb8a4ff1eef62c7738e6c35f049642c
---

# 当前任务已在最新的活动缓存版本中绑定，经验状态为 `capture-ready`，社区分享配置也显示为 public。现在我先用指定条目 ID 检索；只有命中后才会读取该条目的正文和证据。

Conversation excerpt: 当前任务已在最新的活动缓存版本中绑定，经验状态为 `capture-ready`，社区分享配置也显示为 public。现在我先用指定条目 ID 检索；只有命中后才会读取该条目的正文和证据。
指定 ID 的纯 ID 查询没有结果；加入模型与耗时关键词后，知识库返回了同一 ID 前缀的 feed 命中（引用以该 ID 开头）。我会只打开这个命中，核实实际正文；不会采用检索摘要里夹带的其他条目内容。
**命中情况：**用完整条目 ID 单独检索时，BM25 没有返回结果；加入 `Qwen3-0.6B` 和 `1.201 秒` 后，返回并读取了目标 ID 的完整正文。来源为 `vllm-ascend` 域的 `feed`，类型是 `experience`，`withdrawn=false`。正文共 4,655 字符，`next_offset=null`，没有后续页。
**MindIE 引用：**
`mindie://vllm-ascend/fb05d2bf865a620a08481eb46d36d603ef0599de5d1397ec35f12a18cb3ff761@35208d3fa1363eeb518547def70099926da65788d784487d2731cb8a21c108bb`
**Token 数：**正文记录两条请求各生成 32 tokens，共 **64 个输出 tokens**；没有提供 prompt 输入 token 数。4,655 是正文字符数，不是模型 token 数。我没有用其他条目的 token 数补充。
**NPU 证据：**正文记载 vLLM 选择 `NPUPlatform`；推理期间 `npu-smi` 在物理 Phy-ID 1 / NPU 0 Chip 1 观察到任务进程及约 13.4 GB 显存占用，结束后进程消失、显存回到约 2,899 MiB 基线。最终退出码为 0。首次启动因诊断调用初始

## Materials

- [当前任务已在最新的活动缓存版本中绑定，经验状态为 `capture-ready`，社区分享配置也显示为 public。现在我先用指定条目 ID 检索；只有命中后才会读取该条目的正文和证据。](blocks/801fd5c698c906e5753669323778e3d90afe43560a50f6cca4a73276dd396d2e.md): Conversation excerpt: 当前任务已在最新的活动缓存版本中绑定，经验状态为 `capture-ready`，社区分享配置也显示为 public。现在我先用指定条目 ID 检索；只有命中后才会读取该条目的正文和证据。
指定 ID 的纯 ID 查询没有结果；加入模型与耗时关键词后，知识库返回了同一 ID 前缀的 feed 命中（引用以该 ID 开头）。我会只打开这个命中，核实实际正文；不会采用检索摘要里夹带的其他条目内容。
**命中情况：**用完整条目 ID 单独检索时，BM25 没有返回结果；加入 `Qwen3-0.6B` 和 `1.201 秒` 后，返回并读取了目标 ID 的完整正文。来源为 `vllm-ascend` 域的 `feed`，类型是 `experience`，`withdrawn=false`。正文共 4,655 字符，`next_offset=null`，没有后续页。
**MindIE 引用：**
`mindie://vllm-ascend/fb05d2bf865a620a08481eb46d36d603ef0599de5d1397ec35f12a18cb3ff761@35208d3fa1363eeb518547def70099926da65788d784487d2731cb8a21c108bb`
**Token 数：**正文记录两条请求各生成 32 tokens，共 **64 个输出 tokens**；没有提供 prompt 输入 token 数。4,655 是正文字符数，不是模型 token 数。我没有用其他条目的 token 数补充。
**NPU 证据：**正文记载 vLLM 选择 `NPUPlatform`；推理期间 `npu-smi` 在物理 Phy-ID 1 / NPU 0 Chip 1 观察到任务进程及约 13.4 GB 显存占用，结束后进程消失、显存回到约 2,899 MiB 基线。最终退出码为 0。首次启动因诊断调用初始
