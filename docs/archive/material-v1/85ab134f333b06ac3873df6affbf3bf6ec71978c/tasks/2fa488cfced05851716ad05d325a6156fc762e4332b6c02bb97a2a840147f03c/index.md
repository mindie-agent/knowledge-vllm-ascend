---
blocks:
- block_id: fb480c23a3327e920fa2ec0337eb30ca9e8855727fb4f4cf08e4ef93ea11f2e8
  indexed: true
  sha256: 1d37bcea890706686a33810143fc120bb1db8cdc6572ed13925f6df88d09064b
  source_range:
    body_end: 1829
    body_start: 0
    published_commit: 9aab5e1e2f1298f992b1b41e403c2b1e0232d12c
    published_path: cases/2fa488cfced05851716ad05d325a6156fc762e4332b6c02bb97a2a840147f03c.md
  summary: 'Conversation excerpt: 当前搜索返回了转述该 ID 的对话摘要和带有目标 ID 前缀的其他命中；摘要不能替代原始正文。我会再用完整条目 ID 做一次定向检索，然后只对命中的引用调用正文读取。

    完整 ID 定向搜索后，工具确实返回了带同一条目 ID 前缀的 `origin=feed` 命中；我也读到了正文。为核准“公开来源”和版本是否一致，我再按查询直接返回的 pinned ref 读取一次该命中版本。

    已通过 MindIE Agent 启用当前任务，状态为 `capture=bound`、`experience=capture-ready`。本轮只做了知识检索和正文读取，没有执行远端任务或运行 NPU。

    **来源与引用：**知识库返回了 `vllm-ascend` 域的公开 feed 命中（`origin=feed`）。读取到的条目 ID 与指定 ID 完全一致，类型为 `experience`，`withdrawn=false`；正文共 4,988 字符，没有后续页。MindIE 引用：

    `mindie://vllm-ascend/5cdccec11137a3387aba7f7b1e27c9f2b08534b619cc2dc838c622960a4cc8e9@5dd3c315e730da9ae3145066acf1cda36150e4aa8f22a062c8bbca281bb46222`

    **正文记录的推理与 token：**Qwen3-0.6B 两条生成分别输入 6、5 token，输出各 32 token；合计 **11 输入、64 输出 token**。两条生成共 **1.201 秒**，模型加载 20.399 秒。两条输出都达到 32-token 上限。知识库未提供这段对话正文自身的模型 token 计数。

    **NPU 证据与边界：**正文记录 Ascend 插件已激活，EngineCore 使用 `device_config=npu`、`visible_n'
  title: 当前搜索返回了转述该 ID 的对话摘要和带有目标 ID 前缀的其他命中；摘要不能替代原始正文。我会再用完整条目 ID 做一次定向检索，然后只对命中的引用调用正文读取。
entry:
  conditions: {}
  domain: vllm-ascend
  entry_id: 2fa488cfced05851716ad05d325a6156fc762e4332b6c02bb97a2a840147f03c
  kind: experience
  material_digest: 45da4166ab11fda758d0c4b98bb5ce7de5cf4b3cf680e01581f6d8a6df840823
  revision: 9d6a5adb39c0459b1952b7176a086673e7d49154dae5da3a085804888eceecd8
  schema: mindie-entry/3
  summary: 'Conversation excerpt: 当前搜索返回了转述该 ID 的对话摘要和带有目标 ID 前缀的其他命中；摘要不能替代原始正文。我会再用完整条目 ID 做一次定向检索，然后只对命中的引用调用正文读取。

    完整 ID 定向搜索后，工具确实返回了带同一条目 ID 前缀的 `origin=feed` 命中；我也读到了正文。为核准“公开来源”和版本是否一致，我再按查询直接返回的 pinned ref 读取一次该命中版本。

    已通过 MindIE Agent 启用当前任务，状态为 `capture=bound`、`experience=capture-ready`。本轮只做了知识检索和正文读取，没有执行远端任务或运行 NPU。

    **来源与引用：**知识库返回了 `vllm-ascend` 域的公开 feed 命中（`origin=feed`）。读取到的条目 ID 与指定 ID 完全一致，类型为 `experience`，`withdrawn=false`；正文共 4,988 字符，没有后续页。MindIE 引用：

    `mindie://vllm-ascend/5cdccec11137a3387aba7f7b1e27c9f2b08534b619cc2dc838c622960a4cc8e9@5dd3c315e730da9ae3145066acf1cda36150e4aa8f22a062c8bbca281bb46222`

    **正文记录的推理与 token：**Qwen3-0.6B 两条生成分别输入 6、5 token，输出各 32 token；合计 **11 输入、64 输出 token**。两条生成共 **1.201 秒**，模型加载 20.399 秒。两条输出都达到 32-token 上限。知识库未提供这段对话正文自身的模型 token 计数。

    **NPU 证据与边界：**正文记录 Ascend 插件已激活，EngineCore 使用 `device_config=npu`、`visible_n'
  title: 当前搜索返回了转述该 ID 的对话摘要和带有目标 ID 前缀的其他命中；摘要不能替代原始正文。我会再用完整条目 ID 做一次定向检索，然后只对命中的引用调用正文读取。
navigation: 'Conversation excerpt: 当前搜索返回了转述该 ID 的对话摘要和带有目标 ID 前缀的其他命中；摘要不能替代原始正文。我会再用完整条目 ID 做一次定向检索，然后只对命中的引用调用正文读取。

  完整 ID 定向搜索后，工具确实返回了带同一条目 ID 前缀的 `origin=feed` 命中；我也读到了正文。为核准“公开来源”和版本是否一致，我再按查询直接返回的 pinned ref 读取一次该命中版本。

  已通过 MindIE Agent 启用当前任务，状态为 `capture=bound`、`experience=capture-ready`。本轮只做了知识检索和正文读取，没有执行远端任务或运行 NPU。

  **来源与引用：**知识库返回了 `vllm-ascend` 域的公开 feed 命中（`origin=feed`）。读取到的条目 ID 与指定 ID 完全一致，类型为 `experience`，`withdrawn=false`；正文共 4,988 字符，没有后续页。MindIE 引用：

  `mindie://vllm-ascend/5cdccec11137a3387aba7f7b1e27c9f2b08534b619cc2dc838c622960a4cc8e9@5dd3c315e730da9ae3145066acf1cda36150e4aa8f22a062c8bbca281bb46222`

  **正文记录的推理与 token：**Qwen3-0.6B 两条生成分别输入 6、5 token，输出各 32 token；合计 **11 输入、64 输出 token**。两条生成共 **1.201 秒**，模型加载 20.399 秒。两条输出都达到 32-token 上限。知识库未提供这段对话正文自身的模型 token 计数。

  **NPU 证据与边界：**正文记录 Ascend 插件已激活，EngineCore 使用 `device_config=npu`、`visible_n'
schema: mindie-material-task/1
status: complete
task_id: 2fa488cfced05851716ad05d325a6156fc762e4332b6c02bb97a2a840147f03c
---

# 当前搜索返回了转述该 ID 的对话摘要和带有目标 ID 前缀的其他命中；摘要不能替代原始正文。我会再用完整条目 ID 做一次定向检索，然后只对命中的引用调用正文读取。

Conversation excerpt: 当前搜索返回了转述该 ID 的对话摘要和带有目标 ID 前缀的其他命中；摘要不能替代原始正文。我会再用完整条目 ID 做一次定向检索，然后只对命中的引用调用正文读取。
完整 ID 定向搜索后，工具确实返回了带同一条目 ID 前缀的 `origin=feed` 命中；我也读到了正文。为核准“公开来源”和版本是否一致，我再按查询直接返回的 pinned ref 读取一次该命中版本。
已通过 MindIE Agent 启用当前任务，状态为 `capture=bound`、`experience=capture-ready`。本轮只做了知识检索和正文读取，没有执行远端任务或运行 NPU。
**来源与引用：**知识库返回了 `vllm-ascend` 域的公开 feed 命中（`origin=feed`）。读取到的条目 ID 与指定 ID 完全一致，类型为 `experience`，`withdrawn=false`；正文共 4,988 字符，没有后续页。MindIE 引用：
`mindie://vllm-ascend/5cdccec11137a3387aba7f7b1e27c9f2b08534b619cc2dc838c622960a4cc8e9@5dd3c315e730da9ae3145066acf1cda36150e4aa8f22a062c8bbca281bb46222`
**正文记录的推理与 token：**Qwen3-0.6B 两条生成分别输入 6、5 token，输出各 32 token；合计 **11 输入、64 输出 token**。两条生成共 **1.201 秒**，模型加载 20.399 秒。两条输出都达到 32-token 上限。知识库未提供这段对话正文自身的模型 token 计数。
**NPU 证据与边界：**正文记录 Ascend 插件已激活，EngineCore 使用 `device_config=npu`、`visible_n

## Materials

- [当前搜索返回了转述该 ID 的对话摘要和带有目标 ID 前缀的其他命中；摘要不能替代原始正文。我会再用完整条目 ID 做一次定向检索，然后只对命中的引用调用正文读取。](blocks/fb480c23a3327e920fa2ec0337eb30ca9e8855727fb4f4cf08e4ef93ea11f2e8.md): Conversation excerpt: 当前搜索返回了转述该 ID 的对话摘要和带有目标 ID 前缀的其他命中；摘要不能替代原始正文。我会再用完整条目 ID 做一次定向检索，然后只对命中的引用调用正文读取。
完整 ID 定向搜索后，工具确实返回了带同一条目 ID 前缀的 `origin=feed` 命中；我也读到了正文。为核准“公开来源”和版本是否一致，我再按查询直接返回的 pinned ref 读取一次该命中版本。
已通过 MindIE Agent 启用当前任务，状态为 `capture=bound`、`experience=capture-ready`。本轮只做了知识检索和正文读取，没有执行远端任务或运行 NPU。
**来源与引用：**知识库返回了 `vllm-ascend` 域的公开 feed 命中（`origin=feed`）。读取到的条目 ID 与指定 ID 完全一致，类型为 `experience`，`withdrawn=false`；正文共 4,988 字符，没有后续页。MindIE 引用：
`mindie://vllm-ascend/5cdccec11137a3387aba7f7b1e27c9f2b08534b619cc2dc838c622960a4cc8e9@5dd3c315e730da9ae3145066acf1cda36150e4aa8f22a062c8bbca281bb46222`
**正文记录的推理与 token：**Qwen3-0.6B 两条生成分别输入 6、5 token，输出各 32 token；合计 **11 输入、64 输出 token**。两条生成共 **1.201 秒**，模型加载 20.399 秒。两条输出都达到 32-token 上限。知识库未提供这段对话正文自身的模型 token 计数。
**NPU 证据与边界：**正文记录 Ascend 插件已激活，EngineCore 使用 `device_config=npu`、`visible_n
