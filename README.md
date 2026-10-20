<h1 align="center">MindIE Knowledge · vLLM-Ascend</h1>

<p align="center">
  <strong>面向 Ascend NPU 与 <code>vLLM-Ascend</code> 的条件化工程知识库</strong><br>
  <sub>Part of <a href="https://github.com/mindie-agent/mindie-agent">MindIE Agent</a> · 由 <a href="https://github.com/mindie-agent/knowledge"><code>mindie-knowledge</code></a> 提供本地检索、精确全文读取与修订反馈</sub>
</p>

<p align="center">
  <a href="https://github.com/mindie-agent/mindie-agent"><img src="https://img.shields.io/badge/MindIE%20Agent-Ecosystem-7045f5.svg" alt="MindIE Agent Ecosystem"></a>
  <a href="https://github.com/mindie-agent/knowledge"><img src="https://img.shields.io/badge/runtime-mindie--knowledge-0284c7.svg" alt="mindie-knowledge"></a>
  <a href="publication-contract.json"><img src="https://img.shields.io/badge/domain-vllm--ascend-0f766e.svg" alt="Domain vllm-ascend"></a>
  <a href="AGENTS.md"><img src="https://img.shields.io/badge/schema-mindie--graded--unit%2F1-0f766e.svg" alt="Graded unit schema"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/corpus-CC--BY--4.0-0d9488.svg" alt="License CC-BY-4.0"></a>
</p>

<p align="center">
  <strong>中文</strong> · <a href="docs/README.en.md">English</a> · <a href="#-接入">接入</a> · <a href="#-为什么使用知识单元">为什么</a> · <a href="#-检索与全文读取">检索</a> · <a href="#-当前单元">当前单元</a> · <a href="docs/package-format.md">📐 格式规范</a> · <a href="CONTRIBUTING.md">🤝 参与贡献</a>
</p>

---

## 这是什么

本仓库的 `graded-v2` 分支承载 MindIE Agent `0.0.3` 的 `vllm-ascend` 领域分级知识源。它保存的是**分级知识单元**（graded unit，`mindie-graded-unit/1`）：一条可复用的判断、步骤或代码变换，连同它的来源、适用条件、证据强度和修订号。

知识单元不是聊天记录摘要，也不是“放之四海皆准”的最佳实践。每条单元都明确回答两个问题：

- 在哪些**可核验的前提**下，这条结论成立；
- 它依据什么来源和证据，强度到什么程度。

查询时，运行时把任务实际观测到的事实与单元声明的条件逐项比较。条件不满足、缺失或未知，都不会被当作“差不多适用”。

> `graded-v2` 分支使用 `mindie-publication-contract/2` 和 `mindie-graded-unit/1`；`main` 保持旧的 material-v1 语料和旧 contract 字节，供 0.0.2 消费者使用，不是 graded 树。实时单元清单以 [`index/routing-manifest.json`](index/routing-manifest.json) 为准；本文中的表格只列出本次检查时清单中的单元，不是固定计数。

---

## ⚡ 接入

MindIE Agent 支持 **Codex、Claude Code、Kimi、DSH** 四种宿主，各自通过原生插件安装。请按主仓库的安装指南选择对应宿主：

**<https://github.com/mindie-agent/mindie-agent>**

插件就绪后会连接 [`mindie-knowledge`](https://github.com/mindie-agent/knowledge) 运行时，同步本领域内容并建立本地索引。只读的 `knowledge_query` / `knowledge_explain` 不要求开启社区共享；是否允许采集和贡献由用户显式决定。

本仓库不再复制各宿主的安装脚本或本机路径。宿主安装、插件配置与升级以主仓库说明为准。0.0.3 插件显式选择 `graded-v2` 分支，并绑定本领域 `publication-contract.json` 的精确摘要；贡献 PR 以 `graded-v2` 为 base。

---

## ✨ 为什么使用知识单元

| 只看摘要 | 本仓库的知识单元 |
| --- | --- |
| 结论被压缩成语境不明的“经验”，版本、前提和失败分支容易丢失 | 单元正文保留具体条件、步骤、反例与边界；来源通过 `provenance` 绑定 |
| 检索命中后不知道是否适用于当前环境 | 查询先比较任务观测条件；不匹配或未知时明确排除，并给出原因 |
| 旧结论被静默覆盖 | 修订号、内容摘要和精确引用使旧观测可追溯；被替换的引用会显式失败 |
| 只能看到片段，无法核对全文 | `knowledge_query` 返回带摘要的候选；`knowledge_explain` 按精确摘要读取完整正文 |

知识单元可以来自已审核的公开材料、源码、规范或实验记录。允许使用模型辅助提炼，但**模型输出只是草案**：它必须保留来源和证据边界，经过确定性格式/隐私检查和语义审核，才能通过已审核变更进入当前语料。不能把失败实验写成 `reproduced_empirical`，不能编造硬件、版本或性能数据，也不能用模型改写替代来源。

原先的 `material-v1` 任务材料不在 `graded-v2` 分支内；它们保留在 `main` 和不可变历史提交 85ab 中，见[旧材料与历史](#-旧材料与历史)。旧材料是证据，不参与当前单元检索，也不会被重新标记为 graded unit。

---

## 🔎 检索与全文读取

一次典型使用包括：先查询，再按精确引用读全文，可选地记录反馈；跨领域时使用显式路由。

### 1. 用任务实际观测到的事实查询

`knowledge_query` 接受查询文本、可选的条件映射和条数上限：

```python
knowledge_query(
    query="custom op import registration NZ selection",
    conditions={
        "repository": "vllm-project/vllm-ascend",
        "revision": "175ef4d49c2072f9233619636f0309f4f88b4e2c",
    },
)
```

`conditions` 只放**当前任务实际观测到**的软件版本、源码提交、硬件、输入形状等事实。不要把单元自己声明的条件抄进查询来“保证命中”，也不要把 gold patch、oracle 判定、heldout 答案或修复后的结论当作查询事实。

缺失的条件、`unknown` 值和无法比较的版本都会得到 `condition_unknown`，默认不进入匹配结果。这是正确行为，不是检索失败。所有声明条件都成立时才是 `matched`；任一条件明确不符则是 `mismatch`。

### 2. 按返回的引用读取完整正文

查询结果会给出带内容摘要的 `ref`、稳定的 `unit_ref` 和独立的 `feedback_ref`。`knowledge_explain` 读取该精确修订的完整正文、frontmatter、继承闭包和适用性说明：

```python
knowledge_explain(
    ref="mindie://vllm-ascend/units/unit-vllm-ascend-custom-op-gates-import-retry-and-nz@025b36835f18326dda7ef09ea380c8a4da9fd8d34eb3f39841010f302e3501ba",
)
```

引用中的 `@` 后是**裸 64 位内容摘要**（`content_sha256`），不是修订号。单元被语义修改后，旧引用会明确报告已被替换，不会静默返回新内容；仅移动路径、不改变内容的单元保留原摘要和修订号。

读取成功只说明你拿到了这段内容，不说明它自动适用于当前任务。仍需逐项比较条件。

### 3. 记录本地有用性反馈

在实际任务中验证某条单元后，可以可选地提交 `up` / `down`：

```python
knowledge_feedback(
    ref="mindie://vllm-ascend/units/unit-vllm-ascend-custom-op-gates-import-retry-and-nz@025b36835f18326dda7ef09ea380c8a4da9fd8d34eb3f39841010f302e3501ba",
    rating="up",
    reason="Recorded after reproducing the import gate difference in the observed task.",
)
```

- 反馈绑定**实际被返回过的精确内容修订**；没有观测过的修订不能被评价。
- `reason` 可选，单行，最多 1000 字符；本地私密反馈默认不外发。
- 反馈是描述性信号，不是正确性证明、置信度或自动晋升依据。没有反馈不代表内容无效，也不代表无人使用。
- 公开反馈文件使用 `mindie-graded-feedback/1`，只聚合与当前内容摘要一致的投票，旧内容摘要的投票保持独立。

### 4. 显式领域路由

跨领域知识通过显式引用访问，例如 `mindie://operators/units/<unit-id>@<sha256>`。引用中的域名决定由哪个已配置领域引擎处理；未知或未配置的领域会明确失败，不会静默回退到默认领域。

运行时可以配置任意领域。`0.0.3` 的发布范围是 `vllm-ascend` 和 `operators` 两个领域，但这不是把运行时写死为两个领域。本仓库不复制 `operators` 单元，也不把两个领域的同主题内容强行合并。

---

## 📚 当前单元

[`index/routing-manifest.json`](index/routing-manifest.json) 是唯一的实时清单，记录当前单元的稳定 ID、路径、层级、证据强度、条件、内容摘要、继承闭包和 token 估算。下表列出本次检查时清单中的单元；清单仍是权威，新增单元只有在其 Markdown 文件和重建后的清单同时存在时，才成为当前内容。

| 单元 | 层级 / 类别 | 证据强度 | 固定引用 |
| --- | --- | --- | --- |
| [`unit-vllm-ascend-custom-op-gates-import-retry-and-nz`](integration/custom-ops/unit-vllm-ascend-custom-op-gates-import-retry-and-nz.md) | L2 / `sop` | `single_observation` | `mindie://vllm-ascend/units/unit-vllm-ascend-custom-op-gates-import-retry-and-nz@025b36835f18326dda7ef09ea380c8a4da9fd8d34eb3f39841010f302e3501ba` |
| [`unit-vllm-ascend-idle-dp-gdn-state`](vllm-ascend/runtime/idle-dp-gdn-state/unit-vllm-ascend-idle-dp-gdn-state.md) | L2 / `sop` | `hypothesis` | `mindie://vllm-ascend/units/unit-vllm-ascend-idle-dp-gdn-state@72f2bfadc7899c951e711b81719ce6b5854cbfe6dd343b00bc35d75b221deb00` |
| [`unit-vllm-ascend-qwen35-gated-deltanet-decode-adaptation`](vllm-ascend/runtime/qwen35-gated-deltanet-decode/unit-vllm-ascend-qwen35-gated-deltanet-decode-adaptation.md) | L2 / `sop` | `single_observation` | `mindie://vllm-ascend/units/unit-vllm-ascend-qwen35-gated-deltanet-decode-adaptation@33661ebd978f03c8355b1728e082fb8937bb5bf245bea38557d069def80ef14c` |

这些单元都是源码范围的 L2 SOP，不是硬件性能结论。custom-op 单元绑定 `vllm-project/vllm-ascend` 的源码提交 `175ef4d49c2072f9233619636f0309f4f88b4e2c`；idle-DP GDN 单元是 `hypothesis`，条件绑定 base commit `c3394206bdd00b132580ae439555b4f1a70d5bf6`、模型和状态上下文；Qwen3.5 单元是历史 PR7109 的 `single_observation`，条件绑定模型和源码 revision。不要把 `hypothesis` 或单次观察升级为复现测量；每个单元的精确条件以其 frontmatter 和正文为准。

---

## 🗂️ 仓库结构

```text
knowledge-vllm-ascend/
├── publication-contract.json           # contract/2、格式版本与 validator pin
├── AGENTS.md                           # 维护者与 AI Agent 的仓库边界
├── CONTRIBUTING.md                     # 贡献与审核流程
├── index/
│   ├── routing-manifest.json           # 由单元派生的实时路由清单
│   └── shard-topology.json             # 由单元派生的分片拓扑
├── <domain_path>/
│   └── <unit_id>.md                    # 单元正文 + frontmatter，路径与稳定 ID 绑定
├── feedback/
│   └── votes-<id>.json                 # mindie-graded-feedback/1 公开反馈
└── docs/
    ├── README.en.md
    ├── CONTRIBUTING.en.md
    ├── package-format.md
    ├── package-format.en.md
    └── evidence/                       # 来源与审核记录
```

单元路径必须是 `<domain_path>/<unit_id>.md`。`domain_path` 是规范化的仓库相对目录，用于路由和分片，不参与稳定 ID；`unit_id` 以 `unit-` 开头，由小写单词和连字符组成。`index/` 中的两个 JSON 是派生数据，由运行时重建，不应手工编辑。本分支只发布 contract/2 内容；旧 `material-v1` 的本地 archive 副本不在 `graded-v2` 内。

格式字段、摘要计算、条件匹配、继承、反馈和已审核变更的完整规则见[格式规范](docs/package-format.md)。

---

## 🗃️ 旧材料与历史

旧的 `material-v1` 任务语料不在 `graded-v2` 分支内。它保留在 [`main`](https://github.com/mindie-agent/knowledge-vllm-ascend/tree/main)（供 0.0.2 消费者继续使用）和不可变历史提交 [`85ab134f333b06ac3873df6affbf3bf6ec71978c`](https://github.com/mindie-agent/knowledge-vllm-ascend/tree/85ab134f333b06ac3873df6affbf3bf6ec71978c) 中。`graded-v2` 不移动、重写或重新标记这些材料，也不在分支内保留本地 archive 副本。

历史迁移元数据记录了 663 个任务包、2,387 个文件、1,723 个分块和 1 个反馈文件；这些原始 blob 身份保留在 `main` 和 85ab 中。

- 旧材料**不**进入当前 graded unit 检索；`mindie-graded-unit/1` 的语料树不混入旧 `tasks/` 布局。
- 旧材料不是 graded knowledge，也不会被当作当前结论。
- `main` 的公开历史和 85ab 的不可变字节保持原样，不是重新生成的内容。
- 不修改 `main`/85ab 中的旧材料、迁移元数据或 blob 身份来让当前分支看起来更整洁。

---

## ✅ 发布与验证

在 `graded-v2` 分支上，[`publication-contract.json`](publication-contract.json) 声明 `mindie-publication-contract/2`、数据格式和发布检查使用的不可变 validator 修订。`.github/workflows/check.yml` 从该分支的可信基准确读取契约，安装精确的 validator，并以 Git blob 方式校验候选内容，不执行候选代码。

需要区分三个独立事实：

1. **本地/候选树验证**：格式、摘要、继承图、索引一致性、隐私扫描是否通过；
2. **远端发布**：`graded-v2` 的实际回读结果；`main` 保持旧 material 语料和旧 contract 字节，不随本分支发布而改变。
3. **已安装宿主验收**：具体宿主安装该修订后，原生查询、读取、反馈和路由是否实际可用。

一个通过验证的候选树不自动证明后两项，也不把文档变更变成内容自动合入。`graded-v2` 可以在兼容版本内接收内容更新，而保持同一 `publication-contract/2` 字节；`main` 不会随之改变。正式 assets、远端 `graded-v2` 发布和 paid native acceptance 是独立结果；本地准备好的内容不等于 published proof。

文档与工程文件属于 development 审核路径；单元、反馈和派生索引属于 content 审核路径。两条路径的检查结论和合入方式不同，详见[贡献指南](CONTRIBUTING.md)。

---

## 📖 文档与相关仓库

| 主题 | 中文 | English |
| --- | --- | --- |
| **单元格式规范** | [docs/package-format.md](docs/package-format.md) | [docs/package-format.en.md](docs/package-format.en.md) |
| **贡献与审核** | [CONTRIBUTING.md](CONTRIBUTING.md) | [docs/CONTRIBUTING.en.md](docs/CONTRIBUTING.en.md) |
| **仓库边界** | [AGENTS.md](AGENTS.md) | — |
| **实时单元清单** | [index/routing-manifest.json](index/routing-manifest.json) | — |
| **知识运行时** | [mindie-agent/knowledge](https://github.com/mindie-agent/knowledge) | [English README](https://github.com/mindie-agent/knowledge/blob/main/README.en.md) |
| **审核 Bot 契约** | [repository-bot-contract.zh.md](https://github.com/mindie-agent/knowledge/blob/main/docs/repository-bot-contract.zh.md) | [repository-bot-contract.md](https://github.com/mindie-agent/knowledge/blob/main/docs/repository-bot-contract.md) |
| **主仓库安装指南** | [mindie-agent](https://github.com/mindie-agent/mindie-agent) | — |

---

## 📄 许可证

本仓库的公开知识单元与文档采用 [Creative Commons Attribution 4.0 International (CC-BY-4.0)](LICENSE)。知识单元是工程参考，不替代当前环境中的实机验证；适用性始终以任务实际观测和单元声明的条件为准。
