# 单元格式规范（Graded Unit Format Specification）

**中文** | [English](package-format.en.md) · [返回仓库首页](../README.md)

本文说明 `knowledge-vllm-ascend` 中 `mindie-graded-unit/1` 单元的目录结构、Frontmatter 字段、摘要与修订规则、条件匹配、继承、路由索引、公开反馈和已审核变更。当前 C5 单元契约即 `mindie-graded-unit/1`；C6 已审核投影使用 `mindie-reviewed-change/1`，见第 9 节。格式版本与 validator 由仓库根目录的 [`publication-contract.json`](../publication-contract.json) 统一声明。

本仓库使用 `mindie-publication-contract/2`。旧 `tasks/` 材料布局不属于该契约；其字节保留在 [`docs/archive/material-v1/85ab134f333b06ac3873df6affbf3bf6ec71978c/`](archive/material-v1/85ab134f333b06ac3873df6affbf3bf6ec71978c/)，说明见 [`docs/evidence/material-v1-migration.json`](evidence/material-v1-migration.json)。归档材料不是当前单元。

---

## 1. 为什么是“条件化单元”，而不是长文摘录

一个单元把可复用的工程结论压缩成可检索、可校验、可修订的对象，同时保留它成立所需的前提：

- **正文**承载规则、步骤、代码变换、失败条件和边界；
- **Frontmatter**承载稳定 ID、层级、证据强度、条件、触发词、继承和来源；
- **内容摘要**绑定正文与语义元数据，使引用可以精确到字节内容；
- **修订号**记录同一稳定 ID 的语义演进。

检索先按条件筛选，再按词法相关度排序；命中后只返回受预算限制的摘录。需要完整上下文时，调用方按返回的摘要引用读取全文。这样既避免把整篇长文塞进上下文，也避免让模型摘要替代原始结论。

单元可以来自已审核的公开材料、源码、规范或实验记录，也可以由模型辅助提炼。但模型提案不是知识：它必须保留来源和证据边界，经过确定性格式/隐私检查与语义审核，才能成为当前单元。单元正文不是逐字转录；逐字保留的职责属于归档材料。

---

## 2. 目录与文件拓扑

```text
knowledge-vllm-ascend/
├── publication-contract.json                 # mindie-publication-contract/2
├── index/
│   ├── routing-manifest.json                 # mindie-routing-manifest/1（派生）
│   └── shard-topology.json                   # mindie-shard-topology/1（派生）
├── <domain_path>/
│   └── <unit_id>.md                          # mindie-graded-unit/1
├── feedback/
│   └── votes-<id>.json                       # mindie-graded-feedback/1
└── docs/
    ├── README.en.md
    ├── CONTRIBUTING.en.md
    ├── package-format.md
    ├── package-format.en.md
    ├── evidence/
    └── archive/material-v1/85ab134f333b06ac3873df6affbf3bf6ec71978c/
```

### 路径约束

- 单元路径必须恰好是 `<domain_path>/<unit_id>.md`。
- `<domain_path>` 是规范化的仓库相对目录：使用 `/`，不能是绝对路径，不能含 `.`、`..`、反斜杠或控制字符；首段不能是 `tasks`、`feedback`、`index`、`docs` 或 `private`。
- `<unit_id>` 匹配 `unit-[a-z0-9]+(-[a-z0-9]+)*`，稳定 ID 不随路径移动而改变。
- 单元文件必须是普通非可执行 Git blob（`100644`），UTF-8，LF 换行，不超过 256 KiB。
- `index/` 和 `feedback/` 是保留目录；文档与归档放在 `docs/`，不参与单元扫描。

---

## 3. 单元 Frontmatter

### 3.1 字段

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `schema` | string | 固定为 `mindie-graded-unit/1` |
| `id` | string | 稳定单元 ID，`unit-` 加小写单词/连字符；与路径文件名一致 |
| `revision` | integer | 正整数；新单元为 `1`；语义变化加一；仅移动路径不变 |
| `content_digest` | string | `sha256:` 加 64 位小写十六进制语义摘要；必须与重新计算结果一致 |
| `title` | string | 简洁标题，说明适用对象或结论 |
| `category` | string | 与 `tier` 对应：`axiom`、`measurement`、`sop`、`workflow`、`mutator` |
| `tier` | string | `L1`、`L2`、`L3`、`L4`；描述用途，不是置信度 |
| `evidence_strength` | string | `hardware_spec`、`reproduced_empirical`、`single_observation`、`hypothesis` |
| `domain_path` | string | 规范化相对目录；文件路径必须等于 `<domain_path>/<id>.md` |
| `conditions` | object | 非空映射；使用前必须逐项检查的前提 |
| `triggers` | object | `error_signatures` 与 `keywords` 两个字符串列表；可为空 |
| `inherits_from` | list | 同语料内的稳定单元 ID 列表；无继承时为空列表 |
| `provenance` | object | 来源类型、定位、精确摘要、证据引用和替代关系 |

### 3.2 层级与证据强度

| Tier | 对应 category | 用途 |
| --- | --- | --- |
| `L1` | `axiom` / `measurement` | 规范约束，或在声明条件下复现的测量 |
| `L2` | `sop` | 明确前提下的诊断、开发或验证步骤 |
| `L3` | `workflow` | 多步骤、有依赖的任务流程 |
| `L4` | `mutator` | 带来源与验证证据的代码变换 |

`L1` 只接受 `hardware_spec` 或 `reproduced_empirical`。`reproduced_empirical` 和 `L4` 的 `provenance.evidence_refs` 必须非空。`compile_failed`、`infra_failed`、`measurement_invalid`、`in_flight` 等未完成或失败状态不能支撑 `hardware_spec` / `reproduced_empirical`。

层级不是“可信度分数”。不要为了让单元更容易被检索或显得更重要而提升层级；证据强度也不能被层级替代。

### 3.3 来源

`provenance` 至少包含：

| 字段 | 说明 |
| --- | --- |
| `source_type` | 来源类别，例如 `ecosystem_repo`、`session_distill`、`ablation_receipt`、`run_manifest` |
| `source_locator` | 可复审的定位。可以是 `owner/repo@<40-hex>:path` 形式的 Git locator，或仓库内证据路径 |
| `source_digest` | 来源内容的精确 SHA-256，64 位小写十六进制 |
| `evidence_refs` | 相关证据记录、源码行范围或公开链接的字符串列表 |
| `supersedes` | 被本单元语义替代的稳定 ID；没有则为空列表 |
| `experiment_status` | 可选；失败/未完成状态不能支撑硬件或复现证据 |

`provenance` 记录的是来源和证据，不是“已经验收”的声明。`review_reference`、PR 链接或一次格式检查通过，不能替代证据本身。

### 3.4 示例

以下仅示意字段结构，尖括号占位符不是可发布值；真实单元必须由运行时计算摘要并校验：

```yaml
---
schema: mindie-graded-unit/1
id: unit-example-sop
revision: 1
content_digest: sha256:<runtime-computed>
title: Example source-scoped procedure
category: sop
tier: L2
evidence_strength: single_observation
domain_path: integration/example
conditions:
  repository: owner/repo
  revision: <40-hex-commit>
triggers:
  error_signatures: []
  keywords:
    - example
    - procedure
inherits_from: []
provenance:
  source_type: ecosystem_repo
  source_locator: owner/repo@<40-hex-commit>:path/to/source.py
  source_digest: <64-hex-source-digest>
  evidence_refs:
    - docs/evidence/example-review.json
  supersedes: []
---
```

正文从 Frontmatter 结束后开始，写清楚结论、步骤、失败条件和不适用范围。不要在正文里重复整张 Frontmatter 表，也不要用“见上文”代替可执行细节。

---

## 4. 摘要、修订与引用

### 4.1 内容摘要

`content_digest` 是对**语义投影**的 SHA-256：

- 取 Frontmatter 元数据，排除 `domain_path`、`revision` 和 `content_digest` 本身；
- 与正文（去除首尾空白后补一个换行）组成 `[metadata, body]`；
- 使用规范 JSON（键排序、UTF-8、无多余空格）计算 SHA-256；
- 加上 `sha256:` 前缀。

`index/routing-manifest.json` 中的 `content_sha256` 是同一个摘要去掉前缀后的 64 位十六进制。不要手写摘要或从旧文件复制；用运行时解析和计算，避免正文归一化差异导致引用失效。

摘要排除 `domain_path` 和 `revision`，所以仅移动路径不会改变内容身份；正文或语义元数据变化会改变摘要。引用中的 `@<hash>` 永远指向裸 `content_sha256`。

### 4.2 修订

- 新单元：`revision: 1`。
- 语义变化：`revision` 加一，`content_digest` 随新内容变化。
- 仅移动路径：保留原 `revision` 和内容摘要；路径更新只影响路由。
- 被替代的旧修订保留在 Git 历史中；当前语料只保留一个稳定 ID 的当前文件。

不使用旧内容摘要“覆盖”新内容，也不把旧修订的引用静默重定向到新正文。被替换或撤下的精确引用必须显式失败。

### 4.3 引用形态

```text
mindie://<domain>/units/<unit_id>@<content_sha256>
```

例如：

```text
mindie://vllm-ascend/units/unit-vllm-ascend-custom-op-gates-import-retry-and-nz@025b36835f18326dda7ef09ea380c8a4da9fd8d34eb3f39841010f302e3501ba
```

- `@` 后必须是完整 64 位小写十六进制，不能截断。
- `knowledge_explain` 可以读取当前单元的精确引用；不带 `@` 的稳定 `unit_ref` 也可用于读取当前内容。
- `knowledge_feedback` 只接受带 `@` 的精确引用，且该修订必须已被调用方实际观测过。
- `mindie://` 后的域名决定领域路由；未知或未配置的领域明确失败，不回退到默认领域。

---

## 5. 条件匹配

`conditions` 的每一项都是使用前必须成立的前提。运行时逐项比较任务提供的观测事实，所有项都成立才是 `matched`；任一项明确不符是 `mismatch`；缺失、`unknown` 或无法比较是 `unknown`。默认查询排除 `unknown`，不会把“不知道”当成“适用”。

### 5.1 基本规则

- **普通标量/列表**：声明的值是可接受的集合。观测到单个值时，它必须属于该集合；观测到列表时，每个成员都必须被允许。
- **版本说明符**：字符串以 `<`、`>`、`=`、`!`、`~` 等比较符开头时，按版本约束比较。版本格式无法解释时是 `unknown`，不是匹配。
- **`framework`**：可以是版本映射（如 `{"torch": "2.10.0+cpu", "torch_npu": "2.10.0.post4"}`，按键递归比较），也可以是 requirement 字符串列表（如 `torch>=2.10,<3`）。列表形式是**合取**组；缺包、带 marker 的条件依赖或无法解析的 requirement 都是 `unknown`。
- **嵌套对象**：按同样的规则递归比较；观测值不是对象时是 `unknown`。
- **`exact_set`**：`{"exact_set": [...]}` 要求观测集合与声明集合**完整、无序、逐成员相等**；集合顺序不影响匹配，完整的成员身份和关联才重要。成员通常包含形状、dtype、样本数、case 摘要等，不能把一个形状和另一个 case 的 dtype 拼接后声称匹配；缺成员、多成员、未知成员或重复成员都不匹配。
- **`any`**：显式声明该键不受限制；它不能替代真实条件，也不应被用来掩盖未观测的事实。
- **`unknown`**：声明值或观测值为 `unknown` 时永远不匹配。

### 5.2 查询时的条件来源

`knowledge_query` 的 `conditions` 只放**当前任务实际观测到**的事实。不要：

- 把单元自己声明的条件原样抄进查询来强制命中；
- 把 gold patch、oracle 判定、修复后代码或 heldout 答案作为“观测事实”；
- 为了凑齐条件而猜测版本、提交、硬件或形状；
- 把“应该适用”当作“已经观测”。

缺失条件得到 `condition_unknown` 是正常结果。它告诉调用方：当前证据不足以判断适用性，而不是检索器坏了。只有 `matched` 的结果才进入适用性判断；读取全文不等于自动适用。

---

## 6. 继承

`inherits_from` 引用同一语料中的稳定单元 ID。索引构建时会：

- 解析完整闭包，包括间接父单元；
- 拒绝缺失父单元、重复 ID 和继承环；
- 为每个单元计算 `closure_digest`，绑定闭包内每个成员的修订和内容摘要。

查询时，**每个父单元都要独立匹配任务条件**。子单元命中不会让不匹配的父单元变得适用；父单元正文要么完整注入，要么以明确的 `inheritance_excluded` 原因省略，不能截断前提后混入。预算不足时同样显式报告，而不是静默省略。

继承不跨领域。跨领域复用通过显式 `mindie://<domain>/...` 路由和对应领域的配置完成；不要把另一个领域的单元复制进来，也不要让两个领域共享同一个 consent 或继承图。

---

## 7. 路由索引（派生数据）

`index/routing-manifest.json` 使用 `mindie-routing-manifest/1`，由单元重建，包含：

- `schema_version`、`generation`、`unit_count`；
- 每个单元的 `id`、`title`、`tier`、`category`、`evidence_strength`、`domain_path`、`conditions`、`triggers`、`inherits_from`、`inherits_from_closure`；
- `path`、`content_sha256`、`revision`、`content_digest`、`closure_digest`、`token_estimate`、`provenance`。

`generation` 是整个语料的精确状态标识，也是已审核变更的 `base_generation`。`index/shard-topology.json` 使用 `mindie-shard-topology/1`，描述 leaf/subtree/federation 的单元数、字节数、token 估算和超限建议；这些是维护信号，不是自动拆分或自动合并命令。

两个索引都是派生数据，不应手工编辑。用运行时重建或检查：

```bash
python3 scripts/build_routing_index.py --root . --check
python3 scripts/build_routing_index.py --root .
```

`--check` 只验证现有索引是否新鲜，不写入；不带 `--check` 时在完整图校验通过后原子写入。

---

## 8. 公开反馈（`mindie-graded-feedback/1`）

公开反馈位于 `feedback/votes-<id>.json`，只包含已选择公开的投票：

```json
{
  "schema": "mindie-graded-feedback/1",
  "votes": [
    {
      "vote_id": "<opaque-id>",
      "root_id": "<opaque-root-id>",
      "unit_id": "unit-example-sop",
      "revision": 1,
      "content_digest": "sha256:<64-hex-content-digest>",
      "rating": "up",
      "reason": "Optional single-line reason."
    }
  ]
}
```

| 字段 | 说明 |
| --- | --- |
| `vote_id` | 不透明且唯一的投票 ID，非空，最多 256 字符 |
| `root_id` | 不透明匿名根标识，不暴露本地会话；最多 256 字符 |
| `unit_id` | 稳定单元 ID |
| `revision` | **整数**修订号，不是内容摘要 |
| `content_digest` | `sha256:` 加 64 位内容摘要，必须与观测修订一致 |
| `rating` | `up` 或 `down` |
| `reason` | 可选单行文本，最多 1000 字符；公开前必须通过隐私扫描 |

约束：

- 单个文件最多 500 票；同一 `(root_id, unit_id, content_digest)` 只保留一票；
- 同一内容摘要只能对应一个整数修订，反之亦然；
- 只有调用方实际观测过的修订才能投票；
- 公开统计只计算与当前内容摘要一致的票；旧摘要的票保留在本地历史，不当作当前反馈；
- 投票数量是描述性计数，不是置信度、正确性证明或自动晋升条件。没有反馈不代表内容无效。

---

## 9. 已审核变更（C6，`mindie-reviewed-change/1`）

C6 不是“模型提案自动变成单元”。它是一次显式的、可复审的本地投影：

```json
{
  "schema": "mindie-reviewed-change/1",
  "review_reference": "review-record-or-pr",
  "base_generation": "<current routing generation>",
  "changes": [
    {
      "id": "unit-example-sop",
      "expected_content_sha256": null,
      "path": "integration/example/unit-example-sop.md",
      "content": "<complete rendered unit file>"
    }
  ]
}
```

规则：

- `review_reference` 记录审核依据。它可以是 review 记录或 PR 引用，但不是签名，也不证明独立审核已经发生。
- `base_generation` 必须等于当前 `index/routing-manifest.json` 的 `generation`；语料在审核后变化时，计划被拒绝，必须重新审核。
- `changes` 中的 `content` 是**完整渲染的单元文件**，不是摘要、diff 或正文片段；撤下时 `content` 为 `null`。
- `expected_content_sha256` 用于比较并交换：新单元为 `null`；修改既有单元时必须是该单元当前的 `content_sha256`。
- 新单元必须 `revision: 1`；语义修改必须等于旧修订加一；仅移动路径保留修订号和内容摘要。
- 校验先于写入：完整继承图、重复 ID/路径、目标路径冲突和修订规则都在第一次写入前检查。
- `--apply` 只写入声明的单元、移除声明的旧文件并重建路由索引；不提交、不推送、不发布。
- 没有自动语义接受、自动晋升或“格式通过即发布”。写入返回 `partial` / `failed` 时保留实际状态，不静默重试。

```bash
# 只校验，不写入
python3 scripts/apply_reviewed_change.py --root . --plan reviewed-change.json

# 审核通过后应用
python3 scripts/apply_reviewed_change.py --root . --plan reviewed-change.json --apply
```

---

## 10. 校验、隐私与发布事实

### 10.1 确定性检查

- `scripts/build_routing_index.py --check`：验证单元图与派生索引一致。
- `python3 -m mindie_knowledge.publication_check --repo . --revision <40-hex-commit> --domain vllm-ascend`：用当前安装的 validator 校验完整发布树；CI 使用契约 pin 的 validator 复核。
- `python3 -m mindie_knowledge.redact --check ...`：确定性隐私规则集 `r3`。
- 固定版本 Gitleaks 8.30.1：第二道隐私扫描。

隐私检查必须零发现。不要用 allowlist、`.gitleaksignore`、`gitleaks:allow` 或改写扫描器来压掉真实发现；如果示例文本触发误报，保留发现记录并改用等价的无害 Markdown 写法。

### 10.2 `publication-head` 分流

`publication-contract/2` 下：

- **content**：单元 Markdown、`feedback/**`、`index/**`；检查通过后为 `success`，可由内容审核 Bot 按 `mindie-content-review/2` 处理。
- **development**：`README.md`、`AGENTS.md`、`CONTRIBUTING.md`、`docs/**`、`.github/**`、`publication-contract.json`；检查结论为 `neutral`，需要维护者按工程设计原则审核，不能由内容 Bot 自动合入。

### 10.3 三个独立事实

1. **候选树验证**：格式、摘要、继承图、索引一致性、隐私扫描；
2. **远端发布**：`main` 的实际回读；
3. **已安装宿主验收**：具体宿主安装该修订后，原生查询、读取、反馈和路由是否实际可用。

通过候选树验证不自动证明后两项；契约声明、CI 检查和 release/validation evidence 各自记录自己覆盖的事实。本文不声称任何未记录的发布或安装步骤已经完成。

---

## 11. 归档材料

`docs/archive/material-v1/85ab134f333b06ac3873df6affbf3bf6ec71978c/` 保存旧的 `material-v1` 字节，[`docs/evidence/material-v1-migration.json`](evidence/material-v1-migration.json) 记录迁移身份：

- `task_count`: 663
- `archived_file_count`: 2,387
- `block_count`: 1,723
- `feedback_count`: 1
- 每个归档文件保留原始 `git_blob_oid`。

归档不是当前单元，也不会被重新标记为 graded knowledge。它用于追溯来源和保留历史；旧 `tasks/` 布局不会重新进入 contract/2 语料树。早期公开状态和公开历史仍在 Git 历史中，归档保留原始 blob 身份，不是重新生成的内容。
