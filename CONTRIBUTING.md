# 参与贡献（Contributing to `knowledge-vllm-ascend`）

**中文** | [English](docs/CONTRIBUTING.en.md) · [返回仓库首页](README.md)

欢迎为 `knowledge-vllm-ascend` 贡献可在明确条件下复用的 `vllm-ascend` 工程知识。这里的贡献单位是**分级知识单元**（`mindie-graded-unit/1`），不是聊天记录或一篇泛化总结。每个单元都必须带上来源、适用条件、证据强度和修订；格式、隐私与继承关系由确定性检查验证，语义是否成立由审核者判断。

---

## 1. 可以贡献什么

- **新单元**：一条此前没有的 SOP、工作流、测量结论或代码变换，来源和条件完整。
- **修订**：对已有单元正文、条件或来源的实质修改。语义变化必须把 `revision` 加一，并保留旧内容的历史。
- **撤下**：从当前语料移除一个单元。撤下通过已审核变更完成，旧修订仍留在 Git 历史中。
- **公开反馈**：对实际观测过的精确内容修订提交 `up` / `down`。反馈不是单元，也不改变正文。

不要提交：无法复核的版本号、硬件或性能数字；把 `single_observation` 包装成 `reproduced_empirical`；把 gold patch、oracle 判定或 heldout 答案写进查询条件或单元正文；从另一个领域复制单元并改个域名。

---

## 2. 通过 MindIE Agent 贡献（推荐）

Codex、Claude Code、Kimi、DSH 四种宿主都通过各自的**原生插件**接入。请按主仓库的安装指南安装并启用对应插件：

**<https://github.com/mindie-agent/mindie-agent>**

启用社区共享（Opt-in）后，运行时的工作分成四个不同阶段；任务结束本身不会把采集材料自动提升为 graded unit：

1. **授权采集与脱敏**：Stop 钩子/任务采集只在当前授权范围内保留公开对话投影，执行确定性脱敏，并保存选中的材料。采集本身不创建 graded unit。
2. **显式选证的本地提炼**：调用方通过 `distill_selected` 明确提交 `inputs` 与 `task_state`。native worker 可以返回 `propose_graded_units` 或 `noop`，提案在本地暂存；该操作独立于 Stop，不会自行发现 transcript、授予贡献 consent 或提升运行时 pin。
3. **语义审核与 C6 应用**：调用方/审核者负责确认提案的语义、条件和证据；`review_reference` 记录审核依据，不是签名，也不证明独立审核已经发生。只有通过审核的完整单元字节经 `mindie-reviewed-change/1` 应用并重建索引，才形成当前语料的预期变更。
4. **独立的贡献/发布**：外部写入由贡献/发布路径单独治理。构建 batch 时，若目标语料使用 graded units 而仍有 legacy material drafts，构建会抛出 `DistillationRequired`（`legacy_drafts_require_c6`），要求先走 C6；不会静默生成 graded unit。以 `main` 为 base 打开 PR、远端合入和客户端安装是彼此独立的结果，不保证发生。

本仓库不复制安装脚本，也不要求在文档里手写宿主路径。安装与配置以主仓库为准。手工贡献从 `main` checkout 开始，PR 也以 `main` 为 base；已安装宿主实际使用的来源与修订以其配置为准。

---

## 3. 手工提交一个已审核变更（C6）

### 3.1 准备可核验的事实

先确定单元要回答的问题，并收集：

- 来源类型和定位（例如 `owner/repo@<40-hex>:path`，或已脱敏的证据记录路径）；
- 精确的来源摘要（SHA-256）；
- 任务实际观测到的条件：仓库与提交、硬件、CANN/驱动/固件、框架版本、输入形状、dtype、测量或验证边界；
- 证据强度及其边界：`hardware_spec`、`reproduced_empirical`、`single_observation` 或 `hypothesis`。

条件必须是**使用前可检查的前提**，不是“已经修好了”“reviewer 认为可以”或“待办步骤”。没有观测到的版本不要补；缺失和 `unknown` 在查询时不会匹配。

### 3.2 写完整单元，不写摘要替代品

单元文件路径必须是 `<domain_path>/<unit_id>.md`，正文写清楚规则、步骤或变换，以及失败条件、反例和适用范围。Frontmatter 至少包含：

`schema`、`id`、`revision`、`title`、`category`、`tier`、`evidence_strength`、`domain_path`、`conditions`、`triggers`、`inherits_from`、`provenance`。

字段语义、条件写法见[格式规范](docs/package-format.md)。新单元从 `revision: 1` 开始；修改既有单元的语义内容时，修订号加一。仅移动路径、正文和语义元数据都不变时，保留原修订号和内容哈希。不要在 frontmatter 添加旧的 `content_digest` 字段；公开引用使用整数修订号。

### 3.3 先校验，不写入

已审核变更使用 `mindie-reviewed-change/1` 计划。计划包含：

- `review_reference`：审核依据的引用。它记录审核来自哪里，不是签名，也不自动证明独立审核；
- `base_generation`：计划所基于的当前路由 `generation`；
- `changes`：每个被修改单元的完整渲染内容，或撤下标记；每项带稳定 `id`、`expected_content_sha256`、目标 `path` 和 `content`。

先用运行时做**只读校验**：

```bash
python3 scripts/apply_reviewed_change.py --root . --plan reviewed-change.json
```

该命令会检查完整继承图、重复 ID/路径、修订规则和目标路径，并打印将写入/移除的路径。它不会修改文件。若 `base_generation` 已过期，重新基于当前语料准备计划，不要绕过检查。

### 3.4 审核通过后再应用

只有在语义审核完成后才加 `--apply`：

```bash
python3 scripts/apply_reviewed_change.py --root . --plan reviewed-change.json --apply
```

应用会写入声明的单元、撤下声明的旧文件并重建 `index/routing-manifest.json` 与 `index/shard-topology.json`。它不提交、不推送、不发布。若返回 `partial` 或 `failed`，保留实际状态和错误，不要假定写入成功或盲目重试。

### 3.5 检查派生索引

```bash
python3 scripts/build_routing_index.py --root . --check
```

该命令验证单元图并确认两个派生 JSON 与当前单元一致。不要手工编辑 `index/` 下的文件；若索引过期，用同一脚本重建。

### 3.6 隐私与发布检查

至少运行：

```bash
python3 -m mindie_knowledge.redact --check README.md AGENTS.md CONTRIBUTING.md docs
python3 -m mindie_knowledge.publication_check \
  --repo . --revision "$(git rev-parse HEAD)" --domain vllm-ascend
```

并使用已安装的固定版本 Gitleaks 8.30.1 扫描变更的公开文件。扫描必须是真实的确定性检查，不能添加 allowlist、`.gitleaksignore` 或 `gitleaks:allow` 来压掉结果。若示例文本触发误报，保留该发现并改用等价的无害 Markdown 写法，而不是改扫描器或删除事实。

`publication_check` 要求提交对象是完整的 40 位 Git commit；先在本地提交候选树，再校验该 commit。

---

## 4. 隐私红线

以下内容不得进入公开文件、PR 标题或 PR 描述：

- API key、Token、私钥、Cookie、认证头；
- 内网 IPv4/IPv6 地址、带编号的机器名、`<user>@<host>` 登录串；
- 个人主目录、私有工作区路径、内部工单号、未公开的 benchmark/heldout/oracle 标识；
- 任何无法确认已公开的来源内容。

需要表达路径形状时使用 `/home/<user>/...` 等占位符。`main`/85ab 旧材料和证据中已经审核过的历史标识不要复制到新文档或新单元。

---

## 5. 两类 PR 的审核路径

`main` 是 graded 单元的 PR base；`main` 携带当前 `publication-contract/2` 声明，旧 material 归档只是 `docs/archive/material-v1/` 下的证据，不参与活动语料。`publication-head` 会按变更路径分流：

| 类型 | 涉及路径 | 检查结论 | 合入方式 |
| --- | --- | --- | --- |
| **content** | 单元 Markdown、`feedback/**`、`index/**` | `success` | 通过格式、摘要、继承、隐私检查后，由内容审核 Bot 按 [`mindie-content-review/2`](https://github.com/mindie-agent/knowledge/blob/main/docs/repository-bot-contract.zh.md) 处理 |
| **development** | `README.md`、`AGENTS.md`、`CONTRIBUTING.md`、`docs/**`、`.github/**`、`publication-contract.json` | `neutral` | 不允许内容 Bot 自动合入；需要维护者按[工程设计原则](https://github.com/mindie-agent/mindie-agent/blob/main/docs/design-principles.md)审核 |

`success` 只表示候选树通过了该检查，不是语义正确性的证明，也不等于远端 `main` 已更新或已安装宿主已验收；`main` 只随实际合入的变更改变。`neutral` 不是失败，而是要求维护者审核路径。

---

## 6. 修订、反馈与撤下

- **修订**：语义变化修订号加一；只移动路径保留修订号和内容哈希。旧引用在内容变化后应显式失败，不能静默指向新正文。
- **反馈**：只对调用方实际观测过的精确内容修订投票。`reason` 可选、单行、最多 1000 字符；本地私密反馈默认不外发。反馈数量是描述性统计，不是置信度或自动晋升条件。
- **撤下**：通过已审核变更移除当前单元，并说明原因。不要把撤下描述成内容被删除；旧内容仍在 Git 历史中。

---

## 7. 不要做的事

- 不要新增旧 `tasks/` 布局，或把 `main`/85ab 中的旧材料重新标记为 graded unit。
- 除非是单独审核的维护变更，否则不要手工编辑派生索引、`publication-contract.json` 或 CI 工作流；也不要改写 `main`/85ab 中的旧材料来配合文档。
- 不要用模型生成的摘要替代单元正文或来源。
- 不要声称“自动语义接受”“自动晋升”或“安装即验收”；这些步骤不存在。
- 不要以不存在的 `graded-v2` 分支作为 graded 单元 PR 的 base；base 是 `main`。
- 不要在两个领域之间复制单元或强行合并同主题内容；跨领域使用显式路由。

---

## 8. 相关文档

- [单元格式规范](docs/package-format.md) · [English](docs/package-format.en.md)
- [仓库边界与 Agent 规则](AGENTS.md)
- [实时路由清单](index/routing-manifest.json)
- [知识运行时](https://github.com/mindie-agent/knowledge)
- [审核 Bot 契约](https://github.com/mindie-agent/knowledge/blob/main/docs/repository-bot-contract.zh.md)
