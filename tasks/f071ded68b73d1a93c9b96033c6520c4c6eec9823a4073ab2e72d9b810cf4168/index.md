---
blocks:
- block_id: 346c2d249ddaa0447b33aa1ea4d46f8f3c5f96ddac2e9bc412d1f8037ca4788e
  indexed: true
  sha256: 525628af95d901ce50269df4dcb6ba6b5ab468d471df9edc1642e7cfacc34887
  source_range:
    body_end: 2648
    body_start: 0
    published_commit: 9aab5e1e2f1298f992b1b41e403c2b1e0232d12c
    published_path: cases/f071ded68b73d1a93c9b96033c6520c4c6eec9823a4073ab2e72d9b810cf4168.md
  summary: 'Conversation excerpt: 帮我看下旋转量化的权重是哪一个版本，我记得13277是验证过全量权重的

    我先按“旋转量化 = QuaRot / global rotation”这个线索，把历史验证记录、PR #13277 的权重路径和版本标识对齐；先查本地与已有验证证据，不启动四节点服务。

    目前线索里至少有两个 QuaRot 目录：`...quarot-all-0722` 和 `0726_...flex-quarot`。我继续查 #13277 那次“六套 W4A8 全量权重矩阵”的原始记录，避免把后续 #13327/#13407 的 0726 权重混进来。

    “旋转量化”在 #13277 记录里不是唯一目录名。为判断你现在指的是 0722 还是后续 0726，我会用 Chronicle 只看最近相关上下文；它只用于消除这个命名歧义，不做任何远端操作。

    Chronicle 当前未能核实为可用状态，所以按它的约束不读取屏幕；这不影响结论，我改用本地归档和 `gh` 上 #13277 的已发布评论交叉确认。

    你记得对。#13277 的全量权重矩阵里，后续继续用于 QuaRot/128K 排查的版本是：

    `<redacted:credential-high-entropy:42a314f41991>`

    - 0726 版本，363 个顶层文件、346 个 safetensors。

    - `optional/quarot.safetensors` 包含 `(7168, 7168)` 的 `global_rotation`。

    - #13277 上按四节点 TP16×DP4×EP64 全量加载，日志显示 345 个模型分片加 1 个 optional 分片，不是 5-layer 裁剪权重。

    - 搭配的 draft 是 `Kimi-K3-DSpark-GQA-20260801`，标记 `epoch=10 step=1150`。

    不过 #13277 当时其实'
  title: 帮我看下旋转量化的权重是哪一个版本，我记得13277是验证过全量权重的
entry:
  conditions: {}
  domain: vllm-ascend
  entry_id: f071ded68b73d1a93c9b96033c6520c4c6eec9823a4073ab2e72d9b810cf4168
  kind: experience
  material_digest: 412dc794eaca53421f9cf1b87990fee26a7b5abdab6fb50176ad4dd62d1eec2b
  revision: 5ebd20bfc7413ccb8cf72a05ed514260b81d944dfc0393c5b4019b8f701ee812
  schema: mindie-entry/3
  summary: 'Conversation excerpt: 帮我看下旋转量化的权重是哪一个版本，我记得13277是验证过全量权重的

    我先按“旋转量化 = QuaRot / global rotation”这个线索，把历史验证记录、PR #13277 的权重路径和版本标识对齐；先查本地与已有验证证据，不启动四节点服务。

    目前线索里至少有两个 QuaRot 目录：`...quarot-all-0722` 和 `0726_...flex-quarot`。我继续查 #13277 那次“六套 W4A8 全量权重矩阵”的原始记录，避免把后续 #13327/#13407 的 0726 权重混进来。

    “旋转量化”在 #13277 记录里不是唯一目录名。为判断你现在指的是 0722 还是后续 0726，我会用 Chronicle 只看最近相关上下文；它只用于消除这个命名歧义，不做任何远端操作。

    Chronicle 当前未能核实为可用状态，所以按它的约束不读取屏幕；这不影响结论，我改用本地归档和 `gh` 上 #13277 的已发布评论交叉确认。

    你记得对。#13277 的全量权重矩阵里，后续继续用于 QuaRot/128K 排查的版本是：

    `<redacted:credential-high-entropy:42a314f41991>`

    - 0726 版本，363 个顶层文件、346 个 safetensors。

    - `optional/quarot.safetensors` 包含 `(7168, 7168)` 的 `global_rotation`。

    - #13277 上按四节点 TP16×DP4×EP64 全量加载，日志显示 345 个模型分片加 1 个 optional 分片，不是 5-layer 裁剪权重。

    - 搭配的 draft 是 `Kimi-K3-DSpark-GQA-20260801`，标记 `epoch=10 step=1150`。

    不过 #13277 当时其实'
  title: 帮我看下旋转量化的权重是哪一个版本，我记得13277是验证过全量权重的
navigation: 'Conversation excerpt: 帮我看下旋转量化的权重是哪一个版本，我记得13277是验证过全量权重的

  我先按“旋转量化 = QuaRot / global rotation”这个线索，把历史验证记录、PR #13277 的权重路径和版本标识对齐；先查本地与已有验证证据，不启动四节点服务。

  目前线索里至少有两个 QuaRot 目录：`...quarot-all-0722` 和 `0726_...flex-quarot`。我继续查 #13277 那次“六套 W4A8 全量权重矩阵”的原始记录，避免把后续 #13327/#13407 的 0726 权重混进来。

  “旋转量化”在 #13277 记录里不是唯一目录名。为判断你现在指的是 0722 还是后续 0726，我会用 Chronicle 只看最近相关上下文；它只用于消除这个命名歧义，不做任何远端操作。

  Chronicle 当前未能核实为可用状态，所以按它的约束不读取屏幕；这不影响结论，我改用本地归档和 `gh` 上 #13277 的已发布评论交叉确认。

  你记得对。#13277 的全量权重矩阵里，后续继续用于 QuaRot/128K 排查的版本是：

  `<redacted:credential-high-entropy:42a314f41991>`

  - 0726 版本，363 个顶层文件、346 个 safetensors。

  - `optional/quarot.safetensors` 包含 `(7168, 7168)` 的 `global_rotation`。

  - #13277 上按四节点 TP16×DP4×EP64 全量加载，日志显示 345 个模型分片加 1 个 optional 分片，不是 5-layer 裁剪权重。

  - 搭配的 draft 是 `Kimi-K3-DSpark-GQA-20260801`，标记 `epoch=10 step=1150`。

  不过 #13277 当时其实'
schema: mindie-material-task/1
status: complete
task_id: f071ded68b73d1a93c9b96033c6520c4c6eec9823a4073ab2e72d9b810cf4168
---

# 帮我看下旋转量化的权重是哪一个版本，我记得13277是验证过全量权重的

Conversation excerpt: 帮我看下旋转量化的权重是哪一个版本，我记得13277是验证过全量权重的
我先按“旋转量化 = QuaRot / global rotation”这个线索，把历史验证记录、PR #13277 的权重路径和版本标识对齐；先查本地与已有验证证据，不启动四节点服务。
目前线索里至少有两个 QuaRot 目录：`...quarot-all-0722` 和 `0726_...flex-quarot`。我继续查 #13277 那次“六套 W4A8 全量权重矩阵”的原始记录，避免把后续 #13327/#13407 的 0726 权重混进来。
“旋转量化”在 #13277 记录里不是唯一目录名。为判断你现在指的是 0722 还是后续 0726，我会用 Chronicle 只看最近相关上下文；它只用于消除这个命名歧义，不做任何远端操作。
Chronicle 当前未能核实为可用状态，所以按它的约束不读取屏幕；这不影响结论，我改用本地归档和 `gh` 上 #13277 的已发布评论交叉确认。
你记得对。#13277 的全量权重矩阵里，后续继续用于 QuaRot/128K 排查的版本是：
`<redacted:credential-high-entropy:42a314f41991>`
- 0726 版本，363 个顶层文件、346 个 safetensors。
- `optional/quarot.safetensors` 包含 `(7168, 7168)` 的 `global_rotation`。
- #13277 上按四节点 TP16×DP4×EP64 全量加载，日志显示 345 个模型分片加 1 个 optional 分片，不是 5-layer 裁剪权重。
- 搭配的 draft 是 `Kimi-K3-DSpark-GQA-20260801`，标记 `epoch=10 step=1150`。
不过 #13277 当时其实

## Materials

- [帮我看下旋转量化的权重是哪一个版本，我记得13277是验证过全量权重的](blocks/346c2d249ddaa0447b33aa1ea4d46f8f3c5f96ddac2e9bc412d1f8037ca4788e.md): Conversation excerpt: 帮我看下旋转量化的权重是哪一个版本，我记得13277是验证过全量权重的
我先按“旋转量化 = QuaRot / global rotation”这个线索，把历史验证记录、PR #13277 的权重路径和版本标识对齐；先查本地与已有验证证据，不启动四节点服务。
目前线索里至少有两个 QuaRot 目录：`...quarot-all-0722` 和 `0726_...flex-quarot`。我继续查 #13277 那次“六套 W4A8 全量权重矩阵”的原始记录，避免把后续 #13327/#13407 的 0726 权重混进来。
“旋转量化”在 #13277 记录里不是唯一目录名。为判断你现在指的是 0722 还是后续 0726，我会用 Chronicle 只看最近相关上下文；它只用于消除这个命名歧义，不做任何远端操作。
Chronicle 当前未能核实为可用状态，所以按它的约束不读取屏幕；这不影响结论，我改用本地归档和 `gh` 上 #13277 的已发布评论交叉确认。
你记得对。#13277 的全量权重矩阵里，后续继续用于 QuaRot/128K 排查的版本是：
`<redacted:credential-high-entropy:42a314f41991>`
- 0726 版本，363 个顶层文件、346 个 safetensors。
- `optional/quarot.safetensors` 包含 `(7168, 7168)` 的 `global_rotation`。
- #13277 上按四节点 TP16×DP4×EP64 全量加载，日志显示 345 个模型分片加 1 个 optional 分片，不是 5-layer 裁剪权重。
- 搭配的 draft 是 `Kimi-K3-DSpark-GQA-20260801`，标记 `epoch=10 step=1150`。
不过 #13277 当时其实
