---
blocks:
- block_id: cf703b278dafb00eecaa9824bb9004026217d95b063cdf3643b96b2b5fb5529a
  indexed: true
  sha256: 3e37506a35a177365d764769d957257f8f48470aed1acff315892f47adf13f98
  source_range:
    end: 236086
    part: 0
    start: 0
  summary: '助手确认经验贡献已开启：目标库为 mindie-agent/knowledge-vllm-ascend（main 分支），账号 maoxx241，项目根路径已脱敏。状态为 experience: configured、sharing enabled，worker/outbox/summary 服务存活，知识存储就绪。会话 Stop 事件将后台自动沉淀经验且选择持久保存；诊断上报仍关闭。此前 mindie-knowledge outbox 的 KeyError 被说明为服务启动前的历史记录。'
  title: 经验贡献配置完成并启用
- block_id: 5f5ced4cb40754d234d7887ddc991d9cd90b1ecc8f57fefc5a3df96316a05e09
  indexed: true
  sha256: 8f77d9738d2cbc63cd27c3e36ebb95b605f02c5f1bb2665365ea5db378bfe56e
  source_range:
    end: 241940
    part: 0
    start: 236086
  summary: 用户要求凭常识一句话回答 vLLM-Ascend 中运行算子单测前必须确认的 NPU 环境事项。助手给出三点：NPU 设备在线且空闲（可用 npu-smi info 查看）、已 source CANN 环境变量（/usr/local/Ascend/ascend-toolkit/set_env.sh）、torch 与 torch_npu 版本匹配，否则单测会在设备初始化或算子加载阶段直接失败。该回答为常识性建议，未附实际验证结果。
  title: vLLM-Ascend 算子单测前的 NPU 环境确认事项
entry:
  conditions: {}
  domain: vllm-ascend
  entry_id: 276b9c949662da20e9238ae7d9953224d4fea17a4c5714cb3cbbbb3ac0643d2b
  kind: experience
  material_digest: f864f26d48f1a339407bad3ac45f71a68d4cbca100869bba96d6a84460f1f765
  revision: 1220d67e6bb64c70511d60efffb05f6357acf380461b831f49e0c00cd9e3e2a8
  schema: mindie-entry/3
  summary: 早前上下文：经验贡献已配置完成并启用（目标库 mindie-agent/knowledge-vllm-ascend，账号 maoxx241），相关服务存活；诊断上报保持关闭，早前 outbox KeyError 被归为服务启动前的历史记录。本批次新增一条常识性问答：助手说明 vLLM-Ascend 跑算子单测前需确认 NPU 设备在线空闲（npu-smi info）、CANN 环境变量已 source、torch 与 torch_npu 版本匹配，否则会在设备初始化或算子加载阶段失败；该建议未经实际验证。
  title: 经验贡献配置与 vLLM-Ascend 单测环境问答
navigation: 早前上下文：经验贡献已配置完成并启用（目标库 mindie-agent/knowledge-vllm-ascend，账号 maoxx241），相关服务存活；诊断上报保持关闭，早前 outbox KeyError 被归为服务启动前的历史记录。本批次新增一条常识性问答：助手说明 vLLM-Ascend 跑算子单测前需确认 NPU 设备在线空闲（npu-smi info）、CANN 环境变量已 source、torch 与 torch_npu 版本匹配，否则会在设备初始化或算子加载阶段失败；该建议未经实际验证。
schema: mindie-material-task/1
status: complete
task_id: 276b9c949662da20e9238ae7d9953224d4fea17a4c5714cb3cbbbb3ac0643d2b
---

# 经验贡献配置与 vLLM-Ascend 单测环境问答

早前上下文：经验贡献已配置完成并启用（目标库 mindie-agent/knowledge-vllm-ascend，账号 maoxx241），相关服务存活；诊断上报保持关闭，早前 outbox KeyError 被归为服务启动前的历史记录。本批次新增一条常识性问答：助手说明 vLLM-Ascend 跑算子单测前需确认 NPU 设备在线空闲（npu-smi info）、CANN 环境变量已 source、torch 与 torch_npu 版本匹配，否则会在设备初始化或算子加载阶段失败；该建议未经实际验证。

## Materials

- [经验贡献配置完成并启用](blocks/cf703b278dafb00eecaa9824bb9004026217d95b063cdf3643b96b2b5fb5529a.md): 助手确认经验贡献已开启：目标库为 mindie-agent/knowledge-vllm-ascend（main 分支），账号 maoxx241，项目根路径已脱敏。状态为 experience: configured、sharing enabled，worker/outbox/summary 服务存活，知识存储就绪。会话 Stop 事件将后台自动沉淀经验且选择持久保存；诊断上报仍关闭。此前 mindie-knowledge outbox 的 KeyError 被说明为服务启动前的历史记录。
- [vLLM-Ascend 算子单测前的 NPU 环境确认事项](blocks/5f5ced4cb40754d234d7887ddc991d9cd90b1ecc8f57fefc5a3df96316a05e09.md): 用户要求凭常识一句话回答 vLLM-Ascend 中运行算子单测前必须确认的 NPU 环境事项。助手给出三点：NPU 设备在线且空闲（可用 npu-smi info 查看）、已 source CANN 环境变量（/usr/local/Ascend/ascend-toolkit/set_env.sh）、torch 与 torch_npu 版本匹配，否则单测会在设备初始化或算子加载阶段直接失败。该回答为常识性建议，未附实际验证结果。
