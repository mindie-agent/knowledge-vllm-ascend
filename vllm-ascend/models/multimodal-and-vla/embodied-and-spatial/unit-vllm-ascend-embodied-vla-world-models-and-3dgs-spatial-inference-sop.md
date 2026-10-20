---
category: sop
conditions:
  framework: vllm-ascend
  npu_model:
  - Ascend910
  - Ascend910B
  - Ascend910B4
  - Ascend910_93
  - Ascend310P3
  repository:
  - vllm-ascend
  - vllm-project/vllm-ascend
  - vllm-ascend-idle-dp
domain_path: vllm-ascend/models/multimodal-and-vla/embodied-and-spatial
evidence_strength: software_contract
id: unit-vllm-ascend-embodied-vla-world-models-and-3dgs-spatial-inference-sop
inherits_from:
- unit-vllm-ascend-npu-runtime-and-graph-invariants
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/embodied/concepts/manipulation/pi0.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/embodied/concepts/manipulation/pi05.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/embodied/concepts/manipulation/isaac_gr00t.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/embodied/concepts/manipulation/openvla.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/embodied/concepts/techniques/om_atc_offline.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/embodied/concepts/world_models/cosmos_predict2_5.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/spatial/concepts/algorithms/3d_gaussian_splatting.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/spatial/concepts/models/vggt.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/spatial/concepts/models/hunyuan3d.md
  source_digest: 3e3e04b3a9bf9a7cc53485e259d5b35a57100027ae61feb7679428dcd3a53b75
  source_locator: cann/cannbot-knowledge@be5882ef29fa9943305085652d2025a038ba491c:knowledge/model/embodied/concepts/manipulation/pi0.md
  source_type: cannbot_embodied_spatial_distillation
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: Embodied VLA (Pi0, OpenVLA, Isaac GR00T), World Models (Cosmos), and 3D Gaussian
  Splatting (3DGS, VGGT, Hunyuan3D) inference on Ascend NPU
triggers:
  error_signatures:
  - VLA flow-matching action expert Euler ODE denoising loop suffers high host launch
    latency in eager mode
  - Ascend 310P ATC offline OM conversion fails on dynamic shapes or unsupported operators
    in OpenVLA / ACT / Diffusion Policy
  - 3D Gaussian Splatting rendering pipeline bottlenecks on frustum culling, tile
    intersection, or alpha-blending on Atlas A2
  keywords:
  - Pi0
  - Pi05
  - OpenVLA
  - Isaac_GR00T
  - flow_matching_action_expert
  - om_atc_offline
  - Cosmos_Predict
  - 3d_gaussian_splatting
  - FlashGaussianBuildMask
  - CalcRender
  - VGGT
  - Hunyuan3D
---
# Embodied VLA (Pi0, OpenVLA, Isaac GR00T), World Models (Cosmos), and 3D Gaussian Splatting (3DGS, VGGT, Hunyuan3D) inference on Ascend NPU

## Applicability and symptom scope

Use this SOP when adapting or optimizing Embodied AI Vision-Language-Action (VLA) models, robotic world models, or 3D spatial reconstruction/generation pipelines on Ascend Atlas A2 (`910B`), Atlas A3 (`910_93`), or edge Ascend `310P`.

## Procedure

1. **Decouple VLM prefix KV caching from the Flow-Matching Action Expert in Dual-System VLA models (`pi0.md`, `pi05.md`, `isaac_gr00t.md`, `flow_matching_action_expert.md`).**
   - In $\pi_0$, $\pi_{0.5}$, and `Isaac GR00T N1.6`, a Vision-Language backbone (PaliGemma / SmolVLM / Eagle-VLM) encodes multi-camera observations and language instructions once per control step, while a smaller **Flow-Matching Action Expert** executes $K$ iterative Euler ODE denoising steps (typically $K=10$) over an action chunk of horizon $H$ (e.g., $H=50$ steps) attending to the frozen VLM KV cache.
   - **Optimization Contract:**
     1. Compute VLM observation prefill once and freeze its KV cache tensors;
     2. Compile the $K$-step Flow-Matching Action Expert denoising loop as a static-shape `torchair` / `npugraph_ex` graph (or unroll the $K$ Euler steps inside a single captured graph) to eliminate host launch overhead at `20-50 Hz` real-time robot control frequencies;
     3. For training memory refinement (`gradient_checkpoint_refine.md`, `ddp_static_graph.md`), disable outer suffix checkpointing on the lightweight action expert and set `ddp_static_graph=True`.

2. **Compile edge manipulation policies (`ACT`, `Diffusion Policy`, `OpenVLA`, `Spirit-v1.5`) via the Ascend 310P `ONNX -> ATC -> OM` static offline pipeline (`om_atc_offline.md`, `act.md`, `diffusion_policy.md`, `openvla.md`).**
   - On Ascend 310P where online PyTorch graph compilation is unsupported and eager dispatch is slow:
     1. Export the policy (or separated ViT + LLM projector + action head subgraphs) to ONNX with frozen `--input_shape` matching the robot's fixed camera resolution and action chunk horizon;
     2. Run `atc --framework=5 --soc_version=Ascend310P3 --precision_mode=allow_fp32_to_fp16` (using `--op_select_implmode=high_performance` and explicit FP32 preservation on numerically sensitive LayerNorm/Softmax nodes);
     3. Execute the compiled `.om` model via `ais_bench` / ACL C++ runtime with preallocated device I/O buffers.

3. **Adapt World Models (`Cosmos-Predict2.5-2B`, `Cosmos-Transfer2.5-2B`, `AgiBot Arm World Model`) and Autonomous Driving VLA (`Alpamayo-R1`) on Atlas A2/A3 (`cosmos_predict2_5.md`, `cosmos_transfer2_5.md`, `alpamayo_r1.md`).**
   - Replace CUDA FlashAttention/TransformerEngine ops with `torch_npu.npu_fusion_attention` (`TND` / `BNSD`) and fused `npu_rms_norm` / `npu_rotary_mul`, and apply Ulysses CP + CFG parallel across multi-frame world-state rollouts.

4. **Accelerate 3D Gaussian Splatting (`3DGS`) and 3D Spatial Models (`VGGT`, `Hunyuan3D 2.0`) with fused AscendC spatial kernels (`3d_gaussian_splatting.md`, `vggt.md`, `hunyuan3d.md`).**
   - In `3DGS` rendering/training on Atlas A2, replace fine-grained CUDA rasterization kernels with the 6 fused AscendC spatial operators:
     1. `GaussianFilter` (`gaussian_filter.md`): fused frustum culling + scatter compaction;
     2. `Projection3DGS` (`projection_3dgs.md`): fused 3D-to-2D covariance & conic projection (forward + backward);
     3. `SphericalHarmonics` (`spherical_harmonics.md`): fused SH coefficients-to-RGB evaluation;
     4. `PreciseIntersection` / `FlashGaussianBuildMask` (`precise_intersection.md`): exact 2D ellipse-to-pixel-tile intersection mask generation, eliminating false-positive bounding-box tile pairs;
     5. `GaussianSort` (`gaussian_sort.md`): per-tile radix depth sorting;
     6. `CalcRender` (`calc_render.md`): fused tile-based alpha-blending transmittance accumulation.
   - For `VGGT` (Visual Geometry Grounded Transformer) and `Hunyuan3D 2.0`, fuse alternating frame/global attention with `npu_fusion_attention` and replace custom CUDA marching-cubes/mesh rasterizers with NPU-native vector/host-pipelined equivalents.
