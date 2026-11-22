# Third-Party Notices & Upstream Licensing Boundaries
# 第三方组件声明与上游许可证边界说明

This repository contains **only original orchestration, 4-view Direct3D-V3 sparse-voxel fusion, geometry repair, SLIM UV / 4K cage baking, and Two-Band PBR lookdev code**, distributed under the **MIT License** ([LICENSE](LICENSE)). **No third-party model weights, GPL-3.0 binaries, or proprietary 2D/3D art assets are bundled.**

本仓库**仅包含原创的多阶段管线编排、四视角 Direct3D-V3 稀疏体素旋转融合、几何清理修复、SLIM 低岛 UV / 4K 笼罩烘焙及双频段 PBR 材质生成代码**（采用 **MIT 协议**开源，见 [LICENSE](LICENSE)）。仓库内**未打包任何第三方模型权重、GPL-3.0 二进制可执行文件或专有美术测试资产**。

When executing the pipeline in a cloud or local environment, external models and CLI tools are downloaded or invoked at runtime under their respective upstream licenses. Users are responsible for complying with the license terms of whichever upstream components they enable.

---

## 1. Permissive Core Dependencies / 宽松开源核心组件（MIT / Apache-2.0）

Recommended as the default open-source pipeline backbone:

| Component / 组件名称 | Upstream Repository / 上游地址 | License / 许可证 | Role in Pipeline / 在管线中的职责 |
|---|---|---|---|
| **Direct3D-V3** | https://github.com/DreamTechAI/Direct3D-V3 | MIT | Stage 1 `1536³` sparse-voxel refiner (4-view lossless `8³` rotation fusion) / S1 四视角 1536³ 稀疏体素精修主引擎 |
| **Pixal3D** | https://github.com/TencentARC/Pixal3D | Apache-2.0 | Stage 1 multi-view coarse & hard-surface mesh generator / S1 多视图粗模与硬表面道具生成 |
| **LATO.2** | https://github.com/LoHhhha/LATO.2 | MIT | Stage 2 hard-surface low-poly retopology / S2 硬表面低模重拓扑 |
| **StableDelight** (`yoso-delight-v0-4-base`) | https://huggingface.co/Stable-X/yoso-delight-v0-4-base | Apache-2.0 | Stage 3 specular & shadow delighting / S3 高光与投影剥离（去光） |
| **Marigold-V2** | https://github.com/huawei-bayerlab/marigold-v2 | Apache-2.0 | Stage 3 optional intrinsic albedo verification / S3 可选本征反照率校验 |
| **QIE-2511-Studio-DeLight** | https://huggingface.co/prithivMLmods/QIE-2511-Studio-DeLight | Apache-2.0 | Stage 3 optional studio delighting baseline / S3 可选影棚级去光对照基线 |
| **MV-Adapter** | https://github.com/huanngzh/MV-Adapter | Apache-2.0 | Stage 3 optional blind-spot view synthesis / S3 可选盲区辅助视角合成 |

---

## 2. Process-Isolated External GPL-3.0 Tools / 进程级隔离的外部 GPL-3.0 工具

These tools are licensed under **GPL-3.0** and are **never bundled or statically/dynamically linked into this repository**. They are installed externally by the user (see `scripts/install_external_tools.sh`) and invoked strictly as standalone CLI subprocesses over file I/O:

| Tool / 工具名称 | Upstream URL / 上游地址 | License / 许可证 | Invocation Role / 子进程调用职责 |
|---|---|---|---|
| **Blender 5.3** | https://www.blender.org/ | GPL-3.0 | OpenVDB level-set coarse splicing, SLIM UV unwrapping, Cycles 4K welded-cage baking, clay preview rendering, and glTF 2.0 `.glb` export |
| **ComfyUI** | https://github.com/Comfy-Org/ComfyUI | GPL-3.0 | Optional runtime host for Pixal3D (`path_a`) and `RemeshMesh` (CelloCut) workflows |
| **ReQuad** | https://github.com/mlstr0m/requad | GPL-3.0 | Optional quad retopology CLI/addon (supports configurable Layer-2 `X=0` mirror symmetry switch) |
| **QuadWild** | https://github.com/cgg-bern/quadwild-bimdf | GPL-3.0 | Optional quad field & patch meshing CLI engine |

---

## 3. Optional Upstream Models with Non-Commercial or Regional Terms / 含非商用或区域条款的可选上游模型

The following optional upstream components carry **non-commercial** or **regional** restrictions. They are **not bundled** in this repository; review their upstream license agreements before enabling them in commercial production, or use the listed permissive alternatives:

| Component / 组件名称 | Upstream URL / 上游地址 | License & Restriction / 许可限制 | Permissive Alternative in Engine / 管线内置的宽松商用替代方案 |
|---|---|---|---|
| **Hunyuan3D-2.1 / 2mv** | https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1 | Tencent Hunyuan Community License (non-commercial / regional restrictions) | Use **Pixal3D MV** (Apache-2.0) as the coarse shape generator |
| **PartField** | https://github.com/nv-tlabs/PartField | NVIDIA Source Code License (§3.3 Non-Commercial Research) | Disable `PartField` and use crease-guided Blender SLIM UV unwrapping |
| **DINOv3** | https://github.com/facebookresearch/dinov3 | Meta DINOv3 License | Upstream feature extractor used by `Direct3D-V3`; review Meta terms before commercial deployment |
| **Marigold-IID** (`marigold-iid-appearance-v1-1`) | https://huggingface.co/prs-eth/marigold-iid-appearance-v1-1 | CreativeML OpenRAIL++-M | Omit IID maps and use calibrated curvature/AO roughness & metallic derivation (`materials.py`) |
| **Qwen-Image-2.1** | https://huggingface.co/Qwen/Qwen-Image-2.1 | Qwen Research License (Non-Commercial) | Optional Stage 0 relight/matte helper; disabled by default |
