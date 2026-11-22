# 架构与算法设计规范：第一层通用内核（Layer 1）与第二层可配置扩展（Layer 2）

[English](ARCHITECTURE.md) | **简体中文**

---

## 1. 核心架构原则

1. **第一层：100% 品类无关的通用内核（Layer 1）**
   `quadvox/` 内核不包含任何特定角色的坐标硬编码、RGB 颜色阈值或特定专有格式假设。所有几何重建、重拓扑、SLIM 低碎片化 UV 展开、4K 焊合笼罩烘焙与双频段 PBR 材质生成算子均直接作用于任意三维网格。
2. **第二层：基于 YAML 的显式可选扩展（Layer 2）**
   针对人形角色或左右对称资产的高级能力——**正面 ROI 视角优先级锁定（`front_roi_box`）**、**基于 OpenVDB 水平集的双粗模空间分区拼合（`coarse_splice`）** 以及 **`X=0` 镜像对称四边面重拓扑（`symmetry_x`）**——全部收敛为 `configs/presets/*.yaml` 中的可配置参数，并在 `configs/engine_defaults.yaml` 中默认关闭（`false` / `null`）。
3. **宿主机安全与进程级许可隔离**
   - 瘦客户端云端编排器（`runners/`）不在本地控制机加载千万面高模或运行重度 Blender 渲染；重算力阶段统一通过 `quadvox/cloud/colab_resilience.py` 调度至远端 GPU Worker 执行。
   - 外部 GPL-3.0 工具（`Blender`、`ReQuad`、`QuadWild`、`ComfyUI`）绝不随仓库打包分发，统一通过 `quadvox/tools/external_tools.py` 解析可执行文件路径并以独立 CLI 子进程方式调用。
   - 全仓代码严禁硬编码 `/tmp` 或特定机器的用户主目录路径。

---

## 2. Stage 0（`quadvox/s0_views/`）：多视图预处理与正交相机标定

- **无材质哑光白模提示词构建（`turnaround.py`）**：遵循 Direct3D-V3 上游 `assets/prompt.txt` 分布规范，生成剥离高光反射、烘焙阴影与复杂表面纹理的哑光几何白模提示词（Geometry-First Matte Clay Prompting），避免材质纹理干扰 S1 几何推理。
- **RGBA 标准化与轨道相机清单（`preprocess.py`）**：提取高精度 Alpha 遮罩，按统一留边比例（`pad_factor=1.1`）居中对齐 `0° / 90° / 180° / 270°` 四视图，并输出标准轨道相机参数 `transforms.json`。
- **正交相机轮廓标定与形变对齐（`camera.py`）**：针对目标 3D 网格轮廓自动标定各视角的正交缩放与二维平移偏移（`calibrate_s0_ortho_camera`），并提供平滑轮廓对齐形变场计算（`compute_silhouette_alignment_warp`）。
- **四视图输入对齐边界（必须对齐 vs 自动纠偏）**：
  - **代码自动纠偏**：全局画面缩放与画布二维居中偏移无需手动抠像素对齐，由 `preprocess.py` 与 `camera.py` 自动解算纠偏。
  - **必须严格对齐的 3 项输入约束**：
    1. **关键解剖/结构水平高度线与正交零俯仰角（`0° / 90° / 180° / 270°`）**：各水平部位的相对垂直高度比例必须一致且无俯仰透视角差（实证中正面与侧面仅差数像素的 `X` 发卡会在 `Direct3D-V3` 中雕出 `XX` 双重影，未对齐的正背飘带会融合出“薄桨叶”切面伪影）。
    2. **静止姿态与四肢角度严格一致**：T-Pose/A-Pose 手臂高度、脚距与袖口朝向必须统一（侧视图手臂须水平正对镜头截缩），否则 `Pixal3D MV` 易在腋下/躯干间生成粘连桥。
    3. **跨视角语义零矛盾**：正/侧/背不得脑补出互相冲突的部件（如侧面无、背面却凭空出现的马尾或反转的不对称花纹），否则多视角体素交界面会长出“海绵碎桥”。

---

## 3. Stage 1（`quadvox/s1_geometry/`）：四视角 Direct3D-V3 1536³ 无损稀疏体素旋转融合

核心实现位于 `quadvox/s1_geometry/d3dv3_fusion.py`：

1. **精确离散 `8³` 稀疏体素块旋转（`build_view_rotation`、`rotate_latents_forward`、`rotate_latents_inverse`）**
   - 将稀疏体素块坐标 `(batch, px, py, pz)` 与块内 `8×8×8` 体素张量绕 `+Y` 轴做 `0° / 90° / 180° / 270°` 正交旋转重排（`det = +1`）。
   - 每次推理启动前自动执行 `<5ms` 零误差往返自检（`max_abs_err == 0.0`）。
2. **正交深度缓冲遮挡剔除与悬垂内凹防重影抑制（`compute_surface_visibility_and_weights`）**
   - 为 `front (+Z)`、`back (-Z)`、`left (+X)`、`right (-X)` 四个正交方向实时光栅化无孔洞 `512×512` Z-Buffer。
   - 同步计算陡峭遮挡深度断层抑制因子（`cliff_keep`）与向下悬垂内凹区域抑制因子（`crevice_keep`），使内凹盲区平滑退回 C1 连续的前/后主视角权重，消除侧视角掠射引起的几何撕裂与重影。
   - 将表面可见性权重插值映射至稀疏 `8³` 晶格节点（`compute_4view_voxel_weights`），对内部不可见体素采用居中前/后平滑兜底权重。
3. **第二层可选能力：正面 ROI 视角优先级锁定（`ROISovereigntyBox`）**
   - 在归一化 `[0, 1]` 高度锚定坐标系下评估 C1 平滑三维包围盒（支持通过 `anchor_z_box` 动态锚定局部前后深度中心），在包围盒内提升指定主视角（`target_view`）权重并抑制正交侧视角干扰。
4. **跨时间步方向温度锐化与后期 CFG 余弦退火（`compute_step_beta_and_cfg`、`sharpen_view_weights`）**
   - 当采样进度 `t >= 0.55` 时，对四视角权重执行温度锐化 `w^beta / sum(w^beta)`（`512 / 1024 / 1536` 分辨率对应的 `beta_max` 分别为 `3.5 / 4.5 / 5.5`）。
   - 当 `t >= 0.72` 时，将 CFG 引导尺度余弦退火至 `cfg_late_min = 3.5`，消除接缝隆起与高频振铃。

---

## 4. 第二层可选能力：基于 OpenVDB 水平集的双粗模空间分区拼合（`quadvox/s1_geometry/coarse_splice.py`）

当配置 `s1.coarse_splice.enabled: true` 时，系统将两套优势互补的粗模（例如一套整体比例与四肢轮廓准确、另一套正面局部结构清晰）在 `voxel_res=384` 分辨率下转换为 OpenVDB `FloatGrid` 有向距离水平集（Level Set），沿 `split_axis` 在 `transition_range` 区间内构造 C1 连续 `smoothstep` 混合权重场（支持 `lateral_guard_x` / `lateral_guard_y_max` 侧向空间保护），融合后重新提取为单一水密粗模。

---

## 5. Stage 2（`quadvox/s2_retopo/`）与 Stage 3（`quadvox/s3_lookdev/`）

- **Stage 2 — 特征保真重拓扑、低碎片化 SLIM UV 与 4K 焊合笼罩烘焙**
  - `preclean.py` 与 `qem.py`：拓扑保持型二次误差度量减面（基于 PyMeshLab `preserve_topology=True, planar_quadric=False` / pyfqmr / Open3D）。
  - `lato2_repair.py`：针对硬表面 `LATO.2` 低模输出的几何清理（共面重叠面剔除、微孔封堵、全局法线朝向一致性修正）。
  - `quad_retopo.py`：`ReQuad`、`QuadWild` 与 `QuadriFlow` 子进程封装，支持第二层 `symmetry_x` 对称开关（默认 `False`）。
  - `slim_uv_bake.py`：折痕引导的 Blender 5.3 `MINIMUM_STRETCH`（SLIM）低岛数 UV 展开，以及基于焊合膨胀笼罩（`WeldedSmoothBakeCage`）的 4K Cycles 法线、AO 与曲率烘焙。
- **Stage 3 — 区域保护去光、双频反投影与跨接缝三维流形谐波补洞**
  - `delight.py`：带保护掩码曝光与色度约束的区域保护去光。
  - `backproject.py`：高低频分离的 `Two-Band` 多视角反投影（`low_power=4.0`、`detail_power=32.0`）、第二层 `ROISovereigntyBox` 视角优先级锁定、在焊合三维网格拓扑上直接求解稀疏 Dirichlet Laplace 方程的**三维流形谐波补洞**，以及基于欧氏距离变换（EDT）的 UV 接缝外扩填充（Gutter Padding）。
  - `materials.py` 与 `studio_export.py`：基于曲率/AO 的硬表面金属磨边与暗缝积灰 PBR 派生（或通用 ORM 贴图派生），并通过 Blender 5.3 Cycles 导出标准 glTF 2.0 `.glb` 资产。
