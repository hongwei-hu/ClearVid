---
name: implement-enhancement
description: '**WORKFLOW SKILL** — Implement ClearVid enhancement features following the phased priority roadmap from ENHANCEMENT_PROPOSALS.md. USE FOR: adding new preprocessing/postprocessing modules; integrating new models (x4plus, GFPGAN, BasicVSR++); adding encoding options (AV1, 10-bit, CRF); implementing GUI features (preview, smart params); performance optimization (TensorRT, async pipeline). Each step produces code + verification test for user acceptance. Consult [roadmap](./references/roadmap.md) for full priority matrix.'
argument-hint: 'Which phase/step to implement? e.g. "Phase 2 Step 1" or "scene detection"'
---

# ClearVid Enhancement Implementation

按照 `docs/ENHANCEMENT_PROPOSALS.md` 中的优先级路线图，逐步实现功能增强。
每完成一步产出**可运行代码 + 验收测试方案**，用户确认通过后再进入下一步。

## Architecture Context

```
clearvid/app/
├── pipeline.py          # 执行计划构建
├── orchestrator.py      # 编排器：run_single / run_batch
├── config.py            # YAML 配置加载
├── gui.py               # PySide6 GUI
├── cli.py               # Typer CLI (env/probe/plan/run/batch/gui)
├── schemas/models.py    # Pydantic 数据模型
├── models/
│   ├── realesrgan_runner.py   # Real-ESRGAN 视频处理
│   └── codeformer_runner.py   # CodeFormer 人脸修复
├── io/probe.py          # FFprobe 视频探测
├── preprocess/          # ← 空目录，待填充
├── postprocess/         # ← 空目录，待填充
└── presets/default.yaml # 默认配置
```

**关键约束**:
- V1 已实现: Real-ESRGAN (SRVGGNetCompact 4×) + CodeFormer + FFmpeg 流式管道 + NVDEC/NVENC
- 配置通过 `EnhancementConfig` Pydantic 模型管理
- 新增配置字段需同步更新: `schemas/models.py` → `default.yaml` → `gui.py` → `cli.py`

---

## Phase 2 — V1.x 增量（近期，8 步）

### Step 2.1: 单帧实时预览 Before/After ⭐⭐⭐⭐⭐

**目标**: 用户选择视频后，提取代表帧并预览增强效果，无需等待全视频处理。

**实现要点**:
1. 在 `orchestrator.py` 新增 `preview_frame(config, timestamp_sec) → (原始帧PIL, 增强帧PIL)` 方法
2. 使用 FFmpeg 提取指定时间码的单帧 → numpy 数组
3. 复用 `realesrgan_runner` 处理单帧（不启动完整管道）
4. 在 GUI 中添加 "预览" 按钮 + Before/After 分栏视图（QSplitter + QLabel）
5. 支持鼠标拖动时间滑块选择预览帧位置

**新增/修改文件**:
- `clearvid/app/orchestrator.py` — 新增 `preview_frame()`
- `clearvid/app/gui.py` — 预览面板
- `clearvid/app/models/realesrgan_runner.py` — 提取单帧处理方法

**验收测试**:
- [ ] 选择一个 480p 视频，点击预览，3 秒内显示 Before/After 对比
- [ ] 拖动时间滑块到不同位置，预览帧随之更新
- [ ] 预览能正确显示人脸修复效果（当 face_restore_enabled=true）
- [ ] 500MB+ 视频的预览不卡 GUI

---

### Step 2.2: 智能参数推荐 ⭐⭐⭐⭐

**目标**: 根据 probe 结果自动推荐最佳处理参数，降低使用门槛。

**实现要点**:
1. 新增 `clearvid/app/recommend.py`，输入 `VideoMetadata` + `EnvironmentInfo`，输出推荐的 `EnhancementConfig` 字段值
2. 推荐逻辑:
   - 480p 输入 → FHD + quality + realesrgan
   - 720p 输入 → FHD + balanced + realesrgan
   - 1080p 输入 → UHD4K + fast + realesrgan
   - 已是高码率 1080p+ → 提示 "此视频可能不需要超分"
3. 根据 GPU 显存自动推荐 `tile_size`:
   - ≥16GB → 768, ≥8GB → 512, ≥4GB → 256, <4GB → 128
4. GUI 添加 "一键最佳" 按钮，自动填充推荐参数
5. CLI `plan` 命令输出推荐信息

**新增/修改文件**:
- `clearvid/app/recommend.py` — 推荐引擎（新建）
- `clearvid/app/gui.py` — "一键最佳" 按钮
- `clearvid/app/cli.py` — plan 命令增加推荐输出

**验收测试**:
- [ ] 输入 480p 视频 → 自动推荐 FHD + quality
- [ ] 输入 1080p 高码率视频 → 提示不需要超分
- [ ] 不同 GPU 显存 → tile_size 推荐值不同
- [ ] GUI "一键最佳" 按钮正确填充所有参数

---

### Step 2.3: 预处理链 — 降噪/去块/去隔行/色彩归一化 ⭐⭐⭐⭐

**目标**: 在 Real-ESRGAN 推理前增加 FFmpeg 预处理滤镜链，提升超分质量。

**实现要点**:
1. 创建 `clearvid/app/preprocess/__init__.py` + `filters.py`
2. 实现以下可选滤镜（各自独立开关）:
   - **降噪**: `nlmeans` 滤镜（比 hqdn3d 更智能），强度自适应于输入码率
   - **去块效应**: `deblock` 滤镜，自动检测低码率 H.264 视频并启用
   - **去隔行**: `yadif` / `bwdif` 滤镜，根据 probe 检测隔行标志自动开启
   - **色彩归一化**: `colorspace` 滤镜统一到 BT.709
3. 在 `EnhancementConfig` 中添加:
   - `preprocess_denoise: bool = True`
   - `preprocess_deblock: bool = True`  
   - `preprocess_deinterlace: auto | off`
   - `preprocess_colorspace_normalize: bool = True`
4. 在 `realesrgan_runner.py` 的 FFmpeg 解码命令中注入预处理滤镜链
5. GUI 中添加"预处理"折叠面板，显示各滤镜开关

**新增/修改文件**:
- `clearvid/app/preprocess/__init__.py` + `filters.py`（新建）
- `clearvid/app/schemas/models.py` — 新增预处理字段
- `clearvid/app/presets/default.yaml` — 新增预处理默认值
- `clearvid/app/models/realesrgan_runner.py` — 注入预处理滤镜
- `clearvid/app/gui.py` — 预处理面板

**验收测试**:
- [ ] 低码率 480p H.264 视频：开启 deblock 后，超分结果块效应明显减轻
- [ ] 对比 nlmeans 降噪 开/关，高 ISO 视频噪点可见减少
- [ ] 隔行视频自动检测并去隔行，输出无梳齿
- [ ] 各滤镜独立开关正常工作，不互相干扰
- [ ] 预处理对处理速度的影响 < 10%

---

### Step 2.4: AV1 NVENC + CRF 编码 + 10-bit ⭐⭐⭐⭐

**目标**: 提供更高效的编码选项, 降低输出文件体积同时提升视觉质量。

**实现要点**:
1. 在 `schemas/models.py` 中扩展 `encoder` 字段：
   - 新增 `av1_nvenc` 选项（RTX 40/50 系列）
   - 新增 `hevc_nvenc_10bit` 选项
2. 在 `pipeline.py` / `realesrgan_runner.py` 中:
   - 若 encoder=`av1_nvenc`，使用 `-c:v av1_nvenc` + 适配参数
   - 若指定 10-bit，使用 `-pix_fmt p010le`
   - CRF 模式: 以 `-cq` 参数替代固定码率（NVENC 的 CRF 等价参数）
3. `probe.py` 中检测编码器可用性（`ffmpeg -encoders | grep av1_nvenc`）
4. GUI/CLI 公开新编码选项
5. 当检测到不支持 AV1 NVENC 时，自动回退到 HEVC

**新增/修改文件**:
- `clearvid/app/schemas/models.py` — encoder 枚举扩展
- `clearvid/app/io/probe.py` — AV1 检测
- `clearvid/app/pipeline.py` — 编码参数构建
- `clearvid/app/models/realesrgan_runner.py` — 编码命令调整
- `clearvid/app/gui.py` — 编码器选择
- `clearvid/app/presets/default.yaml` — 默认值

**验收测试**:
- [ ] RTX 40/50 卡选择 AV1 NVENC 编码成功，文件比 HEVC 小 ~20-30%
- [ ] 10-bit 输出无色带，大面积渐变色区域明显优于 8-bit
- [ ] CRF 模式输出画质稳定，不同场景码率自适应分配
- [ ] 不支持 AV1 的显卡自动回退 HEVC，不报错

---

### Step 2.5: Real-ESRGAN x4plus 模型切换 ⭐⭐⭐

**目标**: 提供基于 RRDB 架构的 x4plus 模型，细节重建能力更强。

**实现要点**:
1. 在 `schemas/models.py` 新增 `upscale_model` 字段:
   - `general_v3` (SRVGGNetCompact, 当前默认, 速度快)
   - `x4plus` (RRDB, 画质更高, 更慢)
2. 在 `realesrgan_runner.py` 中根据 `upscale_model` 选择对应架构和权重:
   - `general_v3` → `SRVGGNetCompact`, `realesr-general-x4v3.pth`
   - `x4plus` → `RRDBNet`, `RealESRGAN_x4plus.pth`
3. x4plus 权重自动下载（参考现有 general_v3 的自动下载逻辑）
4. `quality_mode=quality` 时推荐使用 x4plus
5. GUI/CLI 公开模型选择

**新增/修改文件**:
- `clearvid/app/schemas/models.py` — 新增 upscale_model 字段
- `clearvid/app/models/realesrgan_runner.py` — 模型加载逻辑
- `clearvid/app/presets/default.yaml` — 默认值
- `clearvid/app/gui.py` — 模型选择控件

**验收测试**:
- [ ] 选择 x4plus 模型，首次运行自动下载权重
- [ ] x4plus 输出对比 general_v3，毛发/纹理细节明显更丰富
- [ ] x4plus VRAM 占用在 tile_size=512 下不超过 8GB
- [ ] 模型切换不影响人脸修复流程

---

### Step 2.6: GFPGAN 可选人脸修复 ⭐⭐⭐

**目标**: 提供 GFPGAN 作为 CodeFormer 的备选人脸修复模型。

**实现要点**:
1. 在 `schemas/models.py` 新增 `face_restore_model` 字段:
   - `codeformer` (默认)
   - `gfpgan`
2. 新建 `clearvid/app/models/gfpgan_runner.py`
   - GFPGAN 已在 `pyproject.toml` 依赖中，直接集成
   - 复用 `codeformer_runner.py` 的人脸检测 + 对齐 + 融合流程
   - 替换 restoration 部分为 GFPGAN 推理
3. 在 `realesrgan_runner.py` 中根据配置选择人脸修复器
4. GUI 添加人脸修复模型下拉选择

**新增/修改文件**:
- `clearvid/app/models/gfpgan_runner.py`（新建）
- `clearvid/app/schemas/models.py` — 新增 face_restore_model
- `clearvid/app/models/realesrgan_runner.py` — 模型选择逻辑
- `clearvid/app/gui.py` — 人脸修复模型选择

**验收测试**:
- [ ] 选择 GFPGAN 模式，人脸修复正常工作
- [ ] 对比 CodeFormer 和 GFPGAN 在同一帧上的效果差异可见
- [ ] 切换模型不影响非人脸区域的超分结果
- [ ] GFPGAN 权重自动下载

---

### Step 2.7: 场景切换检测 ⭐⭐⭐

**目标**: 检测视频中的场景切换点，防止跨场景混合产生伪影。

**实现要点**:
1. 新建 `clearvid/app/preprocess/scene_detect.py`
2. 使用 FFmpeg `select='gt(scene,0.3)'` 获取场景切换时间戳列表
3. 在 `realesrgan_runner.py` 流式处理中:
   - 维护场景切换帧号列表
   - 场景切换点重置缓冲帧（为后续时序稳定做铺垫）
4. 在 `ExecutionPlan` 中输出检测到的场景数量
5. 在 `EnhancementConfig` 中添加:
   - `scene_detect_enabled: bool = True`
   - `scene_detect_threshold: float = 0.3`

**新增/修改文件**:
- `clearvid/app/preprocess/scene_detect.py`（新建）
- `clearvid/app/schemas/models.py` — 场景检测配置
- `clearvid/app/models/realesrgan_runner.py` — 场景切换感知
- `clearvid/app/presets/default.yaml`

**验收测试**:
- [ ] 多场景视频能正确检测场景切换点
- [ ] probe/plan 输出显示检测到的场景数量
- [ ] 场景切换帧无混合伪影
- [ ] 场景检测耗时 < 视频时长的 5%

---

### Step 2.8: 多场景预设 ⭐⭐⭐

**目标**: 内置针对不同场景优化的参数预设, 改善开箱体验。

**实现要点**:
1. 在 `clearvid/app/presets/` 目录下新增:
   - `portrait.yaml` — 人像视频（高 face_restore_strength, quality 模式, x4plus）
   - `landscape.yaml` — 风景/建筑（face_restore 关闭, 强锐化）
   - `fast_preview.yaml` — 快速预览（balanced, 较小 tile, preview_seconds=10）
   - `max_quality.yaml` — 最高质量（x4plus, quality 模式, 10-bit 输出）
   - `old_video.yaml` — 老旧视频（预处理全开, 去隔行, 去块, 强降噪）
2. `config.py` 支持按名称加载预设
3. GUI 添加预设选择下拉框，选择后自动填充参数
4. CLI `run`/`batch` 命令新增 `--preset` 参数

**新增/修改文件**:
- `clearvid/app/presets/*.yaml`（5 个新预设）
- `clearvid/app/config.py` — 预设加载
- `clearvid/app/gui.py` — 预设下拉框
- `clearvid/app/cli.py` — --preset 参数

**验收测试**:
- [ ] 每个预设文件 YAML 格式合法，能正确加载为 EnhancementConfig
- [ ] GUI 选择预设后所有参数正确填充
- [ ] CLI `--preset portrait` 正确加载人像预设
- [ ] 自定义修改后的参数优先于预设默认值

---

## Phase 3 — V2.0（中期，8 步）

### Step 3.1: 时序一致性后处理（光流稳定） ⭐⭐⭐⭐⭐

**目标**: 使用 RAFT 光流对超分后的相邻帧做加权混合, 减少纹理闪烁。

**实现要点**:
1. 新建 `clearvid/app/postprocess/temporal_smooth.py`
2. 集成 RAFT 轻量模型计算相邻帧光流
3. 实现自适应混合:
   - 静态区域 → 强混合 (α=0.6-0.8)，抑制闪烁
   - 运动区域 → 弱混合 (α=0.1-0.2)，保留动态
   - 混合权重由光流幅度决定
4. 缓冲 1 帧即可，内存开销可控
5. 在场景切换点重置缓冲
6. 在 `EnhancementConfig` 新增 `temporal_smooth_enabled` / `temporal_smooth_strength`

**验收测试**:
- [ ] 静态背景区域的纹理闪烁明显减少（逐帧对比 SSIM 波动下降）
- [ ] 快速运动场景不出现拖影/鬼影
- [ ] 场景切换处无异常混合帧
- [ ] 处理速度下降 < 20%

---

### Step 3.2: TensorRT 加速 ⭐⭐⭐⭐⭐

**目标**: 将 Real-ESRGAN/CodeFormer 导出为 TensorRT Engine，2-4× 速度提升。

**实现要点**:
1. 新建 `clearvid/app/models/trt_export.py` — ONNX → TRT 转换工具
2. 新建 `clearvid/app/models/trt_runner.py` — TensorRT 推理运行器
3. `torch.onnx.export()` 导出 Real-ESRGAN / CodeFormer → ONNX
4. `trtexec` 或 Polygraphy 构建 TensorRT Engine
5. 运行时: 优先加载 TRT Engine，不可用时回退 PyTorch
6. CLI 新增 `build-trt` 命令用于预构建引擎

**验收测试**:
- [ ] TRT 引擎构建成功（Real-ESRGAN + CodeFormer）
- [ ] TRT 推理结果与 PyTorch 结果视觉一致（PSNR > 40dB）
- [ ] 处理速度提升 ≥ 2×
- [ ] 无 TensorRT 环境时自动回退 PyTorch，无报错

---

### Step 3.3: BasicVSR++ 视频超分后端 ⭐⭐⭐⭐

**目标**: 引入 BasicVSR++ 作为原生视频超分后端，天然时序一致性。

**实现要点**:
1. 新建 `clearvid/app/models/basicvsr_runner.py`
2. 以滑动窗口（15 帧一组）方式处理
3. 在 `BackendType` 中新增 `basicvsr` 选项
4. 自适应窗口大小（根据 GPU 显存）
5. 权重从 MMEditing 下载

**验收测试**:
- [ ] basicvsr 后端完整视频处理成功
- [ ] 时序一致性显著优于 realesrgan 后端（帧间 SSIM 波动更小）
- [ ] 显存占用在合理范围内（16GB GPU 可处理 720p 输入）
- [ ] 与人脸修复流程兼容

---

### Step 3.4: 异步三级流水线 ⭐⭐⭐⭐

**目标**: 解码→推理→编码三阶段异步重叠, 提升 GPU 利用率和总吞吐量。

**实现要点**:
1. 重构 `realesrgan_runner.py` 为三级并行:
   - Stage 1 (CPU): FFmpeg 解码 → `decode_queue`
   - Stage 2 (GPU): 超分 + 人脸修复 → `encode_queue`
   - Stage 3 (CPU/GPU): FFmpeg 编码
2. 使用 `threading` + `queue.Queue` 实现（不用 asyncio，避免 PyTorch 兼容问题）
3. 队列大小自适应于 GPU 显存

**验收测试**:
- [ ] 吞吐量提升 ≥ 30%
- [ ] GPU 利用率从间歇性提升到持续高位（nvidia-smi 监控）
- [ ] 帧顺序正确，输出无乱序
- [ ] 进度回调仍然准确

---

### Step 3.5: 质量评估指标 (NIQE/BRISQUE) ⭐⭐⭐

**目标**: 自动评估处理后视频的画质, 提供可量化的质量参考。

**实现要点**:
1. 新建 `clearvid/app/postprocess/quality_metrics.py`
2. 集成 `pyiqa` 库的 NIQE / BRISQUE 无参考评分
3. 采样评估（每 N 帧评一次，不全帧评估）
4. 输出: 平均分、最差帧、分数趋势
5. CLI `run` 输出质量评分, GUI 显示评分摘要

**验收测试**:
- [ ] 处理后评分输出正确（NIQE 越低越好，一般 < 6 为佳）
- [ ] 不同 quality_mode 的评分差异合理
- [ ] 评估耗时 < 总处理时间的 5%

---

### Step 3.6: 模型管理系统 ⭐⭐⭐

**目标**: 统一的模型注册表, 支持增删改查和自动下载。

**实现要点**:
1. 新建 `clearvid/app/models/registry.py` — 模型注册表
2. 模型元数据: 名称、架构、适用场景、VRAM 需求、速度/质量评级
3. 模型自动下载 + 版本管理（基于 SHA256 校验）
4. 自定义模型导入（用户提供 `.pth` + 配置 JSON）
5. CLI `models list` / `models download` 命令

**验收测试**:
- [ ] `models list` 显示所有可用模型及状态
- [ ] `models download <name>` 自动下载缺失权重
- [ ] 自定义模型导入后可在处理中选用

---

### Step 3.7: 任务持久化与断点续处理 ⭐⭐⭐

**目标**: 任务队列状态持久化, 崩溃后从断点恢复。

**实现要点**:
1. 在 `task_queue.py` 中增加 SQLite 持久化
2. 记录: 任务配置、已处理帧数、状态
3. 应用重启自动检测未完成任务、提示恢复
4. 断点续处理: 从已处理帧数的下一帧继续

**验收测试**:
- [ ] 处理中途关闭应用，重启后提示恢复
- [ ] 恢复后从断点继续，不重复处理已完成帧
- [ ] 输出视频完整无缺帧

---

### Step 3.8: 打包为独立应用 ⭐⭐⭐

**目标**: 使用 PyInstaller 打包为独立 .exe, 方便分发。

**实现要点**:
1. 创建 `scripts/build.py` 或 PyInstaller spec 文件
2. 内嵌 FFmpeg 运行时
3. 处理 PyTorch + CUDA DLL 依赖
4. 生成 Windows Installer

**验收测试**:
- [ ] 打包后的 exe 在无 Python 环境的 Windows 机器上运行成功
- [ ] GPU 加速正常工作
- [ ] 所有模型权重正确打包/可自动下载

---

## Phase 4 — V3.0+（远期，7 步）

### Step 4.1: SwinIR / HAT 超分模型
### Step 4.2: 智能跳帧处理
### Step 4.3: 视频 Inpainting（水印去除）
### Step 4.4: Video Diffusion 超分
### Step 4.5: RIFE 可选插帧
### Step 4.6: 插件架构
### Step 4.7: 分布式处理

> Phase 4 各步骤的详细实现方案在启动时再展开，避免过早设计。

---

## Workflow — 每步执行流程

对每个 Step 执行以下标准流程:

### 1. 分析现状
- 阅读相关源文件，理解当前实现
- 确认与此 Step 相关的依赖是否就绪（前置 Step 是否已完成）

### 2. 设计与实现
- 最小改动原则: 只新增/修改完成目标所需的代码
- 新增配置字段: `schemas/models.py` → `default.yaml` → `gui.py` → `cli.py` 同步更新
- 新增模块: 遵循现有代码风格（Pydantic 模型、类型注解、日志记录）
- 编写实现代码, 确保无破坏性变更

### 3. 自测
- 运行 `python -m clearvid.app.cli env` 确认环境正常
- 运行 `python -m clearvid.app.cli run` 测试核心路径
- 检查 `get_errors` 确认无类型/语法错误

### 4. 交付验收
- 列出本步涉及的所有文件变更
- 提供验收测试清单（即上方每步的验收测试 checklist）
- 用户执行测试并反馈结果

### 5. 记录
- 更新 `/memories/repo/clearvid.md` 记录新增能力
- 如有经验教训, 记录到 `/memories/session/` 备查

---

## Decision Points

| 场景 | 决策 |
|------|------|
| 用户指定 "Phase 2 Step 3" | 直接实现 Step 2.3（预处理链） |
| 用户说 "下一步" | 查看 todo 列表, 实现下一个未完成 Step |
| 某步验收失败 | 不进入下一步, 修复问题后重新验收 |
| 用户想跳过某步 | 评估依赖关系, 确认无阻塞后跳过 |
| 用户想调整优先级 | 根据用户要求重排, 但标注依赖警告 |
