# ClearVid

[English](README.md) | [简体中文](README.zh-CN.md)

ClearVid 是一个面向 Windows 的视频清晰度增强工具，主要针对真实拍摄素材，重点覆盖 480p 和 720p 输入，并支持输出到 1080p、4K、2x 放大和 4x 放大等常见目标规格。

当前项目已经具备可用的 CLI、桌面 GUI、自动环境与启动引导逻辑、模型权重管理，以及基于 Real-ESRGAN 的主处理链路，并支持可选的人脸修复和后处理。

## 当前状态

当前仓库已经实现的内容包括：

- 面向 Windows 的开发和终端用户使用流程。
- CLI 命令，支持环境检查、视频探测、执行计划、单文件处理、批处理和 GUI 启动。
- 基于 PySide6 的桌面 GUI，包含文件管理、预览、导出队列、预设、历史记录、环境诊断和进度显示。
- 智能推荐流程，可为目标规格、质量模式、模型、编码器、加速器和分块参数给出建议。
- 自动后端选择；当模型运行时不可用时自动回退到 baseline 处理路径。
- Real-ESRGAN 后端，支持 `general_v3` 和 `x4plus` 两条超分模型路径。
- 可选的人脸修复，支持 CodeFormer 和 GFPGAN。
- 预处理选项，支持降噪、去块、反交错和色彩空间归一化。
- 后处理选项，支持锐化和基于光流的时序稳定。
- 支持 8-bit 和 10-bit 输出像素格式，包括 `yuv420p`、`yuv420p10le` 和 `p010le`。
- FFmpeg 与 Python 之间的原始帧流式传输，以减少磁盘 I/O，同时不降低导出质量。
- 支持异步多阶段流水线、动态 batch 以及 TensorRT 引擎缓存。
- 面向便携分发的一次性启动引导流程，可在首次运行时将依赖安装到 `lib/`。
- Real-ESRGAN、CodeFormer、GFPGAN 和 facelib 相关权重的下载管理。

当前版本暂不包含：

- 插帧功能尚未实现。

## 推荐环境

- Windows 11
- Python 3.11 到 3.13
- FFmpeg 和 FFprobe 可通过项目根目录或 `PATH` 访问
- 用于 Real-ESRGAN 路径的 NVIDIA 显卡和较新的驱动
- 用于 GPU 推理的 CUDA 版 PyTorch 环境

仓库中的 `pyproject.toml` 当前支持 Python `>=3.11,<3.14`。

## 安装

### 源码安装

先创建虚拟环境并安装基础包：

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e .
```

按需安装可选功能：

```powershell
python -m pip install -e .[gui,media]
python -m pip install -e .[inference]
python -m pip install -e .[acceleration]
python -m pip install -e .[dev]
```

常见组合：

- 仅 CLI：基础包即可
- 带 GUI 但不启用 AI 推理：`.[gui,media]`
- 本地完整工作站环境：`.[gui,media,inference,acceleration]`

### 便携版构建

仓库内置了用于生成自包含 Windows 发布包的脚本：

```powershell
.\scripts\build_dist.ps1
```

它会生成一个便携 ZIP，包含：

- Python Embeddable 运行时
- ClearVid 源码
- 启动脚本
- 如果能找到则一并打包 FFmpeg 二进制
- `lib/`、`outputs/`、`weights/` 和 `samples/` 等占位目录

首次启动时，bootstrap launcher 会先把依赖安装到 `lib/`，然后再启动 GUI。

## 启动方式

### CLI 入口

安装完成后，可使用以下入口：

- `clearvid`
- `clearvid-gui`

### Windows 启动脚本

仓库还提供了可直接双击运行的启动脚本：

```powershell
.\Start_ClearVid_GUI.bat
```

该脚本会自动判断当前应进入：

- 便携版 launcher 模式，或
- 开发环境 GUI 模式

## CLI 命令示例

查看本地环境：

```powershell
clearvid env
```

探测视频信息：

```powershell
clearvid probe samples\sample.mp4
```

生成执行计划并输出智能推荐：

```powershell
clearvid plan samples\sample.mp4 outputs\sample_plan.mp4 --target-profile fhd --backend auto --quality-mode quality
```

运行 baseline 验证导出：

```powershell
clearvid run samples\sample.mp4 --target-profile fhd --backend baseline --output outputs\sample_fhd_baseline.mp4
```

只处理前 20 秒做快速预览：

```powershell
clearvid run samples\sample.mp4 --target-profile fhd --backend auto --preview-seconds 20 --output outputs\sample_preview.mp4
```

使用更高画质的 `x4plus`、GFPGAN、人脸修复、TensorRT 和 10-bit 输出：

```powershell
clearvid run samples\sample.mp4 --target-profile fhd --backend realesrgan --upscale-model x4plus --face-restore-model gfpgan --inference-accelerator tensorrt --output-pixel-format yuv420p10le --output outputs\sample_fhd_x4plus_10bit.mp4
```

批量处理整个目录：

```powershell
clearvid batch samples --target-profile fhd --backend auto --quality-mode balanced --output-dir outputs
```

从 CLI 启动 GUI：

```powershell
clearvid gui
```

或者直接使用：

```powershell
clearvid-gui
```

## 模型与运行时行为

### 可用后端

- `auto`：优先使用 Real-ESRGAN，运行时不可用时自动回退
- `realesrgan`：当前主要画质路径
- `baseline`：基于 FFmpeg 的验证/兜底路径

### 可用超分模型

- `general_v3`：更轻量的默认路径
- `x4plus`：更高画质的 RRDB 路径，但计算开销更高

### 可用人脸修复模型

- `codeformer`：偏自然、保真可调
- `gfpgan`：偏强化修复和美化

### 自动下载的权重

ClearVid 当前可以按需自动下载以下权重：

- `realesr-general-x4v3.pth`
- `RealESRGAN_x4plus.pth`
- `codeformer.pth`
- `GFPGANv1.4.pth`
- `detection_Resnet50_Final.pth`
- `parsing_parsenet.pth`

如果你希望手动放置权重，请查看 [weights/README.zh-CN.md](weights/README.zh-CN.md)。

## GUI 亮点

当前 GUI 包含：

- 三栏主界面，分别用于文件列表、预览和导出设置
- 内置预设和智能参数推荐
- 队列导出和批量导出
- 处理过程中的实时阶段与进度更新
- 历史记录和最近文件追踪
- 环境诊断和日志面板
- 输出命名规则和导出配置导入/导出
- 加速器、tile 和 batch size 等性能控制项

## 性能说明

当前的性能优化策略是在不改变最终质量路径的前提下提升吞吐量。

已实现的关键优化包括：

- FFmpeg 与 Python 之间使用原始帧流式传输，而不是 PNG 序列中转
- 异步三阶段处理流水线
- 基于显存和模型路径的动态 batch-size 选择
- `weights/trt_cache` 下的专用 TensorRT 引擎缓存目录
- 推理运行时就绪时优先走 GPU，不可用时自动回退到 baseline

`fast` 质量模式会更激进地关闭高成本后处理，以提升处理速度。

## 贡献者常用命令

运行测试：

```powershell
pytest
```

运行 Ruff：

```powershell
ruff check .
```

## 备注

- 该仓库以 Windows 为优先平台。
- CLI 和 GUI 都依赖可访问的 FFmpeg。
- baseline 后端仍然适合作为诊断和兜底路径，但不是主要画质目标。
- 当前代码库中的主画质路径是 Real-ESRGAN，加上可选的人脸修复和后处理。