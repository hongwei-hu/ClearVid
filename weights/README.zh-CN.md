# ClearVid 权重目录说明

本目录用于存放 ClearVid 运行时所需的模型权重，以及 TensorRT 相关缓存。

仓库会保留目录结构和说明文件，但不应提交实际模型文件或本机生成缓存。

## 目录用途

推荐目录结构如下：

```text
weights/
  codeformer/
  facelib/
  gfpgan/
  realesrgan/
  trt_cache/
```

各目录的用途：

- `weights/realesrgan/`: Real-ESRGAN 超分模型权重
- `weights/codeformer/`: CodeFormer 人脸修复权重
- `weights/gfpgan/`: GFPGAN 人脸修复权重
- `weights/facelib/`: 人脸检测与解析辅助权重
- `weights/trt_cache/`: TensorRT 引擎缓存和相关生成物

## 自动下载行为

当前版本支持按需自动下载常用权重。

首次实际执行导出时，如果缺少所需权重，程序会根据当前配置自动下载对应文件。

当前会按需下载的常见文件包括：

- `realesr-general-x4v3.pth`
- `RealESRGAN_x4plus.pth`
- `codeformer.pth`
- `GFPGANv1.4.pth`
- `detection_Resnet50_Final.pth`
- `parsing_parsenet.pth`

如果启用了 TensorRT，加速过程中还可能在 `weights/trt_cache/` 下生成本机相关缓存文件。

## 手动放置权重

如果你希望提前准备权重，也可以手动放置到对应目录。

示例：

```text
weights/
  realesrgan/
    realesr-general-x4v3.pth
    RealESRGAN_x4plus.pth
  codeformer/
    codeformer.pth
  gfpgan/
    GFPGANv1.4.pth
  facelib/
    detection_Resnet50_Final.pth
    parsing_parsenet.pth
```

## Git 提交策略

建议提交到仓库的内容：

- 本说明文件
- 各子目录的占位文件

不建议提交到仓库的内容：

- `.pth` 模型权重
- `.onnx` 文件
- `.engine` 文件
- `weights/trt_cache/` 下的缓存文件
- 其他本机生成的中间产物

## 相关说明

项目总说明请查看仓库根目录下的 README。

如果你只是普通使用者，通常不需要手动下载权重；保持目录结构存在即可，程序会在需要时自动补齐。
