# ClearVid 模型权重说明

当前项目已经具备：

- 5090 兼容的 PyTorch 运行环境
- Real-ESRGAN、GFPGAN、CodeFormer 相关 Python 依赖

如果 GUI 顶部状态显示 Real-ESRGAN 不可用，常见原因通常是运行时环境或依赖链异常。

如果只是缺少默认权重，当前版本会在首次实际执行 Real-ESRGAN 导出时自动下载默认权重，不需要你手动放置。

如果启用了人脸修复，当前版本也会在首次运行时自动下载 CodeFormer 和 facelib 所需权重。

## 推荐目录

请将超分模型权重放到以下目录：

- weights/realesrgan/

人脸增强相关权重后续建议放到：

- weights/codeformer/
- weights/facelib/

## 当前建议优先准备的 Real-ESRGAN 权重

程序默认会优先使用以下权重之一：

- RealESRGAN_x4plus.pth
- realesr-general-x4v3.pth

放置示例：

```text
weights/
  realesrgan/
    RealESRGAN_x4plus.pth
```

## 当前状态说明

当前版本的工作流是：

1. 工具可以直接启动并通过 GUI 导出视频
2. 自动后端在 Real-ESRGAN 可用时会优先选择 Real-ESRGAN
3. 如果默认权重不存在，首次运行 Real-ESRGAN 时会自动下载 `realesr-general-x4v3.pth`
4. 如果启用了人脸修复，首次运行时会自动下载 `weights/codeformer/codeformer.pth` 以及 `weights/facelib/` 下的人脸检测与解析权重
5. 你也可以手动放入其他 `.pth` 权重文件覆盖默认选择
