# ViT Nuitka 项目指南

本项目提供 Vision Transformer (ViT) 的训练与推理实现，并通过 Nuitka 编译为仅暴露 `train_vit` 与 `infer_vit` 两个 API 的二进制模块，便于分发与集成。

---

## 环境准备
- Python 3.12（与示例产物 `api.cpython-312-*.so` 一致）
- 依赖安装：
  ```bash
  python -m pip install torch torchvision numpy pillow
  ```

---

## 1. 使用 Nuitka 重新编译
1. 安装 Nuitka（若尚未安装）：
   ```bash
   python -m pip install nuitka
   ```
2. 在项目根目录执行打包脚本（默认输出到 `dist/`）：
   ```bash
   bash build_nuitka.sh
   ```
   - 可通过附加参数自定义 Nuitka 选项，例如：
     ```bash
     bash build_nuitka.sh dist --onefile
     ```
3. 编译产物仅包含 `train_vit` 与 `infer_vit` 两个对外 API。示例调用（假设输出为 `dist/api.so`）：
   ```python
   import api

   api.train_vit(epochs=1, batch_size=32)
   results = api.infer_vit("tests/image.png", checkpoint="outputs/best.pth")
   ```
4. 发布时只需分发 `dist/` 中的编译产物与运行所需依赖，无需附带源代码。

---

## 2. 二进制模块使用说明

### 2.1 模块导入
将 Nuitka 生成的二进制文件（如 `api.cpython-312-darwin.so`）放置到可导入路径（项目根目录或自定义 `site-packages`），即可通过 `import api` 使用。

### 2.2 暴露的 API
- `train_vit(**options) -> float`：执行完整训练流程，返回验证集最佳准确率。
- `infer_vit(inputs, **options) -> Dict[str, List[Tuple[str, float]]]`：对图像执行推理，返回每张图像的 Top-k 预测结果。

其余实现均已封装在二进制中，无法直接访问源代码。

### 2.3 训练示例
```python
import api

best_acc = api.train_vit(
    data_path="./data",
    epochs=5,
    batch_size=64,
    model="vit_small",
    amp=True,
    output_dir="./outputs",
)
print(f"Best validation accuracy: {best_acc * 100:.2f}%")
```

常用可选参数：
- `data_path`：数据集根目录，默认 `./data`。
- `num_classes`：类别数，未指定时将根据 CIFAR10 数据自动推断。
- `device`：计算设备（如 `"cuda"`、`"cpu"`、`"mps"`），默认自动选择。
- `resume`：恢复训练使用的 checkpoint 路径。

### 2.4 推理示例
```python
import api

results = api.infer_vit(
    ["examples/dog.png", "examples/cat.jpg"],
    checkpoint="outputs/best.pth",
    model="vit_small",
    image_size=224,
    topk=3,
)

for path, predictions in results.items():
    print(path)
    for idx, (label, score) in enumerate(predictions, start=1):
        print(f"  Top {idx}: {label} ({score * 100:.2f}%)")
```

常用可选参数：
- `class_names`：包含类别名称的文本文件路径（每行一个类），默认使用内置的 CIFAR10 标签。
- `amp`：在 CUDA 上启用混合精度推理。
- `quiet`：默认为 `True`，如需打印详细信息可设为 `False`。

### 2.5 注意事项
- 训练和推理均依赖 Torch 与 TorchVision，需确保版本兼容。
- 推理未提供 `checkpoint` 时将使用随机初始化模型（仅适用于功能验证）。
- 训练过程中会在 `output_dir` 下保存 `last.pth` 与 `best.pth` 供后续推理加载。
- 若需重新构建或更新二进制产物，请参考「使用 Nuitka 重新编译」章节。
