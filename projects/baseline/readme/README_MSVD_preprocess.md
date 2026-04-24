# MSVD 标注预处理（preprocess_annotations.py）

## 脚本位置

```bash
datasets/MSVD/preprocess_annotations.py
```

## 功能

- 读取原始 `annotations.txt`（格式：`video_id caption`）。
- 为每个视频内部的 caption 添加句子 ID（`sen_id`），并写出：
  - `annotations_preprocessed.txt`，格式：
    ```text
    video_id sen_id caption
    -4wsuPCjDBc_5_15 0 a squirrel is eating a peanut
    -4wsuPCjDBc_5_15 1 a chipmunk is eating
    ...
    ```
- **注意**：只处理原始 `annotations.txt`，不负责 train/val/test 的划分。

## 使用方法

在项目根目录下：

```bash
cd datasets/MSVD
python preprocess_annotations.py
```

运行完成后，当前目录会生成：

- `annotations_preprocessed.txt`

## 依赖

- Python 标准库（`os`, `collections`），不依赖第三方包。
