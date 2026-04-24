# MSVD 标注划分（split_annotations_by_splits.py）

## 脚本位置

```bash
datasets/MSVD/split_annotations_by_splits.py
```

## 前置条件

当前目录下已经存在：

- `annotations_preprocessed.txt`  
  由 `preprocess_annotations.py` 生成，格式：
  ```text
  video_id sen_id caption
  ```
- 官方提供的三个划分文件：
  - `train.txt`
  - `val.txt`
  - `test.txt`

它们的格式均为：

```text
video_id caption
```

即：第一列是 `video_id`，后面是 caption 文本。

## 功能

- 从 `annotations_preprocessed.txt` 中筛选出属于 train/val/test 三个 split 的行。
- **仅根据 video_id** 判断所属的 split。
- **不重新分配 `sen_id`**，直接沿用 `annotations_preprocessed.txt` 中已有的句子 ID。

生成三个输出文件：

- `train_preprocessed.txt`
- `val_preprocessed.txt`
- `test_preprocessed.txt`

格式与 `annotations_preprocessed.txt` 一致：

```text
video_id sen_id caption
```

## 使用方法

在项目根目录下：

```bash
cd datasets/MSVD
python split_annotations_by_splits.py
```

运行完成后，当前目录会额外生成：

- `train_preprocessed.txt`
- `val_preprocessed.txt`
- `test_preprocessed.txt`

## 典型训练脚本中使用方式（示例）

在训练 MSVD 时，可以将 `annotations_path` 指向对应的子集，例如：

```bash
# 训练集
--annotations_path ../datasets/MSVD/train_preprocessed.txt

# 验证集
--annotations_path ../datasets/MSVD/val_preprocessed.txt

# 测试集
--annotations_path ../datasets/MSVD/test_preprocessed.txt
```

实际使用时，可以在 dataloader 中根据 `split` 参数选择不同的文件路径，或在外部脚本中按需传入。 
