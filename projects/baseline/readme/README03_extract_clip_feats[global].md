
# `extract_clip_feats.py` 说明书（**贴合当前代码**）

> 适用文件：`extract_clip_feats.py`（特征抽取主脚本）、`dataset_msrvtt_raw.py`（原始帧数据集）、`models.py`（仅使用其中的 `load_clip_model`）。本文档严格依据这三份源码撰写。:contentReference[oaicite:0]{index=0} :contentReference[oaicite:1]{index=1} :contentReference[oaicite:2]{index=2}

---

## 1. 功能与数据流（一图流）

**目标**：批量抽取视频的 CLIP 视觉特征，并在**抽取阶段就对齐序列长度**，输出统一的 `(feats_padded, mask)`：  
- `feats_padded`: `(max_frames, d_model)`  
- `mask`: `(max_frames,)`，`1=有效帧，0=padding`。  

**流程**：
1) 读取原始视频，按配置抽帧 → 得到 `video: [T, 1, 3, H, W]` 与 `video_mask: [T]`；T ≤ `max_frames`。
2) 过滤无效帧（`mask==1`），按批喂给 `clip_model.encode_image`，拼成 `output: [T_valid, d_model]`。
3) **零填充**到 `feats_padded: [max_frames, d_model]`，与 `original_mask: [max_frames]` 成对保存到 `pickle`。

> 该对齐在 worker 内完成，最后主进程**合并分片**写出一个总的 `pickle` 文件。

---

## 2. 输入侧：原始帧数据集（MSRVTT 示例）

构造：`MSRVTT_Raw_Dataset(csv_path, videos_dir, ..., max_frames, frame_order, slice_framepos, transform_type, extract_type, ...)`

- `__getitem__` 返回：  
  - `video_id: str`  
  - `video: np.ndarray`，形如 `[[T, 1, 3, H, W]]`（外层 batch=1），T ≤ `max_frames`  
  - `video_mask: np.ndarray`，形如 `[[T]]`（外层 batch=1），值域 `{0,1}`（1=有效帧）  
  内部会根据 `extract_type/k_split/fps`、`slice_framepos(0/1/2)`、`frame_order(0/1/2)` 对帧进行采样与排序，最终将有效帧数记录到 `video_mask`。

**抽帧与预处理关键点**：  
- `RawVideoExtractor` 负责读取与变换（分辨率由 `image_resolution` 控制），并将帧整理为 `[frames, 1, 3, H, W]`。
- 若原始帧多于 `max_frames`，可选择**头/尾/均匀**三种裁剪策略；不足则由 `video_mask` 标明有效段。

---

## 3. 模型加载：`models.load_clip_model`

在主进程里**仅调用一次**以加载“主模型”（放在 `args.device` 上），随后 `state_dict` 会被拷到 CPU 供各 worker 复用，避免重复下载与不一致。

**函数签名（仅与抽取相关）**：
```python
clip_model, preprocess = load_clip_model(
    model_name="ViT-B/32",
    device="cuda:0",
    local_files_only=True,
    cache_dir="/mnt/.../models/clip_models/"
)
````

* **本地优先**：函数默认“尝试本地 → 若失败则尝试网络下载至 `cache_dir`”（即使 `local_files_only=True` 也会在本地失败后尝试下载）。成功后打印“CLIP模型缓存位置”。
* **Windows**：函数内部会将 `cache_dir` 重定向到固定 Windows 路径，便于**先在 Windows 缓存后拷到 Linux**；Linux/macOS 使用传入的 `cache_dir`。

> ⚠️ **重要提示（当前源码行为）**：`models.py` 文件末尾**会在导入时立即执行一次 `load_clip_model()`**，因此仅仅 `import load_clip_model` 也会触发一次模型加载与日志打印；随后 `extract_clip_feats.py` 再调用一次，故你可能看到**重复两条**“加载本地CLIP模型 ...”日志。若不需要该副作用，建议在 `models.py` 中移除（或注释）那两行。

---

## 4. 并行与切分

* 设备列表：`--work_devices`（如 `"0,0,1,1,2,2,2,3,3,3"`）决定**worker 数与各自运行设备**；允许同一 GPU 多个 worker，也可用 `cpu`。未传时默认“**每张可见 GPU 各 1 个**”。
* 数据切片：将数据集按 worker 均分，**前若干切片+1** 以平衡。每个 worker 输出一个临时 part，结束后主进程合并。

---

## 5. 前向与对齐（逐样本）

以单个样本为例（worker 内）：

1. **取样本并过滤无效帧**

   ```python
   video_id, video, video_mask = raw_dataset[i]
   tensor = video[0]                    # (T, 1, 3, H, W)
   original_mask = video_mask[0].copy() # (max_frames,)
   tensor = tensor[video_mask[0] == 1]  # (T_valid, 1, 3, H, W)
   tensor = tensor.view(T_valid, 3, H, W)
   ```

   说明：`original_mask` 记录原始有效帧位置，T\_valid ≤ `max_frames`。

2. **分批送入 CLIP**
   批大小由 `--process_frames_size` 控制，`encode_image` 输出 `[bs, d_model]`，将各批拼接成 `output: (T_valid, d_model)`。`d_model` 由骨干决定（如 `ViT-B/32 → 512`）。

3. **Padding 到 (max\_frames, d\_model)**

   ```python
   feats_padded = np.zeros((max_frames, d_model), dtype=output.dtype)
   feats_padded[:min(T_valid, max_frames)] = output[:min(T_valid, max_frames)]
   ```

   保存：`local_data[video_id] = (feats_padded, original_mask.astype(np.int64))`。

> 极端兜底：若某视频没有有效帧，则用映射表推断 `d_model` 并以全零特征占位（保证形状一致）。

---

## 6. 输出文件与示例

**文件类型**：`pickle`
**结构**：`Dict[str, Tuple[np.ndarray, np.ndarray]]`

* `data[video_id] = (feats_padded, mask)`
* `feats_padded`: `(max_frames, d_model)`；`float16`（GPU）或 `float32`（CPU）
* `mask`: `(max_frames,)`，`int64`，`1=有效帧`，`0=padding`。

**读取示例（最小可运行片段）**：

```python
import pickle, numpy as np

path = "datasets/MSRVTT/feats/ViT-B-32_k_split_ks12_features.pickle"
data = pickle.load(open(path, "rb"))

vid = next(iter(data))
feats_padded, mask = data[vid]

print(vid)                    # e.g. 'video1234'
print(feats_padded.shape)     # (max_frames, d_model), e.g. (12, 512)
print(feats_padded.dtype)     # float16 (GPU) / float32 (CPU)
print(mask.shape, mask.dtype) # (12,), int64
print(mask)                   # [1,1,1,1,0,0,...]
```

**拼 batch（送入下游模型）**：

```python
import torch
# 假设取 N 条样本
batch = [data[k] for k in list(data.keys())[:8]]
feats = torch.tensor(np.stack([x[0] for x in batch], axis=0))  # [B, max_frames, d_model]
mask  = torch.tensor(np.stack([x[1] for x in batch], axis=0))  # [B, max_frames]
```

---

## 7. 运行示例

### 单 GPU（或 CPU）

```bash
python extract_clip_feats.py \
  --device cuda:0 \
  --work_devices 0 \
  --dataset_type msrvtt \
  --dataset_dir ../datasets \
  --extract_type k_split --max_frames 12 \
  --pretrained_clip_name "ViT-B/32" \
  --process_frames_size 20
```

> 若不传 `--work_devices`，默认“**每张可见 GPU 各 1 个 worker**”；无 GPU 时回退 `cpu`。

### 多 GPU（不均匀并发）

```bash
python extract_clip_feats.py \
  --device cuda:0 \
  --work_devices 0,0,1,1,1,2,2,3 \
  --dataset_type msrvtt --dataset_dir ../datasets \
  --extract_type k_split --max_frames 12 \
  --pretrained_clip_name "ViT-B/32" \
  --process_frames_size 20
```

* 数据均分到各 worker，前若干切片 +1；各自输出临时 part，最终合并。

---

## 8. 参数速查（与源码一致）

| 参数                       | 位置         | 说明                                                        |
| ------------------------ | ---------- | --------------------------------------------------------- |
| `--dataset_type`         | 脚本参数       | 目前示例实现 `msrvtt`。                                          |
| `--dataset_dir`          | 脚本参数       | 构造 `MSRVTT/` 下的 `raw/`、`msrvtt.csv` 等路径，并在 `feats/` 下写结果。 |
| `--max_frames`           | 脚本参数 & 数据集 | 控制每视频抽帧上限与输出序列长度。                                         |
| `--slice_framepos`       | 脚本参数 & 数据集 | 超长裁剪策略：0 头 / 1 尾 / 2 均匀。                                  |
| `--frame_order`          | 脚本参数 & 数据集 | 帧顺序：0 顺序 / 1 逆序 / 2 随机。                                   |
| `--extract_type`         | 脚本参数 & 数据集 | `fps` 或 `k_split`。                                        |
| `--transform_type`       | 脚本参数 & 数据集 | 预处理风格（0=默认 CLIP 预处理）。                                     |
| `--pretrained_clip_name` | 脚本参数 & 模型  | 如 `"ViT-B/32"`；影响 `d_model`。                              |
| `--process_frames_size`  | 脚本参数       | 单次前向帧批大小，建议按显存调节。                                         |
| `--device`               | 脚本参数       | **主模型**加载设备（用于拷贝 `state_dict`）。                           |
| `--work_devices`         | 脚本参数       | 并行设备列表，决定 worker 数与各自设备。                                  |
| `--save_file`            | 脚本参数       | 若未指定，将自动拼接 `<model>_<type>_ks<max>_features.pickle`。      |

---

## 9. 性能与稳定性建议

* **先本地缓存权重**：把对应 `.pt` 放入 `clip_cache_dir` 后再运行，可完全离线且更稳。`load_clip_model` 会打印缓存位置。
* **调小前向批**：OOM 时可将 `--process_frames_size` 从 20 降到 8/4。
* **worker 分配**：同卡多 worker 会竞争显存/带宽，视机器情况调优 `work_devices`。

---

## 10. 常见问题（基于当前实现）

**Q1：为什么启动时会打印两次“加载本地CLIP模型 ...”？**
A：`models.py` 在文件末尾**会立刻调用一次 `load_clip_model()`**（导入即执行），而 `extract_clip_feats.py` 主流程又调用一次，故出现两次日志。可在 `models.py` 中移除该调用，避免副作用。

**Q2：`local_files_only=True` 是否绝不联网？**
A：当前实现中，即使传入 `True`，**本地加载失败时仍会尝试下载**（以提升易用性）；若你需要**严格离线**，请先手动将模型权重放入缓存目录。

**Q3：输出的 `d_model` 是多少？**
A：由骨干决定（如 `ViT-B/32 → 512`，`ViT-L/14 → 768` 等）；脚本内维护了一份映射表，且在无效帧极端情况下用于兜底形状。

---

## 11. 最小可复现清单

1. 准备数据目录：`../datasets/MSRVTT/{raw, msrvtt.csv, MSRVTT_data.json}`。
2. 将所需 CLIP 权重 `.pt` 放入：`/mnt/sda/Disk_D/zhangwei/projects/VideoCaption_Reconstruction/project/models/clip_models/`（或按需修改 `extract_clip_feats.py` 中的 `clip_cache_dir` 变量）。
3. 运行命令（单卡示例）：

```bash
python extract_clip_feats.py \
  --device cuda:0 \
  --work_devices 0 \
  --dataset_type msrvtt --dataset_dir ../datasets \
  --extract_type k_split --max_frames 12 \
  --pretrained_clip_name "ViT-B/32" \
  --process_frames_size 20
```

运行结束后，在 `MSRVTT/feats/` 下获得类似
`ViT-B-32_k_split_ks12_features.pickle` 的文件（或 `--save_file` 指定的路径）。

```
```
