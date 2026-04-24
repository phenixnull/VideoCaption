# RawVideoExtractor 使用文档

`RawVideoExtractor` 是一个基于 OpenCV (CV2) 的视频帧提取工具，用于预处理视频数据并转换为 PyTorch 张量格式。它支持多种视频采样和转换策略，适用于视频分析和深度学习任务。

## 安装依赖

```bash
pip install torch==1.7.1 torchvision
pip install opencv-python
pip install Pillow
pip install numpy
```

## 基本使用

```python
from raw_video_extractor import RawVideoExtractor

# 初始化提取器
extractor = RawVideoExtractor(
    size=224,                # 输出图像尺寸
    framerate=1,             # 采样帧率(fps模式下)
    sample_type='k_split',   # 采样类型: 'fps' 或 'k_split'
    k_segments=20,           # k_split模式下的段数
    extract_loc='start'      # 提取位置: 'start', 'mid', 或 'end'
)

# 从视频文件中提取帧
video_input, shape = extractor.get_video_data(
    video_path='path/to/video.mp4',
    start_time=None,         # 可选: 开始时间(秒)
    end_time=None,           # 可选: 结束时间(秒)
    patch=0,                 # 可选: 图像分块数量(0表示不分块)
    overlapped=0             # 可选: 分块重叠率(0-1之间)
)

# 处理原始视频数据
processed_tensor = extractor.process_raw_data(video_input['video'])

# 可选: 调整帧顺序
reordered_tensor = extractor.process_frame_order(
    video_input['video'],
    frame_order=0            # 0: 正常顺序, 1: 反向顺序, 2: 随机顺序
)
```

## 参数详解

### 初始化参数

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| size | int | 224 | 处理后图像的宽高(像素) |
| framerate | int | -1 | 采样帧率(fps模式下) |
| transform_type | int | 0 | 转换类型: 0=默认, 1=物体/iou/时序/动作, 2=i3d |
| sample_type | str | 'k_split' | 采样类型: 'fps'(按帧率) 或 'k_split'(分段) |
| k_segments | int | 20 | k_split模式下的段数 |
| extract_loc | str | 'start' | k_split模式下提取位置: 'start'(开始), 'mid'(中间), 'end'(结束), 'tsn'(随机采样) |
| random_seed | int | 42 | TSN模式下的随机种子，确保结果可复现 |

### 视频提取参数

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| video_path | str | 必填 | 视频文件路径 |
| start_time | int | None | 开始处理的时间(秒) |
| end_time | int | None | 结束处理的时间(秒) |
| patch | int | 0 | 图像分块数量(0表示不分块, 2表示2x2, 3表示3x3, 以此类推) |
| overlapped | float | 0 | 分块之间的重叠率(0-1之间) |

### 帧顺序参数

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| frame_order | int | 0 | 0: 正常顺序, 1: 反向顺序, 2: 随机顺序 |

## 采样模式说明

### FPS模式 (`sample_type='fps'`)

按指定的帧率(`framerate`)从视频中均匀采样帧。

- 如果 `framerate=1`, 则每秒提取1帧
- 如果 `framerate=2`, 则每秒提取2帧
- 如果 `framerate=-1` 或 `framerate=0`, 则使用原视频帧率

### K分段模式 (`sample_type='k_split'`)

将视频平均分为K段，每段提取一帧。

- `k_segments`: 指定分段数量
- `extract_loc`: 指定每段中的提取位置
  - `'start'`: 提取每段开始处的帧
  - `'mid'`: 提取每段中间的帧
  - `'end'`: 提取每段结束处的帧
  - `'tsn'`: TSN模式，在每段内随机采样一帧（Temporal Segment Networks风格）

## 输出格式

`get_video_data()` 返回两个值:

1. `video_input`: 字典，包含键 `'video'` 对应的视频张量
2. `shape`: 张量的形状

视频张量的形状取决于是否使用分块:

- 无分块 (`patch=0`): `[frames, channels, height, width]`
- 有分块 (`patch>0`): `[frames, patches_num, channels, patch_height, patch_width]`

其中，`patches_num = patch * patch`（例如，`patch=3` 时，`patches_num=9`）

## 应用示例

### 示例1: 提取视频关键帧

```python
extractor = RawVideoExtractor(sample_type='k_split', k_segments=10, extract_loc='mid')
frames, _ = extractor.get_video_data('video.mp4')
# frames['video'] 包含10个关键帧
```

### 示例2: 高密度采样

```python
extractor = RawVideoExtractor(sample_type='fps', framerate=5)
frames, _ = extractor.get_video_data('video.mp4')
# frames['video'] 包含每秒5帧的视频
```

### 示例3: 带分块的视频处理

```python
extractor = RawVideoExtractor(sample_type='k_split', k_segments=8)
frames, _ = extractor.get_video_data('video.mp4', patch=2, overlapped=0.2)
# frames['video'] 包含8个关键帧，每帧分为2x2=4个块，块间有20%重叠
```

### 示例4: TSN随机采样模式

```python
extractor = RawVideoExtractor(
    sample_type='k_split', 
    k_segments=16, 
    extract_loc='tsn',
    random_seed=42
)
frames, _ = extractor.get_video_data('video.mp4')
# frames['video'] 包含16个随机采样的帧，每个段内随机选择一帧
# 使用固定随机种子确保结果可复现
```

## TSN模式详解

TSN (Temporal Segment Networks) 模式是一种特殊的采样策略，它结合了k_split分段和随机采样的优点：

1. **分段策略**: 首先将视频按照 `k_segments` 参数平均分为K段
2. **随机采样**: 在每个段内随机选择一帧进行提取
3. **可复现性**: 通过 `random_seed` 参数确保随机结果可复现

### TSN模式的优势

- **时间覆盖**: 确保从视频的整个时间跨度中采样
- **多样性**: 每个段内的随机采样增加了帧选择的多样性
- **可复现**: 固定随机种子保证实验结果的一致性
- **适用性**: 特别适合动作识别和视频分类任务

### 使用建议

- 对于短视频（<30秒），建议使用 `k_segments=8-16`
- 对于长视频（>1分钟），建议使用 `k_segments=16-32`
- 在训练时可以使用不同的 `random_seed` 增加数据多样性
- 在测试时应使用固定的 `random_seed` 确保结果可复现

### 示例4: TSN随机采样模式

```python
extractor = RawVideoExtractor(
    sample_type='k_split', 
    k_segments=16, 
    extract_loc='tsn',
    random_seed=42
)
frames, _ = extractor.get_video_data('video.mp4')
# frames['video'] 包含16个随机采样的帧，每个段内随机选择一帧
# 使用固定随机种子确保结果可复现
```

您提出了两个很好的问题：

1. 关于 `k_split` 模式下帧数不足的情况：
   
   当视频总帧数不足 k_segments (例如默认的20帧) 时，代码会按实际情况处理。在这种情况下，分段大小 `segment_size = frameCount // k` 会变得很小，甚至可能为0。

   - 如果 `segment_size` 为0，那么某些段可能会指向相同的帧
   - 在极端情况下，如果视频只有1帧，所有段都会提取这同一帧
   - 代码没有专门处理帧数不足的边界情况，所以当帧数远小于k时，可能会重复提取相同的帧

2. 关于 `fps` 采样模式的帧数：
   
   是的，您的理解完全正确。在 `fps` 采样模式下，每个视频提取的帧数是不固定的，而是取决于：
   
   - 视频的总时长（秒）
   - 设定的采样帧率 `framerate`
   
   提取的帧数 = 视频总秒数 × 采样帧率
   
   例如：
   - 30秒视频，`framerate=1`，会提取30帧
   - 10秒视频，`framerate=2`，会提取20帧
   - 5秒视频，`framerate=5`，会提取25帧

因此，如果您需要固定数量的帧，应使用 `k_split` 模式；如果需要按时间间隔均匀采样且不在意总帧数，则使用 `fps` 模式更合适。