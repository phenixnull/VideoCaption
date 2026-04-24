# MSVD 数据目录说明（datasets/MSVD）

本目录用于存放 **MSVD (YouTube2Text)** 数据的原始文件、官方划分以及预处理后的标注与特征。

---

## 下载链接

- 标注（captions）下载链接：

  ```sh
  https://www.kaggle.com/datasets/vtrnanh/msvd-dataset-corpus?resource=download
  ```

- 视频（YouTubeClips）下载链接：

  ```sh
  https://www.cs.utexas.edu/~ml/clamp/videoDescription/YouTubeClips.tar
  ```

## 目录结构与文件说明

```text
MSVD/
  YouTubeClips/                # 解压后的原始视频（.avi），来自 YouTubeClips.tar
  YouTubeClips.tar             # 官方提供的视频压缩包
  archive.zip                  # Kaggle 语料压缩包（包含 video_corpus.csv 等）

  video_corpus.csv             # Kaggle 提供的原始多语言标注语料

  annotations.txt              # 从 video_corpus.csv 提取/整理后的原始标注（video_id caption）
  annotations_preprocessed.txt # 在 annotations.txt 基础上，加入句子 ID（video_id sen_id caption）

  train.txt                    # 官方/约定的 train 划分（video_id caption）
  val.txt                      # 官方/约定的 val 划分（video_id caption）
  test.txt                     # 官方/约定的 test 划分（video_id caption）

  train_preprocessed.txt       # 由 annotations_preprocessed.txt 结合 train.txt 过滤后的子集
  val_preprocessed.txt         # 由 annotations_preprocessed.txt 结合 val.txt 过滤后的子集
  test_preprocessed.txt        # 由 annotations_preprocessed.txt 结合 test.txt 过滤后的子集

  feats/                       # 预提取好的 CLIP 视频特征（pickle），训练/推理时直接读取

  preprocess_annotations.py    # 将 annotations.txt -> annotations_preprocessed.txt 的预处理脚本
  split_annotations_by_splits.py # 按 train/val/test.txt 的 video_id 划分 *_preprocessed.txt 的脚本

  extract_nouns.py             # 基于 GlobalAI API 的多阶段短语标注抽取脚本
  .keys                        # GlobalAI API keys（每行一个，可多 key 轮换）

  annotations/nouns/val/       # 输出目录示例（每个 video_id 一个 json）
  annotations/noun_vectors/val/ # 对应的 multi-hot 向量（每个 video_id 一个 .npy）
```

---

## 生成视频级别视觉短语标注（extract_nouns.py）

脚本 `extract_nouns.py` 对同一视频的多条 caption 做两阶段处理：

- **阶段 1（obj keys）**：抽取视频级对象键（名词/复合名词），去重。
- **阶段 2（phrases）**：对每个对象键，在所有 captions 中抽取该对象的**最小短语集合**（短语允许包含动词/修饰/副词），但必须遵循“单对象”原则。

### 单对象最小短语规则（核心）

对每个 `obj`：

- 短语必须包含该 `obj`（模式 A 为严格字面包含）。
- 短语只能包含 **1 个对象**（即当前 `obj`），不能包含其它实意名词对象。
- 不需要宾语：例如 `two men are hugging a lion` 对于 `men` 应输出 `two men are hugging`。
- 不要背景/地点短语：例如 `on the field`、`in the room` 等应被剔除；
  但如果 `obj` 本身就是地点名词（如 `field`, `room`），则在句中找与其最贴近的最小短语即可。

### 运行方式

在 `datasets/MSVD` 目录下运行：

```bash
# 单文件处理（单 key 顺序执行）
python extract_nouns.py --input val_preprocessed.txt --output annotations/nouns/val

# 单文件并行处理（多 key 并行，推荐）
python extract_nouns.py --input val_preprocessed.txt --output annotations/nouns/val --parallel

# 指定并行 worker 数量
python extract_nouns.py -i val_preprocessed.txt -o annotations/nouns/val -p -w 50

# 一键处理所有 split（train/val/test），并行模式
python extract_nouns.py --all --parallel

# 一键处理所有 split，指定 worker 数
python extract_nouns.py --all --parallel --workers 100
```

多次运行会跳过已存在的输出文件，便于断点续跑。

### 并行处理参数

| 参数 | 短参数 | 说明 |
|------|--------|------|
| `--parallel` | `-p` | 启用多 key 并行处理 |
| `--workers` | `-w` | 并行 worker 数量（默认=key 数量） |
| `--all` | `-a` | 自动处理 train/val/test 所有划分 |

**并行处理原理**：
- 使用 `ThreadPoolExecutor` 实现多线程并行
- 每个视频分配一个 API key（轮询分配）
- 线程安全的进度追踪和失败记录
- 自动跳过已处理的视频

### 参数：短语映射模式 A/B

- **模式 A（默认）**：严格字面匹配，短语必须字面包含 `obj`（大小写不敏感），并会做后处理裁剪去掉宾语/介词短语尾巴。
- **模式 B**：语义/指代匹配，允许同义词或指代（更全但更容易跑偏）。

```bash
# 默认（A）
python extract_nouns.py --phrase_mode A

# 语义/指代（B）
python extract_nouns.py --phrase_mode B
```

限制每个对象最多保留短语数量（防止输出过长导致 JSON 截断）：

```bash
python extract_nouns.py --max_phrases_per_obj 8
```

### 输出格式

每个视频输出一个 JSON：

`annotations/nouns/val/<video_id>.json`

结构：

```json
{
  "video_objs": {
    "obj1": ["phrase1", "phrase2"],
    "obj2": [],
    "...": []
  }
}
```

### 失败与日志

输出目录会生成：

- `_failed.txt`：兼容旧格式（简单三列）。
- `_failed_detail.tsv`：详细失败原因（`API_FAIL` / `PARSE_FAIL` / `VALIDATION_FAIL` / `OBJ_PHRASES_FAIL` 等）。

---

## Multi-hot 名词向量（noun_vectors）

脚本在保存 JSON 的同时，会自动生成对应的 **multi-hot 向量**，用于训练时快速索引名词。

### 输出位置

```text
annotations/noun_vectors/
  train/
    <video_id>.npy
  val/
    <video_id>.npy
  test/
    <video_id>.npy
```

### 向量格式

- **形状**：`[vocab_size]`，vocab_size = 49412（扩展后的 CLIP tokenizer）
- **类型**：`float32`
- **含义**：`vector[token_id] = 1.0` 表示该名词存在

### 示例

假设视频有名词 `["man", "car", "dog"]`：

```python
# man -> token_id=786
# car -> token_id=1203  
# dog -> token_id=2456

vector = np.zeros(49412)
vector[786] = 1.0   # man
vector[1203] = 1.0  # car
vector[2456] = 1.0  # dog
```

### 解码函数（调试用）

脚本内置解码函数，方便调试：

```python
from extract_nouns import decode_noun_vector, decode_noun_vector_file

# 方式 1：从文件解码
result = decode_noun_vector_file("annotations/noun_vectors/val/video_id.npy")
print(result['tokens'])  # ['man', 'car', 'dog', ...]
print(result['ids'])     # [786, 1203, 2456, ...]
print(result['count'])   # 3

# 方式 2：从向量解码
import numpy as np
vector = np.load("annotations/noun_vectors/val/video_id.npy")
result = decode_noun_vector(vector)
```

### Tokenizer 说明

使用的是扩展后的 `Tokenizer_M`（基于 CLIP tokenizer）：

- **原始词表**：49408 tokens
- **新增 tokens**：`[MASK]`, `[OBJ_CLS]`, `[OBJ_END]`, `[OBJ_SEP]`
- **扩展后词表**：49412 tokens
- **PAD token**：id=0

Tokenizer 路径：`project/models/tokenizer_m/tokenizer/`

---

## 完整处理流程

1. **准备 API keys**：在 `.keys` 文件中放入 GlobalAI API keys（每行一个）

2. **一键处理所有划分**：
   ```bash
   python extract_nouns.py --all --parallel --workers 100
   ```

3. **输出结构**：
   ```text
   annotations/
     nouns/
       train/*.json      # JSON 格式的名词-短语映射
       val/*.json
       test/*.json
     noun_vectors/
       train/*.npy       # Multi-hot 向量
       val/*.npy
       test/*.npy
   ```

4. **训练时使用**：
   ```python
   # 加载 JSON（用于文本生成）
   with open(f"annotations/nouns/{split}/{video_id}.json") as f:
       data = json.load(f)
       nouns = list(data["video_objs"].keys())
       phrases = data["video_objs"]
   
   # 加载向量（用于 loss 计算或条件输入）
   vector = np.load(f"annotations/noun_vectors/{split}/{video_id}.npy")
   ```
