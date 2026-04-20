
# `load_tokenizers.py` 使用说明（贴合当前实现 / 中文）

> 适用文件：`load_tokenizers.py`。本文只描述**实际生效的代码路径**，不包含注释掉的未启用代码。:contentReference[oaicite:0]{index=0}

---

## 1. 模块功能概述

- **缓存优先加载** Hugging Face 版 **CLIP 分词器**（`openai/clip-vit-base-patch32`）。本地命中即用；未命中则联网下载到同一 `cache_dir`，之后可离线复用。:contentReference[oaicite:1]{index=1}
- **Windows 专用便捷代理**：仅在 Windows 平台，且未外部设置代理时，为开发便捷设置本地代理；Linux/macOS 不改动网络环境。:contentReference[oaicite:2]{index=2}
- **工程化 padding 约定**：提供 `CLIPTokenizer_Custom`，可将 **`pad_token_id` 统一为 `0`**（与许多下游管线一致），并带自检的编码/解码测试。:contentReference[oaicite:3]{index=3}

---

## 2. 主要 API

### 2.1 `load_clip_tokenizer(cache_dir: str) -> transformers.CLIPTokenizer`
- **作用**：加载 `openai/clip-vit-base-patch32` 的分词器，优先本地，必要时下载到 `cache_dir`。:contentReference[oaicite:4]{index=4}
- **平台行为**：
  - **Windows**：用 `os.environ.setdefault` 尝试设定 `HTTP_PROXY/HTTPS_PROXY`（不覆盖你已有设置）；同时将 `cache_dir` 指向 Windows 的固定路径，便于先在 Windows 缓存后拷到 Linux。:contentReference[oaicite:5]{index=5}
  - **Linux/macOS**：不改环境变量。:contentReference[oaicite:6]{index=6}
- **缓存策略**：先 `local_files_only=True` 读取，失败则回退到在线下载（`local_files_only=False`）至同一 `cache_dir`。:contentReference[oaicite:7]{index=7}
- **返回**：`CLIPTokenizer`，默认最大长度 **77**（CLIP 规范，详见下文的测试用例）。:contentReference[oaicite:8]{index=8}

### 2.2 `class CLIPTokenizer_Custom(swap_pad_token: bool = True)`
- **作用**：在标准 `CLIPTokenizer` 之上进行轻量封装：
  1) 内部调用 `load_clip_tokenizer()` 加载分词器；打印特殊 token 信息；  
  2) 如 `swap_pad_token=True`，调用 `_swap_pad_token_with_id_0()`，将 **`pad_token_id` 交换为 `0`**。:contentReference[oaicite:9]{index=9}
- **关键实现**（均在类内完成）：  
  - 重新构造 `vocab`，把 `pad_token` 的 ID 设为 0，并将原本 **ID=0 的 token** 交换到原 `pad_token_id`；同步更新 `ids_to_tokens`、`special_tokens_map['pad_token']`、`pad_token` 与 `pad_token_id`；最后打印校验信息。:contentReference[oaicite:10]{index=10}
  - 转发常用方法：`encode`、`encode_plus`、`get_vocab`、`convert_tokens_to_ids`、`convert_ids_to_tokens` 等。:contentReference[oaicite:11]{index=11}
  - `vocab_size` 可读/可写属性（写入仅更新缓存值，便于你在外部添加 token 后立即反映词表大小）。:contentReference[oaicite:12]{index=12}
  - `test_encoding(texts)`：对输入文本做**单条**与**批量**编码，强制 `padding="max_length"`, `max_length=77`, `truncation=True`, `return_tensors="pt"`；打印 `input_ids/attention_mask`、解码结果，并**检查 PAD 是否为 0**。:contentReference[oaicite:13]{index=13}

---

## 3. 输入 / 输出示例与形状（Shape）

### 3.1 单条编码（固定到 77）
```python
from load_tokenizers import CLIPTokenizer_Custom

tok = CLIPTokenizer_Custom(swap_pad_token=True)
enc = tok.encode_plus(
    "一个简短的句子。",
    padding="max_length",   # 固定长度
    max_length=77,          # CLIP 最大长度
    truncation=True,        # 超长截断
    return_tensors="pt"
)

print(enc["input_ids"].shape)       # torch.Size([1, 77])
print(enc["attention_mask"].shape)  # torch.Size([1, 77])
print(enc["input_ids"][0][-5:])     # 观察末尾是否为 0（PAD）
````

* **输出张量**：

  * `input_ids`：`[B, L] = [1, 77]`
  * `attention_mask`：`[1, 77]`（非 PAD 位置为 1，PAD 位置为 0）
* 若启用了 `swap_pad_token=True`，PAD 的 ID 将是 **0**，掩码与 0 对齐。

### 3.2 批量编码（固定到 77）

```python
texts = ["你好 CLIP", "这是一段稍长一些的文本，用来测试批量编码。"]
batch = tok.batch_encode_plus(
    texts,
    padding="max_length",
    max_length=77,
    truncation=True,
    return_tensors="pt"
)
print(batch["input_ids"].shape)      # torch.Size([2, 77])
print(batch["attention_mask"].shape) # torch.Size([2, 77])
```

* **输出张量**：

  * `input_ids`：`[B, 77]`
  * `attention_mask`：`[B, 77]`
* 可用 `attention_mask==0` 的位置检查对应 `input_ids` 是否为 0（PAD）。

---

## 4. 使用方式

### 4.1 作为库调用（推荐）

```python
from load_tokenizers import CLIPTokenizer_Custom

tokenizer = CLIPTokenizer_Custom(swap_pad_token=True)  # 将 PAD 规范为 0
ids = tokenizer.encode("A short sentence.", add_special_tokens=True)
```

* `swap_pad_token=True` 会将分词器内部的 `pad_token_id` 交换为 0，随后编码时使用 `padding="max_length"` 即会输出 0 作为 PAD 值。

### 4.2 作为脚本运行（自检）

```bash
python load_tokenizers.py
```

* 程序会实例化 `CLIPTokenizer_Custom()` 并对两段文本执行 `test_encoding`：打印单条与批量编码结果、是否用 0 作为 PAD、以及解码正确性。

---

## 5. 跨平台与缓存策略

* **Windows**：若你未设置代理变量，函数会以 `setdefault` 方式设置 `HTTP_PROXY/HTTPS_PROXY` 为本地端口，同时将 `cache_dir` 重定向到 Windows 固定路径，方便先在 Windows 缓存好再拷到 Linux 使用。
* **Linux/macOS**：不改动网络环境变量；`cache_dir` 使用你传入的路径。
* **缓存优先**：首次尝试 `local_files_only=True`（纯本地）；若失败，自动切换到联网下载并写入同一 `cache_dir`，之后可离线复用。

> 建议将 `cache_dir` 设为稳定可写目录（如 `./models/clip_tokenizer/`），并在缓存完后可选设置 `HF_HUB_OFFLINE=1` 与 `TRANSFORMERS_OFFLINE=1` 来强制离线。

---

## 6. 与下游训练/推理的对齐建议

* **Padding 与掩码**：启用 `swap_pad_token=True` 后，PAD=0，`attention_mask=0` 正好与 0 对齐，collate 与 mask 处理清晰一致。
* **损失函数**：若使用 `ignore_index` 策略，确保不会把合法类别 0 也忽略掉（通常 `ignore_index` 用于标签空间，而 `input_ids` 的 PAD=0 影响的是语言建模输入侧）。
* **长度上限**：CLIP 文本通用最大长度为 77；固定批可用 `padding="max_length", max_length=77, truncation=True`。

---

## 7. 常见问题（FAQ）

**Q1：为什么我看到“从本地加载失败，尝试从网络下载”的打印？**
A：函数先以 `local_files_only=True` 读取本地缓存，若未命中或损坏，会切换到在线下载流程写入同一 `cache_dir`。

**Q2：如何确认 PAD 已经是 0？**
A：运行 `python load_tokenizers.py`，查看 `test_encoding` 的输出；或手工打印 `tokenizer.pad_token_id` 是否为 0，并检查固定长度编码中 `attention_mask==0` 的位置对应的 `input_ids` 是否为 0。

**Q3：离线环境如何使用？**
A：先在能联网的机器上缓存到目标 `cache_dir`，再拷贝到离线环境；之后 `local_files_only=True` 即可纯离线加载。

---

## 8. 最小验证片段

```python
from load_tokenizers import CLIPTokenizer_Custom
tok = CLIPTokenizer_Custom(swap_pad_token=True)

enc = tok.encode_plus(
    "hello world",
    padding="max_length",
    max_length=77,
    truncation=True,
    return_tensors="pt"
)
print("pad_token_id:", tok.pad_token_id)      # 期望 0
print(enc["input_ids"][0][-8:])               # 末尾应为 0
print(enc["attention_mask"][0][-8:])          # 对应应为 0
```

（运行 `python load_tokenizers.py` 可查看更完整的单条/批量测试打印。）

---

