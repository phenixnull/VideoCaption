import torch
import os
import sys
import json
from transformers import CLIPTokenizer


def load_clip_tokenizer(cache_dir: str = "/mnt/sda/Disk_D/zhangwei/projects/VideoCaption_Reconstruction/project/models/clip_tokenizer/"):
    """
    加载 HuggingFace 版的 CLIP 分词器（openai/clip-vit-base-patch32）。

    行为说明：
    - **Windows**：为了便于科学上网，默认仅在 Windows 下尝试设置本地代理
      HTTP(S)_PROXY= http://127.0.0.1:7890 。使用 `setdefault`，若你已在外部
      设置了代理变量，则不会被覆盖。
    - **Linux / macOS**：不改动任何代理相关环境变量，保持系统原状。
    - 缓存目录 `cache_dir`：优先从该目录读取；若缺失并且可联网，会自动下载到该目录。

    参数
    ----
    cache_dir : str
        模型与分词器的缓存目录（会自动创建）。建议传入稳定的可写路径。

    返回
    ----
    tokenizer : transformers.CLIPTokenizer
        可直接用于对文本进行分词，最大长度为 77 tokens（CLIP 规范）。
    """
    # --- 仅 Windows 下设置本地代理（不覆盖已存在的环境变量） ---
    # 先在windows开猫运行这个缓存到models/clip_tokenizer/然后上传linux
    if sys.platform.startswith("win"):  # 或者用 os.name == "nt"
        os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:7890")
        os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:7890")
        cache_dir = r"D:\Users\Administrator\Desktop\DeepLearningProjects_SELF\VideoCaption_Reconstruction\project\models\clip_tokenizer"

    # --- 保证缓存目录存在，并使用绝对路径（更稳） ---
    cache_dir = os.path.abspath(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    try:
        print("Tokenizer loading from local cache..,")
        # --- 加载分词器；会优先命中 cache_dir，本地没有则联网下载到该目录 ---
        tokenizer = CLIPTokenizer.from_pretrained(
            "openai/clip-vit-base-patch32",
            cache_dir=cache_dir,
            local_files_only=True  # 默认即为 False：允许缺失时下载
        )
    except:
        print("Loading tokenizer from cache failed, try to download from internet.")

        tokenizer = CLIPTokenizer.from_pretrained(
            "openai/clip-vit-base-patch32",
            cache_dir=cache_dir,
            local_files_only=False  # 默认即为 False：允许缺失时下载
        )
    return tokenizer


class CLIPTokenizer_Custom:
    """
    自定义Tokenizer类，继承CLIPTokenizer的功能，添加可写的vocab_size属性，
    并将pad_token的ID从49409更改为0
    """

    def __init__(self, swap_pad_token=True):
        # 加载原始的CLIPTokenizer
        print("Loading Tokenizer ...")
        self._tokenizer = load_clip_tokenizer()

        # 确保特殊token已被添加

        # 先添加标准的特殊token
        # self._tokenizer.add_special_tokens({
        #     'cls_token': '<|startoftext|>',
        #     'sep_token': '<|endoftext|>',
        #     'unk_token': '<|unknown|>',
        #     'pad_token': '<|pad|>',
        #     # 'mask_token': '<|mask|>'#临时用，记得替换掉
        # })

        # # 再添加自定义的特殊token
        # self._tokenizer.add_special_tokens({
        #     'additional_special_tokens': ['<|nouns|>', '<|verbs|>', '<|sentence|>']
        # })
        #
        # # 保存token id以便后续使用
        # self.noun_token_id = self._tokenizer.convert_tokens_to_ids('<|nouns|>')
        # self.verb_token_id = self._tokenizer.convert_tokens_to_ids('<|verbs|>')
        # self.sent_token_id = self._tokenizer.convert_tokens_to_ids('<|sentence|>')

        # 打印特殊token信息（交换前）
        # print("=== 初始Token设置 ===")
        for token_name, token_value in self._tokenizer.special_tokens_map.items():
            token_id = self._tokenizer.convert_tokens_to_ids(token_value)
            print(f"{token_name}: {token_value} (ID: {token_id})")

        # 如果需要交换pad_token和ID=0的token
        if swap_pad_token:
            self._swap_pad_token_with_id_0()

        # 存储可写的vocab_size
        self._vocab_size = len(self._tokenizer.get_vocab())

        print("\n当前词汇表大小:", self._vocab_size)

    def _swap_pad_token_with_id_0(self):
        """
        将pad_token的ID从49409更改为0，同时将ID为0的token更改为49409
        """
        # 获取当前pad_token和对应的ID
        pad_token = self._tokenizer.pad_token
        pad_token_id = self._tokenizer.convert_tokens_to_ids(pad_token)

        # 查找ID为0的token
        id_0_token = None
        for token, idx in self._tokenizer.get_vocab().items():
            if idx == 0:
                id_0_token = token
                break

        # print(f"\n交换前 pad_token: {pad_token} (ID: {pad_token_id})")
        # print(f"交换前 ID=0的token: {id_0_token}")

        # 创建一个新的词汇表
        new_vocab = {}
        for token, idx in self._tokenizer.get_vocab().items():
            if token == pad_token:
                new_vocab[token] = 0  # pad_token的ID改为0
            elif token == id_0_token:
                new_vocab[token] = pad_token_id  # 原ID=0的token改为pad_token的ID
            else:
                new_vocab[token] = idx  # 其他token保持不变

        # 完全替换tokenizer的词汇表
        self._tokenizer.vocab = new_vocab

        # 更新ids_to_tokens映射
        self._tokenizer.ids_to_tokens = {id: token for token, id in new_vocab.items()}

        # 重要：更新special_tokens_map中的pad_token
        self._tokenizer.special_tokens_map['pad_token'] = id_0_token

        # 更新pad_token和pad_token_id
        self._tokenizer.pad_token = id_0_token
        self._tokenizer.pad_token_id = 0

        # 打印交换后的特殊token信息
        # print("\n=== 交换后的Token设置 ===")
        for token_name, token_value in self._tokenizer.special_tokens_map.items():
            token_id = self._tokenizer.convert_tokens_to_ids(token_value)
            # print(f"{token_name}: {token_value} (ID: {token_id})")

        # 验证交换是否成功
        id_0_token_after = self._tokenizer.convert_ids_to_tokens(0)
        # print(f"\n交换后 ID=0的token: {id_0_token_after}")

        # 确认pad_token的ID是否为0
        pad_token_id_after = self._tokenizer.convert_tokens_to_ids(self._tokenizer.pad_token)
        # print(f"交换后 pad_token: {self._tokenizer.pad_token} (ID: {pad_token_id_after})")

        # 不要使用断言，改用打印警告
        if pad_token_id_after != 0:
            print(f"警告: 交换可能不完全成功，pad_token的ID不是0，而是{pad_token_id_after}")
        else:
            print("交换成功: pad_token的ID现在是0")

    # 将原始tokenizer的所有属性和方法转发到此类
    def __getattr__(self, name):
        # 如果当前类没有该属性，则委托给原始tokenizer
        if name != '_tokenizer' and hasattr(self._tokenizer, name):
            return getattr(self._tokenizer, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    # 可写的vocab_size属性
    @property
    def vocab_size(self):
        return self._vocab_size

    @vocab_size.setter
    def vocab_size(self, value):
        self._vocab_size = value
        # print(f"vocab_size已更新为: {value}")

    # 重写add_special_tokens方法，使其自动更新vocab_size
    def add_special_tokens(self, special_tokens_dict):
        # 调用原始tokenizer的方法
        num_added = self._tokenizer.add_special_tokens(special_tokens_dict)
        # 更新我们的vocab_size
        if num_added > 0:
            self._vocab_size = len(self._tokenizer.get_vocab())
            # print(f"添加了{num_added}个特殊token，vocab_size现在是: {self._vocab_size}")
        return num_added

    # 其他需要重写的重要方法
    def encode(self, text, **kwargs):
        return self._tokenizer.encode(text, **kwargs)

    def encode_plus(self, text, **kwargs):
        return self._tokenizer.encode_plus(text, **kwargs)

    def get_vocab(self):
        return self._tokenizer.get_vocab()

    def convert_tokens_to_ids(self, tokens):
        return self._tokenizer.convert_tokens_to_ids(tokens)

    def convert_ids_to_tokens(self, ids):
        return self._tokenizer.convert_ids_to_tokens(ids)

    # 添加一个用于测试的方法
    def test_encoding(self, texts):
        """
        测试编码功能，特别是padding以及解码功能

        Args:
            texts: 要编码的文本列表
        """
        print("\n=== 编码测试 ===")

        # 单独编码每个文本并打印结果
        print("单独编码每个文本:")
        for i, text in enumerate(texts):
            encoding = self._tokenizer.encode_plus(
                text,
                padding='max_length',
                max_length=77,
                truncation=True,  # 增加截断设置
                return_tensors='pt'
            )

            print(f"\n文本 {i + 1}: {text}")
            print(f"input_ids: {encoding['input_ids'][0]}")
            print(f"attention_mask: {encoding['attention_mask'][0]}")

            # 检查是否包含ID=0
            has_pad = 0 in encoding['input_ids'][0]
            print(f"包含ID=0 (padding): {has_pad}")

            # 测试解码功能
            # 1. 解码整个序列
            decoded_text = self._tokenizer.decode(encoding['input_ids'][0], skip_special_tokens=True)
            print(f"完整解码结果: \"{decoded_text}\"")
            print(f"解码结果与原文匹配: {text in decoded_text}")

            # 2. 只解码非padding部分（使用attention_mask）
            non_padding_positions = (encoding['attention_mask'][0] == 1).nonzero(as_tuple=True)[0]
            non_padding_ids = encoding['input_ids'][0][non_padding_positions]
            decoded_non_padding = self._tokenizer.decode(non_padding_ids, skip_special_tokens=True)
            print(f"非padding部分解码结果: \"{decoded_non_padding}\"")

            # 3. 测试ID=0（pad_token）单独解码
            pad_token_decoded = self._tokenizer.decode(torch.tensor([0]), skip_special_tokens=False)
            print(f"ID=0单独解码结果: \"{pad_token_decoded}\"")
            print(f"ID=0是否解码为pad_token: {pad_token_decoded.strip() == self._tokenizer.pad_token}")

            # 4. 解码一个包含padding的短序列
            # 创建一个包含真实token和padding的序列
            sample_seq = torch.cat([non_padding_ids[:5], torch.zeros(5, dtype=torch.long)])
            decoded_with_padding = self._tokenizer.decode(sample_seq, skip_special_tokens=False)
            print(f"包含padding的短序列解码: \"{decoded_with_padding}\"")

            # 5. 检查特殊token的解码
            special_tokens_decoded = {}
            for token_name, token_value in self._tokenizer.special_tokens_map.items():
                token_id = self._tokenizer.convert_tokens_to_ids(token_value)
                decoded = self._tokenizer.decode(torch.tensor([token_id]), skip_special_tokens=False)
                special_tokens_decoded[token_name] = decoded
            print(f"特殊tokens解码结果: {special_tokens_decoded}")

        # 批量编码
        print("\n批量编码所有文本:")
        try:
            # 确保设置truncation=True和padding='max_length'
            batch_encoding = self._tokenizer.batch_encode_plus(
                texts,
                padding='max_length',
                max_length=77,
                truncation=True,  # 关键：确保截断
                return_tensors='pt'
            )

            # 打印批量编码结果的形状
            print(f"batch_encoding['input_ids']形状: {batch_encoding['input_ids'].shape}")
            print(f"batch_encoding['attention_mask']形状: {batch_encoding['attention_mask'].shape}")

            # 验证padding使用的是ID=0
            print("\n验证padding使用ID=0:")
            # 通过attention_mask找到填充的位置
            for i, (ids, mask) in enumerate(zip(batch_encoding['input_ids'], batch_encoding['attention_mask'])):
                padding_positions = (mask == 0).nonzero(as_tuple=True)[0]
                if len(padding_positions) > 0:
                    # 检查padding位置的token ID是否为0
                    padding_ids = ids[padding_positions]
                    is_pad_id_0 = (padding_ids == 0).all().item()
                    print(f"文本 {i + 1} padding位置的ID是否为0: {is_pad_id_0}")

                    # 解码整个批次
                    batch_decoded = self._tokenizer.decode(ids, skip_special_tokens=True)
                    print(f"批次 {i + 1} 解码结果: \"{batch_decoded}\"")
                    print(f"批次解码结果与原文匹配: {texts[i] in batch_decoded}")
                else:
                    print(f"文本 {i + 1} 没有padding")

            # 测试整个批次的解码
            print("\n整个批次的解码结果:")
            batch_decoded_all = self._tokenizer.batch_decode(
                batch_encoding['input_ids'],
                skip_special_tokens=True
            )
            for i, (original, decoded) in enumerate(zip(texts, batch_decoded_all)):
                print(f"文本 {i + 1}:")
                print(f"原文: \"{original}\"")
                print(f"解码: \"{decoded}\"")
                print(f"匹配: {original in decoded}")

            return batch_encoding

        except Exception as e:
            print(f"批量编码失败: {str(e)}")
            return None


class Tokenizer_M:
    """
    可扩展的 CLIP Tokenizer，支持：
    1. 添加自定义 tokens（如 [MASK], [OBJ_CLS] 等）
    2. PAD token 映射到 id=0（后处理方式）
    3. 自动替换 padding id（保留 SEP token）
    4. 同步更新模型 embedding（自动冻结旧行）
    5. 保存和加载复用
    6. 兼容 dataset_msrvtt_feats.py 的使用方式
    
    流程与 test_temp_clip_embedding_expand.py 完全一致
    
    使用示例：
    
    # ========== 训练阶段 ==========
    # 1. 创建 tokenizer 并添加新 tokens
    tokenizer = Tokenizer_M(
        new_tokens=['[MASK]', '[OBJ_CLS]', '[OBJ_END]', '[OBJ_SEP]'],
        custom_pad_token='[PAD]',
        custom_pad_id=0
    )
    
    # 2. 加载模型并同步 embedding
    import clip
    model, _ = clip.load("ViT-B/32", device="cuda")
    tokenizer.resize_model_embeddings(model, freeze_old_embeddings=True)
    
    # 3. 训练...
    for batch in dataloader:
        encoded = tokenizer.encode_plus(batch['text'], 
                                       padding='max_length', 
                                       max_length=77,
                                       return_tensors='pt')
        outputs = model.encode_text(encoded['input_ids'])
        # 只有新增的 token embedding 会被更新
    
    # 4. 保存
    torch.save(model.state_dict(), './models/tokenizer_m/model.pt')
    tokenizer.save_pretrained('./models/tokenizer_m/tokenizer')
    
    # ========== 推理阶段 ==========
    # 1. 加载 tokenizer
    tokenizer = Tokenizer_M.from_pretrained('./models/tokenizer_m/tokenizer')
    
    # 2. 加载模型并同步 embedding（确保尺寸一致）
    model, _ = clip.load("ViT-B/32", device="cuda")
    tokenizer.resize_model_embeddings(model, freeze_old_embeddings=False)
    
    # 3. 加载训练好的权重
    model.load_state_dict(torch.load('./models/tokenizer_m/model.pt'))
    
    # 4. 推理
    encoded = tokenizer.encode_plus(text, ...)
    features = model.encode_text(encoded['input_ids'])
    """
    
    def __init__(self, 
                 new_tokens=None, 
                 custom_pad_token='[PAD]', 
                 custom_pad_id=0,
                 cache_dir=None,
                 auto_replace_pad=True):
        """
        初始化可扩展的 CLIP Tokenizer
        
        参数:
            new_tokens: list, 要添加的新 token 列表，如 ['[MASK]', '[OBJ_CLS]', '[OBJ_END]', '[OBJ_SEP]']
            custom_pad_token: str, 自定义 PAD token 名称，默认 '[PAD]'
            custom_pad_id: int, 自定义 PAD token 的 id，默认 0
            cache_dir: str, tokenizer 缓存目录
            auto_replace_pad: bool, 是否自动替换 padding id，默认 True
        """
        print(f"[Tokenizer_M] 初始化...")
        
        # 1. 加载原始 tokenizer
        if cache_dir is None:
            cache_dir = "./models/clip_tokenizer" if not sys.platform.startswith("win") else \
                        r"D:\Users\Administrator\Desktop\DeepLearningProjects_SELF\VideoCaption_Reconstruction\project\models\clip_tokenizer"
        
        self._tokenizer = load_clip_tokenizer(cache_dir=cache_dir)
        self.original_vocab_size = len(self._tokenizer)
        print(f"[Tokenizer_M] 原始词表大小: {self.original_vocab_size}")
        
        # 2. 保存原始 padding id（用于后处理）
        self.original_pad_id = self._tokenizer.pad_token_id
        self.custom_pad_token = custom_pad_token
        self.custom_pad_id = custom_pad_id
        self.auto_replace_pad = auto_replace_pad
        
        # 3. 添加新 tokens
        self.new_tokens = new_tokens if new_tokens is not None else []
        self.added_tokens_count = 0
        if len(self.new_tokens) > 0:
            print(f"[Tokenizer_M] 尝试新增 token: {self.new_tokens}")
            self.added_tokens_count = self._tokenizer.add_tokens(self.new_tokens)
            print(f"[Tokenizer_M] 实际新增 token 数量: {self.added_tokens_count}")
            print(f"[Tokenizer_M] 扩展后词表大小: {len(self._tokenizer)}")
            
            # 打印新增 token 的 id
            for token in self.new_tokens:
                if token in self._tokenizer.get_vocab():
                    token_id = self._tokenizer.convert_tokens_to_ids(token)
                    print(f"[Tokenizer_M]   '{token}' -> id={token_id}")
        
        # 4. 设置自定义 PAD token（后处理方案）
        if custom_pad_id == 0:
            old_token_at_0 = self._tokenizer.convert_ids_to_tokens([custom_pad_id])[0]
            print(f"\n[Tokenizer_M] 配置自定义 PAD token: '{custom_pad_token}' -> id={custom_pad_id}")
            print(f"[Tokenizer_M] 原始 id={custom_pad_id} 对应的 token: '{old_token_at_0}'")
            print(f"[Tokenizer_M] Tokenizer 原始 pad_token_id: {self.original_pad_id}")
            
            # 修改 decoder，让 id=0 显示为 [PAD]
            self._tokenizer.decoder[custom_pad_id] = custom_pad_token
            # 同时修改 encoder（双向映射）
            if hasattr(self._tokenizer, 'encoder'):
                self._tokenizer.encoder[custom_pad_token] = custom_pad_id
            
            print(f"[Tokenizer_M] 策略: Tokenizer 继续使用 id={self.original_pad_id} 进行 padding")
            print(f"[Tokenizer_M]       编码后会自动将 {self.original_pad_id} 替换为 {custom_pad_id}")
            print(f"[Tokenizer_M] 说明: 最终 PAD token 使用 id=0 的 embedding（原为'{old_token_at_0}'），不新增词表项")
        
        # 5. 存储可写的 vocab_size
        self._vocab_size = len(self._tokenizer)
        print(f"\n[Tokenizer_M] 初始化完成，当前词表大小: {self._vocab_size}")
        print(f"[Tokenizer_M] 新增 token 数量: {self.added_tokens_count}")
        print(f"[Tokenizer_M] 自动替换 padding: {self.auto_replace_pad}\n")
    
    def _replace_pad_id(self, input_ids, sep_id=None):
        """
        将 input_ids 中的 padding 位置的 original_pad_id 替换为 custom_pad_id
        
        如果 original_pad_id == sep_id，则只替换第一个 SEP 之后的 original_pad_id，保留 SEP 本身
        """
        if not self.auto_replace_pad or self.original_pad_id == self.custom_pad_id:
            return input_ids
        
        output_ids = input_ids.clone()
        
        # 如果 original_pad_id 和 sep_id 相同，需要特殊处理
        if sep_id is not None and self.original_pad_id == sep_id:
            # 遍历每个样本
            for i in range(output_ids.shape[0]):
                # 找到第一个 SEP 的位置
                sep_positions = (output_ids[i] == sep_id).nonzero(as_tuple=True)[0]
                if len(sep_positions) > 0:
                    first_sep_pos = sep_positions[0].item()
                    # 只替换 SEP 之后的 original_pad_id
                    mask = output_ids[i, first_sep_pos + 1:] == self.original_pad_id
                    output_ids[i, first_sep_pos + 1:][mask] = self.custom_pad_id
        else:
            # 如果不冲突，直接替换所有
            mask = (output_ids == self.original_pad_id)
            output_ids[mask] = self.custom_pad_id
        
        return output_ids
    
    def encode_plus(self, text, **kwargs):
        """
        编码文本，自动替换 padding id
        
        兼容 dataset_msrvtt_feats.py 的调用方式
        """
        # 调用原始 tokenizer 的 encode_plus
        encoded = self._tokenizer.encode_plus(text, **kwargs)
        
        # 如果返回 tensors 并且需要替换 padding
        if self.auto_replace_pad and 'input_ids' in encoded:
            if isinstance(encoded['input_ids'], torch.Tensor):
                sep_id = self._tokenizer.eos_token_id
                encoded['input_ids'] = self._replace_pad_id(encoded['input_ids'], sep_id=sep_id)
        
        return encoded
    
    def __call__(self, text, **kwargs):
        """
        编码文本，自动替换 padding id
        
        兼容标准的 tokenizer 调用方式
        """
        # 调用原始 tokenizer
        encoded = self._tokenizer(text, **kwargs)
        
        # 如果返回 tensors 并且需要替换 padding
        if self.auto_replace_pad and 'input_ids' in encoded:
            if isinstance(encoded['input_ids'], torch.Tensor):
                sep_id = self._tokenizer.eos_token_id
                encoded['input_ids'] = self._replace_pad_id(encoded['input_ids'], sep_id=sep_id)
        
        return encoded
    
    # 将原始 tokenizer 的所有属性和方法转发到此类
    def __getattr__(self, name):
        # 如果当前类没有该属性，则委托给原始 tokenizer
        if name != '_tokenizer' and hasattr(self._tokenizer, name):
            return getattr(self._tokenizer, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    # 可写的 vocab_size 属性
    @property
    def vocab_size(self):
        return self._vocab_size
    
    @vocab_size.setter
    def vocab_size(self, value):
        self._vocab_size = value
    
    def get_vocab(self):
        """返回词表"""
        return self._tokenizer.get_vocab()
    
    def convert_tokens_to_ids(self, tokens):
        """转换 tokens 到 ids"""
        return self._tokenizer.convert_tokens_to_ids(tokens)
    
    def convert_ids_to_tokens(self, ids):
        """转换 ids 到 tokens"""
        return self._tokenizer.convert_ids_to_tokens(ids)
    
    def decode(self, token_ids, **kwargs):
        """解码 token ids"""
        return self._tokenizer.decode(token_ids, **kwargs)
    
    def batch_decode(self, sequences, **kwargs):
        """批量解码"""
        return self._tokenizer.batch_decode(sequences, **kwargs)
    
    def __len__(self):
        """返回词表大小"""
        return self._vocab_size
    
    def save_pretrained(self, save_directory):
        """
        保存 tokenizer 和配置信息
        
        保存内容：
        1. tokenizer 文件（标准 HuggingFace 格式）：
           - vocab.json
           - merges.txt
           - tokenizer_config.json
           - special_tokens_map.json
           - added_tokens.json
        2. tokenizer_m_config.json（Tokenizer_M 元信息）
        
        标准保存路径示例：
            ./models/tokenizer_m/tokenizer/
        
        保存后的文件结构：
            tokenizer/
            ├── vocab.json              # 词表（49412 个 tokens）
            ├── merges.txt              # BPE 合并规则
            ├── tokenizer_config.json   # Tokenizer 配置
            ├── special_tokens_map.json # 特殊 token 映射
            ├── added_tokens.json       # 新增 token 列表
            └── tokenizer_m_config.json # ⭐ Tokenizer_M 元信息
        """
        print(f"\n[Tokenizer_M] 保存 tokenizer 到: {save_directory}")
        
        # 确保目录存在
        os.makedirs(save_directory, exist_ok=True)
        
        # 1. 保存原始 tokenizer
        self._tokenizer.save_pretrained(save_directory)
        print(f"[Tokenizer_M] ✅ Tokenizer 文件已保存")
        
        # 2. 保存 Tokenizer_M 的配置信息
        config = {
            'new_tokens': self.new_tokens,
            'added_tokens_count': self.added_tokens_count,
            'original_vocab_size': self.original_vocab_size,
            'current_vocab_size': self._vocab_size,
            'custom_pad_token': self.custom_pad_token,
            'custom_pad_id': self.custom_pad_id,
            'original_pad_id': self.original_pad_id,
            'auto_replace_pad': self.auto_replace_pad,
            'tokenizer_m_version': '1.0'
        }
        
        config_path = os.path.join(save_directory, 'tokenizer_m_config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"[Tokenizer_M] ✅ 配置信息已保存: {config_path}")
        print(f"[Tokenizer_M] 保存完成！")
        
        return save_directory
    
    @classmethod
    def from_pretrained(cls, 
                       load_directory, 
                       cache_dir=None,
                       auto_replace_pad=None):
        """
        从保存的目录加载 Tokenizer_M
        
        参数:
            load_directory: str, 保存的目录路径
            cache_dir: str, tokenizer 缓存目录（可选，用于回退）
            auto_replace_pad: bool, 是否自动替换 padding（可覆盖保存的配置）
        
        返回:
            Tokenizer_M 实例
        """
        print(f"\n[Tokenizer_M] 从目录加载: {load_directory}")
        
        # 1. 尝试加载配置文件
        config_path = os.path.join(load_directory, 'tokenizer_m_config.json')
        
        if os.path.exists(config_path):
            print(f"[Tokenizer_M] 找到配置文件: {config_path}")
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            print(f"[Tokenizer_M] 配置信息:")
            print(f"  - 原始词表大小: {config['original_vocab_size']}")
            print(f"  - 当前词表大小: {config['current_vocab_size']}")
            print(f"  - 新增 token 数量: {config['added_tokens_count']}")
            print(f"  - 新增 tokens: {config['new_tokens']}")
            
            # 2. 创建实例（不添加 token，因为已经在保存的 tokenizer 中）
            instance = cls.__new__(cls)
            
            # 3. 加载 tokenizer
            from transformers import CLIPTokenizer
            instance._tokenizer = CLIPTokenizer.from_pretrained(
                load_directory,
                local_files_only=True
            )
            print(f"[Tokenizer_M] ✅ Tokenizer 加载成功，词表大小: {len(instance._tokenizer)}")
            
            # 4. 恢复配置
            instance.new_tokens = config['new_tokens']
            instance.added_tokens_count = config['added_tokens_count']
            instance.original_vocab_size = config['original_vocab_size']
            instance._vocab_size = config['current_vocab_size']
            instance.custom_pad_token = config['custom_pad_token']
            instance.custom_pad_id = config['custom_pad_id']
            instance.original_pad_id = config['original_pad_id']
            instance.auto_replace_pad = auto_replace_pad if auto_replace_pad is not None else config.get('auto_replace_pad', True)
            
            # 5. 确保 decoder 映射正确（PAD token）
            if instance.custom_pad_id == 0:
                instance._tokenizer.decoder[instance.custom_pad_id] = instance.custom_pad_token
                if hasattr(instance._tokenizer, 'encoder'):
                    instance._tokenizer.encoder[instance.custom_pad_token] = instance.custom_pad_id
            
            print(f"[Tokenizer_M] ✅ 配置恢复完成")
            print(f"[Tokenizer_M] 自动替换 padding: {instance.auto_replace_pad}")
            
            return instance
        
        else:
            # 回退：尝试作为普通 tokenizer 加载
            print(f"[Tokenizer_M] ⚠️  未找到 tokenizer_m_config.json")
            print(f"[Tokenizer_M] 尝试作为普通 CLIP tokenizer 加载...")
            
            try:
                from transformers import CLIPTokenizer
                tokenizer = CLIPTokenizer.from_pretrained(
                    load_directory,
                    local_files_only=True
                )
                
                # 创建一个基础实例（没有新 token）
                instance = cls.__new__(cls)
                instance._tokenizer = tokenizer
                instance.new_tokens = []
                instance.added_tokens_count = 0
                instance.original_vocab_size = len(tokenizer)
                instance._vocab_size = len(tokenizer)
                instance.custom_pad_token = '[PAD]'
                instance.custom_pad_id = 0
                instance.original_pad_id = tokenizer.pad_token_id
                instance.auto_replace_pad = auto_replace_pad if auto_replace_pad is not None else True
                
                print(f"[Tokenizer_M] ✅ 加载完成（作为基础 tokenizer）")
                print(f"[Tokenizer_M] ⚠️  注意：没有新增 token 信息")
                
                return instance
            
            except Exception as e:
                raise ValueError(f"无法从 {load_directory} 加载 tokenizer: {e}")
    
    def get_added_tokens_info(self):
        """获取新增 token 的信息"""
        info = {
            'new_tokens': self.new_tokens,
            'added_count': self.added_tokens_count,
            'original_vocab_size': self.original_vocab_size,
            'current_vocab_size': self._vocab_size,
            'custom_pad_token': self.custom_pad_token,
            'custom_pad_id': self.custom_pad_id,
            'original_pad_id': self.original_pad_id
        }
        
        # 获取每个新 token 的 id
        token_ids = {}
        for token in self.new_tokens:
            if token in self._tokenizer.get_vocab():
                token_ids[token] = self._tokenizer.convert_tokens_to_ids(token)
        info['token_ids'] = token_ids
        
        return info
    
    def print_info(self):
        """打印 tokenizer 信息"""
        info = self.get_added_tokens_info()
        print("\n" + "=" * 80)
        print("【Tokenizer_M 信息】")
        print(f"  原始词表大小: {info['original_vocab_size']}")
        print(f"  当前词表大小: {info['current_vocab_size']}")
        print(f"  新增 token 数量: {info['added_count']}")
        print(f"  PAD token: '{info['custom_pad_token']}' -> id={info['custom_pad_id']}")
        print(f"  原始 PAD id: {info['original_pad_id']}")
        print(f"  CLS token: '{self._tokenizer.bos_token}' -> id={self._tokenizer.bos_token_id}")
        print(f"  SEP token: '{self._tokenizer.eos_token}' -> id={self._tokenizer.eos_token_id}")
        
        if len(info['token_ids']) > 0:
            print(f"\n  新增 tokens:")
            for token, token_id in info['token_ids'].items():
                print(f"    '{token}' -> id={token_id}")
        print("=" * 80 + "\n")
    
    # ========== Embedding 同步更新方法 ==========
    
    def resize_model_embeddings(self, model, freeze_old_embeddings=True):
        """
        同步更新模型的 token embedding
        
        参数:
            model: 模型对象（OpenAI CLIP 或 HuggingFace 模型）
            freeze_old_embeddings: bool, 是否冻结旧的 embedding 行（只训练新增的）
        
        返回:
            更新后的 embedding 层
        """
        if self.added_tokens_count == 0:
            print("[Tokenizer_M] 没有新增 token，无需更新 embedding")
            return None
        
        print(f"\n[Tokenizer_M] 开始同步更新模型 embedding...")
        print(f"[Tokenizer_M] 新增 token 数量: {self.added_tokens_count}")
        
        # 检测模型类型
        if hasattr(model, 'token_embedding'):  # OpenAI CLIP
            print("[Tokenizer_M] 检测到 OpenAI CLIP 模型")
            return self._resize_openai_clip_embedding(model, freeze_old_embeddings)
        elif hasattr(model, 'get_input_embeddings'):  # HuggingFace 模型
            print("[Tokenizer_M] 检测到 HuggingFace 模型")
            return self._resize_hf_embedding(model, freeze_old_embeddings)
        else:
            raise ValueError("不支持的模型类型！请传入 OpenAI CLIP 或 HuggingFace 模型")
    
    def _resize_openai_clip_embedding(self, model, freeze_old_embeddings=True):
        """扩展 OpenAI CLIP 的 token_embedding"""
        import torch.nn as nn
        
        old_emb = model.token_embedding
        old_vocab_size, embed_dim = old_emb.weight.shape
        new_vocab_size = old_vocab_size + self.added_tokens_count
        
        print(f"[Tokenizer_M] 原始 embedding: {old_emb.weight.shape}")
        print(f"[Tokenizer_M] 目标 embedding: ({new_vocab_size}, {embed_dim})")
        
        # 创建新的 embedding
        new_emb = nn.Embedding(new_vocab_size, embed_dim)
        
        with torch.no_grad():
            # 复制旧的权重
            new_emb.weight[:old_vocab_size].copy_(old_emb.weight)
            
            # 初始化新增的权重（使用旧权重的统计信息）
            base_mean = float(old_emb.weight.mean())
            base_std = float(old_emb.weight.std())
            new_emb.weight[old_vocab_size:].normal_(mean=base_mean, std=max(base_std * 0.5, 1e-2))
        
        # 替换模型的 embedding
        model.token_embedding = new_emb
        
        print(f"[Tokenizer_M] ✅ Embedding 扩展完成: {new_emb.weight.shape}")
        
        # 冻结旧的 embedding 行
        if freeze_old_embeddings and self.added_tokens_count > 0:
            self._register_freeze_hook(new_emb, old_vocab_size)
            print(f"[Tokenizer_M] ✅ 已冻结前 {old_vocab_size} 行（仅训练新增的 {self.added_tokens_count} 行）")
        
        return new_emb
    
    def _resize_hf_embedding(self, model, freeze_old_embeddings=True):
        """扩展 HuggingFace 模型的 embedding"""
        old_emb = model.get_input_embeddings()
        old_vocab_size = old_emb.num_embeddings
        
        print(f"[Tokenizer_M] 原始 embedding: {old_emb.weight.shape}")
        
        # 使用 HuggingFace 的内置方法
        model.resize_token_embeddings(self._vocab_size)
        new_emb = model.get_input_embeddings()
        
        print(f"[Tokenizer_M] ✅ Embedding 扩展完成: {new_emb.weight.shape}")
        
        # 冻结旧的 embedding 行
        if freeze_old_embeddings and self.added_tokens_count > 0:
            self._register_freeze_hook(new_emb, old_vocab_size)
            print(f"[Tokenizer_M] ✅ 已冻结前 {old_vocab_size} 行（仅训练新增的 {self.added_tokens_count} 行）")
        
        return new_emb
    
    def _register_freeze_hook(self, embedding, num_old_tokens):
        """注册梯度 hook，冻结旧 token 的 embedding"""
        def freeze_hook(grad):
            if grad is None:
                return grad
            # 前 num_old_tokens 行的梯度置零
            frozen_grad = torch.zeros_like(grad[:num_old_tokens])
            trainable_grad = grad[num_old_tokens:]
            return torch.cat([frozen_grad, trainable_grad], dim=0)
        
        embedding.weight.register_hook(freeze_hook)


# 使用示例
if __name__ == "__main__":
    # # 示例1：原有的 CLIPTokenizer_Custom
    # print("=== 示例1：CLIPTokenizer_Custom ===")
    # tokenizer = CLIPTokenizer_Custom()
    # test_texts = ["这是一个短文本", "这是一个稍微长一点的文本，需要更多的token来表示!."]
    # tokenizer.test_encoding(test_texts)
    
    # 示例2：新的 Tokenizer_M（添加自定义 tokens）
    print("\n=== 示例2：Tokenizer_M（添加新 tokens）===")
    tokenizer_m = Tokenizer_M(
        new_tokens=['[MASK]', '[OBJ_CLS]', '[OBJ_END]', '[OBJ_SEP]'],
        custom_pad_token='[PAD]',
        custom_pad_id=0,
        auto_replace_pad=True
    )
    
    # 打印信息
    tokenizer_m.print_info()
    
    # 测试编码
    test_text = "[OBJ_CLS] a man is talking [OBJ_END]"
    print(f"测试文本: {test_text}")
    
    # 测试 encode_plus（与 dataset_msrvtt_feats.py 兼容）
    encoded = tokenizer_m.encode_plus(
        test_text,
        padding='max_length',
        max_length=77,
        truncation=True,
        return_tensors='pt'
    )
    
    print(f"\n编码结果（完整序列，max_length=77）:")
    print(f"  input_ids shape: {encoded['input_ids'].shape}")
    print(f"  attention_mask shape: {encoded['attention_mask'].shape}")
    
    # 显示完整的 77 个 token ids
    ids_list = encoded['input_ids'][0].tolist()
    print(f"\n  完整 input_ids (77个):")
    print(f"  {ids_list}")
    
    # 解码为 tokens
    tokens = tokenizer_m.convert_ids_to_tokens(ids_list)
    print(f"\n  完整 tokens (77个):")
    print(f"  {tokens}")
    
    # 找出有效 token 的数量（非 padding）
    valid_count = encoded['attention_mask'][0].sum().item()
    print(f"\n  有效 token 数量（非padding）: {valid_count}")
    print(f"  padding 数量: {77 - valid_count}")
    
    # 测试 __call__ 方法
    print(f"\n" + "=" * 80)
    print("使用 __call__ 方法测试:")
    encoded2 = tokenizer_m(test_text, padding='max_length', max_length=77, 
                           truncation=True, return_tensors='pt')
    print(f"  input_ids shape: {encoded2['input_ids'].shape}")
    print(f"  与 encode_plus 结果一致: {torch.equal(encoded['input_ids'], encoded2['input_ids'])}")
    
    # 测试更复杂的句子
    print(f"\n" + "=" * 80)
    test_text2 = "[OBJ_CLS] a man in black [OBJ_SEP] a red car [OBJ_END] is talking [MASK]"
    print(f"测试更复杂的文本: {test_text2}")
    
    encoded3 = tokenizer_m.encode_plus(
        test_text2,
        padding='max_length',
        max_length=77,
        truncation=True,
        return_tensors='pt'
    )
    
    ids_list3 = encoded3['input_ids'][0].tolist()
    tokens3 = tokenizer_m.convert_ids_to_tokens(ids_list3)
    valid_count3 = encoded3['attention_mask'][0].sum().item()
    
    print(f"\n  完整 input_ids (77个):")
    print(f"  {ids_list3}")
    print(f"\n  完整 tokens (77个):")
    print(f"  {tokens3}")
    print(f"\n  有效 token 数量: {valid_count3}, padding 数量: {77 - valid_count3}")
    
    # 示例：同步更新模型 embedding
    print(f"\n" + "=" * 80)
    print("示例：同步更新模型 embedding")
    print("=" * 80)
    
    # 模拟加载 OpenAI CLIP 模型
    try:
        import clip
        print("\n[示例] 加载 OpenAI CLIP 模型...")
        clip_model, _ = clip.load("ViT-B/32", device="cpu", jit=False)
        
        print(f"[示例] 原始 embedding shape: {clip_model.token_embedding.weight.shape}")
        
        # 同步更新 embedding
        new_embedding = tokenizer_m.resize_model_embeddings(
            clip_model, 
            freeze_old_embeddings=True
        )
        
        print(f"[示例] 更新后 embedding shape: {clip_model.token_embedding.weight.shape}")
        print(f"[示例] 词表大小: {len(tokenizer_m)}")
        print(f"[示例] Embedding 大小: {clip_model.token_embedding.num_embeddings}")
        print(f"[示例] ✅ 词表与 Embedding 大小一致: {len(tokenizer_m) == clip_model.token_embedding.num_embeddings}")
        
        # 测试前向传播
        print(f"\n[示例] 测试前向传播...")
        test_ids = encoded3['input_ids']  # 使用之前编码的结果
        with torch.no_grad():
            text_features = clip_model.encode_text(test_ids)
        print(f"[示例] ✅ 前向传播成功，输出 shape: {text_features.shape}")
        
    except Exception as e:
        print(f"\n[示例] OpenAI CLIP 测试跳过: {e}")
        print("[示例] 请确保已安装 CLIP: pip install git+https://github.com/openai/CLIP.git")
        print("\n[示例] 使用方法:")
        print("  tokenizer = Tokenizer_M(new_tokens=['[MASK]', ...])")
        print("  # 加载模型后")
        print("  tokenizer.resize_model_embeddings(model, freeze_old_embeddings=True)")
    
    # 示例：保存和加载 Tokenizer_M
    print(f"\n" + "=" * 80)
    print("示例：保存和加载 Tokenizer_M")
    print("=" * 80)
    
    # 保存 tokenizer（标准路径）
    save_path = "./models/tokenizer_m/tokenizer"
    print(f"\n[示例] 保存 tokenizer 到: {save_path}")
    tokenizer_m.save_pretrained(save_path)
    
    # 加载 tokenizer
    print(f"\n[示例] 从保存路径加载 tokenizer...")
    tokenizer_loaded = Tokenizer_M.from_pretrained(save_path)
    
    # 验证加载的 tokenizer
    print(f"\n[示例] 验证加载的 tokenizer:")
    tokenizer_loaded.print_info()
    
    # 测试编码结果是否一致
    test_verify = "[OBJ_CLS] hello world [OBJ_END]"
    enc_original = tokenizer_m.encode_plus(
        test_verify,
        padding='max_length',
        max_length=77,
        truncation=True,
        return_tensors='pt'
    )
    enc_loaded = tokenizer_loaded.encode_plus(
        test_verify,
        padding='max_length',
        max_length=77,
        truncation=True,
        return_tensors='pt'
    )
    
    is_same = torch.equal(enc_original['input_ids'], enc_loaded['input_ids'])
    print(f"\n[示例] 原始 tokenizer 和加载的 tokenizer 编码结果一致: {is_same}")
    
    if is_same:
        print(f"[示例] ✅ 保存和加载功能正常！")
        print(f"[示例] 编码结果:")
        print(f"  {enc_loaded['input_ids'][0].tolist()}")
    else:
        print(f"[示例] ❌ 编码结果不一致！")
        print(f"  原始: {enc_original['input_ids'][0, :20].tolist()}")
        print(f"  加载: {enc_loaded['input_ids'][0, :20].tolist()}")
    
    # 显示保存的文件结构
    print(f"\n" + "=" * 80)
    print(f"保存的文件结构（标准 HuggingFace 格式）:")
    print(f"=" * 80)
    print(f"{save_path}/")
    if os.path.exists(save_path):
        for file in sorted(os.listdir(save_path)):
            file_path = os.path.join(save_path, file)
            if os.path.isfile(file_path):
                size_kb = os.path.getsize(file_path) / 1024
                if file == 'tokenizer_m_config.json':
                    print(f"├── {file} ⭐ (Tokenizer_M 元信息, {size_kb:.1f} KB)")
                else:
                    print(f"├── {file} ({size_kb:.1f} KB)")
    
    print(f"\n[示例] 💡 标准使用流程:")
    print(f"  训练时：")
    print(f"    - 保存模型: torch.save(model.state_dict(), './models/tokenizer_m/model.pt')")
    print(f"    - 保存 tokenizer: tokenizer.save_pretrained('./models/tokenizer_m/tokenizer')")
    print(f"  推理时：")
    print(f"    - 加载 tokenizer: tokenizer = Tokenizer_M.from_pretrained('./models/tokenizer_m/tokenizer')")
    print(f"    - 加载模型并同步: tokenizer.resize_model_embeddings(model)")
    print(f"    - 加载权重: model.load_state_dict(torch.load('./models/tokenizer_m/model.pt'))")