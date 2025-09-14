# BaseTransformer：基础 Transformer 及其相关模型变体

## 项目简介

本项目是面向 **Transformer 初学者**的入门级学习实践，通过手动实现基础架构的 Transformer 文本翻译模型（中英方向）以及 Transformer （文本翻译）等模型变体，帮助初学者理解 Transformer 的核心原理（尤其是多头注意力机制），并对比 PyTorch 官方库接口的使用方式。

项目分为多个模块，定位为「学习型 Demo」，不追求生产级性能，而是聚焦于**模型架构拆解、关键模块实现细节、训练推理流程梳理**，适合刚接触 Transformer 的开发者入门参考。

## 模块一：基础 Transformer 实现的文本翻译模型

### 数据集说明

#### 数据来源

文本翻译使用的基准数据集来源于国际顶级机器翻译评测任务 **WMT 2021 News Translation Task**（[官方链接](https://www.statmt.org/wmt21/translation-task.html)），该数据集由多个权威平行语料库整合而成，包括：

- ParaCrawl（网页平行语料）
- News-Commentary（新闻评论平行语料）
- Wiki-Titles（维基标题平行语料）
- UN Parallel Corpus（联合国平行语料）
- WikiMatrix（维基百科平行语料）
- CCMT（中文机器翻译评测语料）

原始数据集包含 **2500 万+ 中英双语句对**，覆盖新闻、百科、官方文档等多种场景，具备良好的语言多样性。

#### 项目使用数据集

对于文本翻译项目，为平衡「演示效率」与「模型泛化能力」，项目提供两个子数据集，存放于 `data/translate/` 目录下：
| 文件名 | 数据量 | 用途 |
|--------|--------|------|
| `TranslationData.csv` | 2 万条双语句对 | 基础演示、快速验证模型流程（训练耗时短，适合调试） |
| `2m_WMT21.csv` | 200 万条双语句对 | 提升模型泛化能力（数据量更大，训练后模型适配更多语言场景） |

> 注：所有数据集已预处理为「中文句子（空格分词）,英文句子（空格分词）」的 CSV 格式，可直接通过`TranslationDataset`类加载。

### 代码结构与说明

项目核心代码按「功能模块」拆分，结构清晰，便于针对性学习：

| 代码文件                         | 核心功能                      | 学习重点                                                                                                                                                             |
| -------------------------------- | ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `TranslationDataset.py`          | 数据集加载与预处理            | 1. 文本 → 索引的映射（构建词汇表，含`<bos>/<eos>/<pad>/<unk>`特殊标记）<br>2. 序列填充/截断（适配模型输入长度）<br>3. 批量数据生成（兼容 DataLoader）                |
| `TextTranslate.ipynb`            | 手动实现 Transformer 完整架构 | 1. 位置编码（正余弦编码）<br>2. 多头注意力（自注意力+交叉注意力，含因果掩码/填充掩码）<br>3. 编码器层/解码器层（残差连接+层归一化+前馈网络）<br>4. 完整训练/推理流程 |
| `TorchOfficialTransformer.ipynb` | PyTorch 官方库接口实现        | 1. 官方`nn.Transformer`的参数配置（与手动实现对齐）<br>2. 官方接口与自定义模块的结合（如自定义嵌入+位置编码）<br>3. 高效调用技巧（如自动因果掩码生成）               |

#### 1. 手动实现 Transformer

通过 `TextTranslate.ipynb` 可深入理解 Transformer 的「从 0 到 1」构建过程，关键模块拆解：

- **嵌入与位置编码**：解决文本离散输入的向量表示，补充序列位置信息
- **多头注意力**：拆解注意力计算流程（QKV 投影、缩放、掩码、Softmax、加权求和），理解「多头」如何捕捉多维度关联
- **编码器/解码器堆叠**：学习残差连接与层归一化的作用（缓解梯度消失，稳定训练）
- **训练逻辑**：适配翻译任务的「错位输入-目标」设计

##### 2. 官方库接口使用

通过 `TorchOfficialTransformer.ipynb` 对比手动实现与官方接口的差异：

- 官方库的参数映射（如`nhead`、`dim_feedforward`与手动实现的对应关系）
- 官方优化（如`batch_first=True`适配批量输入格式，`generate_square_subsequent_mask`简化掩码生成）
- 自定义模块（如位置编码）与官方接口的兼容方式

#### 实践验证：单样本过拟合与翻译推理

项目提供完整的训练与推理代码，支持：

- 小样本过拟合（快速验证模型是否能学习简单映射）
- 批量训练（使用 200 万数据集提升泛化能力）
- 自回归翻译推理（输入中文句子，输出英文翻译结果）

### 使用说明

#### 1. 数据准备

将数据集文件（`TranslationData.csv`/`2m_WMT21.csv`）放入 `data/translate/` 目录，确保路径与代码中 `file_path` 参数一致（如 `../data/translate/TranslationData.csv`）。

#### 2. 模型训练

##### 1.手动实现模型训练

```python
dataset = TranslationDataset(file_path="../data/TranslationData.csv", max_lines=20000)
model = Transformer(src_vocab_size=len(dataset.ch_token_to_index), ...)
# 配置优化器、损失函数、学习率调度器后启动训练
```

##### 2.官方库模型训练

```python
dataset = TranslationDataset(file_path="../data/2m_WMT21.csv", max_lines=2000000)
model = ChEnTransformer(src_vocab_size=len(dataset.ch_token_to_index), ...)
# 复用手动实现的训练逻辑，但需要替换模型实例以及传参方式（详见代码）
```

##### 3. 翻译推理

使用项目提供的 `translate` 函数，输入中文句子（需空格分词）即可得到英文翻译：

```python
from TorchOfficialTransformer import translate

# 示例：中文输入（空格分词）
chinese_input = "表演 的 明星 是 X 女孩 团队 — — 由 一对 具有 天才 技艺 的 艳舞 女孩 们 组成 ， 其中 有些 人 受过 专业 的 训练 。"
# 执行翻译
result = translate(chinese_input, model, dataset, device="cuda")
# 输出结果
print("英文翻译:", " ".join([token for token in result if token not in ["<bos>", "<eos>"]]))
```

##### 注意事项

1. **分词格式**：输入中文句子需按数据集格式进行「空格分词」，否则会被识别为`<unk>`（未知词）
2. **设备配置**：训练建议使用 GPU（`device="cuda"`），2 万条数据训练约 10 分钟，200 万数据建议使用多 GPU 或调整 batch_size
3. **超参数调整**：若训练不稳定，可调整`d_model`（需为`nhead`的整数倍）、`dropout`（防止过拟合）
4. **检查点**：项目提供`save_checkpoint`/`load_checkpoint`函数，支持中断后继续训练

## 模块二：仅使用 Transformer 解码器的文本生成模型

### 数据集说明

文本生成数据集（[官方链接](https://github.com/liuhuanyong/MusicLyricChatbot)）包含以下几个字段：

- singer: 歌手名
- album: 专辑
- song: 歌曲名称
- author: 作词者
- composer: 作曲者
  整个歌曲信息文件包含 140068 首歌曲 ，示例如下：

```
{　　"_id" : { "$oid" : "5bbdf280831b976548aa1509" },
    "singer" : "张学友",
    "song" : "耐人寻味",
    "geci" : [
            "在每个佳节 亦送上鲜花", "还常来电话 留言传达我牵挂", "在每次分散 亦送你归家", "无聊时玩耍 忘形时
            候说婚嫁", "*如最美满爱情 以这一生作保证", "说到最煽情 而你却说你清醒*", "这一宵 我再次借酒消愁 仍难忘透
            ", "我有哪里错漏 错要提分手", "这一天 对我有哪些要求 何妨直说", "你却说到以后 变做什么好友", "假使间 我很
            丑也不温柔 如何闷透", "你也说个理由 免我撒赖不走", "想不到 每到了紧要关头 词难达意", "我要对你挽留 却愤怒
            转身走", "是你太高贵 是我也潇洒", "和谐如像完 就算闲日会吵架", "望戏里主角 换上了婚纱", "何妨来玩耍 问你
            何日会想嫁" ],
    "composer" : "",
    "author" : "劳子",
    "album" : "《Jacky Cheung 15-Disc2》"
}
```

对于文本生成所用到的歌词数据文件 lyrics.csvF，项目已将源 json 文件中的所有中文歌词提取到该文件中，并存放于`data/generate/` 目录下

### 代码结构与说明

#### 1. 手动实现 Transformer 解码器

通过 `TextGenerate.ipynb` 可理解仅使用解码器进行文本生成的核心逻辑，关键模块拆解：

- **嵌入与位置编码**：与模块一一致，通过词嵌入将歌词字符转换为向量，结合正余弦位置编码补充序列顺序信息
- **解码器层**：单解码器层包含两个子层
  - 带因果掩码的自注意力（防止模型关注未来字符，确保生成的时序合理性）
  - 前馈网络（通过线性变换与非线性激活捕捉特征）
  - 每个子层均采用「残差连接+层归一化」稳定训练
- **多层堆叠**：通过 `nn.ModuleList` 堆叠多个解码器层，提升模型对长序列的建模能力
- **输出层**：通过线性变换将解码器输出映射到词汇表维度，实现字符预测

##### 2. 官方库接口使用

通过 `TorchOfficialGenerate.ipynb` 对比手动实现与 PyTorch 官方解码器的差异：

- 官方 `nn.TransformerDecoder` 与 `nn.TransformerDecoderLayer` 的参数映射（如 `nhead` 对应多头数，`batch_first=True` 适配批量输入格式）
- 使用官方 `generate_square_subsequent_mask` 自动生成因果掩码，无需手动构建上三角矩阵
- 由于文本生成仅需解码器，将 `memory` 设为零矩阵适配官方接口格式

#### 实践验证：歌词生成效果与训练监控

项目提供完整的训练与生成流程，支持：

- 小批量快速验证（使用 `nrows` 参数限制数据量，快速测试模型是否收敛）
- 全量数据训练（使用 14 万+ 歌词数据提升生成质量）
- 可控文本生成（通过输入前缀、调整温度参数 `tempreture` 控制生成多样性）

### 使用说明

#### 1. 数据准备

将歌词数据集文件（`lyrics.csv`）放入 `data/generate/` 目录，确保路径与代码中 `file_path` 参数一致（如 `../data/generate/lyrics.csv`）。

#### 2. 模型训练

##### 1. 手动实现模型训练

```python
dataset = LyricsDataset(file_path="../data/generate/lyrics.csv", nrows=-1)
model = TextGenerate(
    d_model=512,
    vocab_size=len(dataset.token_to_index),
    num_layers=6,
    heads=8,
    dropout=0.1
)
# 配置优化器（Adam）、学习率调度器（含预热机制）、损失函数（CrossEntropyLoss）后启动训练
```

##### 2. 官方库模型训练

```python
dataset = LyricsDataset(file_path="../data/generate/lyrics.csv", nrows=-1)
model = TextGenerate(
    d_model=512,
    vocab_size=len(dataset.token_to_index),
    num_layers=6,
    heads=8,
    dropout=0.1
)
# 复用手动实现的训练逻辑，需注意传递官方接口所需的tgt_mask参数（详见代码）
```

##### 3. 歌词生成

使用项目提供的 `predict` 函数，输入前缀文本即可生成连贯歌词：

```python
from TextGenerate import predict

# 示例：输入前缀（用"/"分割不同分句的开头）
prefix = "晚风/我们"
# 执行生成（最大长度50字符，温度0.95控制多样性）
generated = predict(
    text=prefix,
    model=model,
    max_length=50,
    separator="/",
    device="cuda",
    to_index=dataset.token_to_index,
    to_token=dataset.index_to_token,
    tempreture=0.95
)
# 输出结果
print("生成歌词:", generated)
```

##### 注意事项

1. **输入格式**：前缀文本需用 `separator`（默认"/"）分割分句开头，模型会按顺序生成连贯内容
2. **温度参数**：`tempreture` 越小生成结果越稳定（重复度可能较高），越大越多样（可能出现不合理字符）
3. **设备配置**：建议使用 GPU 训练，全量数据（2 万首歌词）训练约 20-30 小时（视 GPU 性能而定）
4. **生成长度**：`max_length` 控制最大生成字符数，达到该长度或生成 `<eos>` 标记时停止
5. **过拟合处理**：若训练损失低但生成质量差，可增大 `dropout` 或减少 `num_layers` 缓解过拟合
