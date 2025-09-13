# BaseTransformer：基础 Transformer 及其相关模型变体

## 项目简介

本项目是面向 **Transformer 初学者**的入门级学习实践，通过手动实现基础架构的 Transformer 文本翻译模型（中英方向）以及 Transformer （文本翻译）等模型变体，帮助初学者理解 Transformer 的核心原理（尤其是多头注意力机制），并对比 PyTorch 官方库接口的使用方式。

项目定位为「学习型 Demo」，不追求生产级性能，而是聚焦于**模型架构拆解、关键模块实现细节、训练推理流程梳理**，适合刚接触 Transformer 的开发者入门参考。

## 数据集说明

### 数据来源

文本翻译使用的基准数据集来源于国际顶级机器翻译评测任务 **WMT 2021 News Translation Task**（[官方链接](https://www.statmt.org/wmt21/translation-task.html)），该数据集由多个权威平行语料库整合而成，包括：

- ParaCrawl（网页平行语料）
- News-Commentary（新闻评论平行语料）
- Wiki-Titles（维基标题平行语料）
- UN Parallel Corpus（联合国平行语料）
- WikiMatrix（维基百科平行语料）
- CCMT（中文机器翻译评测语料）

原始数据集包含 **2500 万+ 中英双语句对**，覆盖新闻、百科、官方文档等多种场景，具备良好的语言多样性。

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

### 项目使用数据集

对于文本翻译项目，为平衡「演示效率」与「模型泛化能力」，项目提供两个子数据集，存放于 `data/translate/` 目录下：
| 文件名 | 数据量 | 用途 |
|--------|--------|------|
| `TranslationData.csv` | 2 万条双语句对 | 基础演示、快速验证模型流程（训练耗时短，适合调试） |
| `2m_WMT21.csv` | 200 万条双语句对 | 提升模型泛化能力（数据量更大，训练后模型适配更多语言场景） |

> 注：所有数据集已预处理为「中文句子（空格分词）,英文句子（空格分词）」的 CSV 格式，可直接通过`TranslationDataset`类加载。

对于文本生成所用到的歌词文件，项目已将源 json 中的所有中文歌词提取出来，并存放于`data/generate/` 目录下

## 代码结构与说明

项目核心代码按「功能模块」拆分，结构清晰，便于针对性学习：

| 代码文件                         | 核心功能                      | 学习重点                                                                                                                                                             |
| -------------------------------- | ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `TranslationDataset.py`          | 数据集加载与预处理            | 1. 文本 → 索引的映射（构建词汇表，含`<bos>/<eos>/<pad>/<unk>`特殊标记）<br>2. 序列填充/截断（适配模型输入长度）<br>3. 批量数据生成（兼容 DataLoader）                |
| `BaseTransformer.ipynb`          | 手动实现 Transformer 完整架构 | 1. 位置编码（正余弦编码）<br>2. 多头注意力（自注意力+交叉注意力，含因果掩码/填充掩码）<br>3. 编码器层/解码器层（残差连接+层归一化+前馈网络）<br>4. 完整训练/推理流程 |
| `TorchOfficialTransformer.ipynb` | PyTorch 官方库接口实现        | 1. 官方`nn.Transformer`的参数配置（与手动实现对齐）<br>2. 官方接口与自定义模块的结合（如自定义嵌入+位置编码）<br>3. 高效调用技巧（如自动因果掩码生成）               |

## 核心功能与学习路径

### 1. 基础学习：手动实现 Transformer

通过 `BaseTransformer.ipynb` 可深入理解 Transformer 的「从 0 到 1」构建过程，关键模块拆解：

- **嵌入与位置编码**：解决文本离散输入的向量表示，补充序列位置信息
- **多头注意力**：拆解注意力计算流程（QKV 投影、缩放、掩码、Softmax、加权求和），理解「多头」如何捕捉多维度关联
- **编码器/解码器堆叠**：学习残差连接与层归一化的作用（缓解梯度消失，稳定训练）
- **训练逻辑**：适配翻译任务的「错位输入-目标」设计

### 2. 进阶对比：官方库接口使用

通过 `TorchOfficialTransformer.ipynb` 对比手动实现与官方接口的差异：

- 官方库的参数映射（如`nhead`、`dim_feedforward`与手动实现的对应关系）
- 官方优化（如`batch_first=True`适配批量输入格式，`generate_square_subsequent_mask`简化掩码生成）
- 自定义模块（如位置编码）与官方接口的兼容方式

### 3. 实践验证：单样本过拟合与翻译推理

项目提供完整的训练与推理代码，支持：

- 小样本过拟合（快速验证模型是否能学习简单映射）
- 批量训练（使用 200 万数据集提升泛化能力）
- 自回归翻译推理（输入中文句子，输出英文翻译结果）

## 环境依赖

```bash
# 核心依赖
torch==2.0+  # 建议2.0及以上版本，根据设备选择 CPU/GPU 版本
pandas==1.5+ # 加载CSV数据集
matplotlib==3.7+ # 绘制训练损失曲线
```

## 使用说明

### 1. 数据准备

将数据集文件（`TranslationData.csv`/`2m_WMT21.csv`）放入 `data/translate/` 目录，确保路径与代码中 `file_path` 参数一致（如 `../data/translate/TranslationData.csv`）。

### 2. 模型训练

#### 方式 1：手动实现模型训练（基础学习）

```python
dataset = TranslationDataset(file_path="../data/TranslationData.csv", max_lines=20000)
model = BaseTransformer(src_vocab_size=len(dataset.ch_token_to_index), ...)
# 配置优化器、损失函数、学习率调度器后启动训练
```

#### 方式 2：官方库模型训练（进阶对比）

```python
dataset = TranslationDataset(file_path="../data/2m_WMT21.csv", max_lines=2000000)
model = ChEnTransformer(src_vocab_size=len(dataset.ch_token_to_index), ...)
# 复用手动实现的训练逻辑，但需替换模型实例以及传参方式（详见代码）
```

### 3. 翻译推理

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

## 注意事项

1. **分词格式**：输入中文句子需按数据集格式进行「空格分词」，否则会被识别为`<unk>`（未知词）
2. **设备配置**：训练建议使用 GPU（`device="cuda"`），2 万条数据训练约 10 分钟，200 万数据建议使用多 GPU 或调整 batch_size
3. **超参数调整**：若训练不稳定，可调整`d_model`（需为`nhead`的整数倍）、`dropout`（防止过拟合）
4. **检查点**：项目提供`save_checkpoint`/`load_checkpoint`函数，支持中断后继续训练

## 学习资源推荐

1. Transformer 原论文：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)
2. PyTorch 官方文档：[nn.Transformer](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html)
3. WMT 2021 任务说明：[WMT21 Translation Task Overview](https://www.statmt.org/wmt21/pdf/2021.wmt-1.100.pdf)

通过本项目的手动实现与官方接口对比，可快速建立对 Transformer 的直观理解，为后续学习打下基础。
