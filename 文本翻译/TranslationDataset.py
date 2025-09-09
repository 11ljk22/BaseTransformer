import torch
import pandas as pd
from torch import nn
from torch.utils.data import Dataset
from torch.nn.functional import pad


class TranslationDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        max_lines: int = 300,
    ):
        super().__init__()
        self.max_lines = max_lines  # 读取的数据行数

        (
            self.ch_token_to_index,  # 将单个中文词转化为对应的索引
            self.ch_index_to_token,  # 将单个中文索引转化为对应的词
            self.ch_data,  # 一个列表，每个列表中是一条中文数据（对应原始数据的一行）
            self.en_token_to_index,  # 将单个英文词转化为对应的索引
            self.en_index_to_token,  # 将单个英文索引转化为对应的词
            self.en_data,  # 一个列表，每个列表中是一条英文数据（对应原始数据的一行）
        ) = self.get_dictionary(file_path)

        self.ch_index_data = self.get_index_data(
            self.ch_data, self.ch_token_to_index
        )  # ch_index_data:将ch_data中的每一个元素用相应的索引表示所形成的列表
        self.en_index_data = self.get_index_data(
            self.en_data, self.en_token_to_index
        )  # en_index_data:将en_data中的每一个元素用相应的索引表示所形成的列表

        # 为 <bos> 和 <eos> 预留空间,所以加 2
        self.ch_max_length = (max(len(tokens) for tokens in self.ch_index_data)) + 2
        self.en_max_length = (max(len(tokens) for tokens in self.en_index_data)) + 2

    def get_index_data(self, data, to_index):
        """将传入的data中的每一个元素转化为对应的索引表示"""
        res = []
        for line in data:
            tmp = []
            for word in line.split(" "):
                tmp.append(to_index[word])
            res.append(tmp)
        return res

    def read_file(self, file_path):
        if self.max_lines >= 0:
            df = pd.read_csv(file_path, nrows=self.max_lines)  # 指定读取的行数
        else:
            df = pd.read_excel(file_path)
        chinese_data = df.values[:, 0]
        english_data = df.values[:, 1]
        return chinese_data, english_data

    def get_dictionary(self, file_path: str):
        chinese_data, english_data = self.read_file(file_path)

        def add_dictionary(data):
            index_to_token = [
                "<unk>",
                "<bos>",
                "<eos>",
                "<pad>",
            ]  # <unk>表示未知词元,<bos>表示序列的开始,<eos>表示序列的结束,<pad>表示填充
            token_to_index = {}
            # 先添加上面这4个token
            for idx, token in enumerate(index_to_token):
                token_to_index[token] = idx

            # 再添加数据中的唯一token
            for line in data:
                for word in line.split(" "):
                    if word not in token_to_index:
                        index_to_token.append(word)
                        token_to_index[word] = len(index_to_token) - 1

            return token_to_index, index_to_token, data

        ch_token_to_index, ch_index_to_token, ch_data = add_dictionary(chinese_data)
        en_token_to_index, en_index_to_token, en_data = add_dictionary(english_data)
        return (
            ch_token_to_index,
            ch_index_to_token,
            ch_data,
            en_token_to_index,
            en_index_to_token,
            en_data,
        )

    def __getitem__(self, index):
        # 在序列开头添加<bos>,在序列最后添加<eos>
        src = self.ch_index_data[index].copy()
        src.insert(0, self.ch_token_to_index["<bos>"])
        src.append(self.ch_token_to_index["<eos>"])

        tgt = self.en_index_data[index].copy()
        tgt.insert(0, self.en_token_to_index["<bos>"])
        tgt.append(self.en_token_to_index["<eos>"])
        # 依据self.ch_max_length,将插入<bos>,<eos>后的src序列用<pad>填充到self.ch_max_length长度
        src_padded = pad(
            torch.tensor(src),
            pad=(0, self.ch_max_length - len(src)),
            mode="constant",
            value=self.ch_token_to_index["<pad>"],
        )
        # 依据self.en_max_length,将插入<bos>,<eos>后的tgt序列用<pad>填充到self.en_max_length长度
        tgt_padded = pad(
            torch.tensor(tgt),
            pad=(0, self.en_max_length - len(tgt)),
            mode="constant",
            value=self.en_token_to_index["<pad>"],
        )

        return src_padded, tgt_padded

    def __len__(self):
        # 返回数据集的总元素个数
        return len(self.ch_index_data)


if __name__ == "__main__":
    dataset = TranslationDataset("./data/translation_dataset.csv", max_lines=1)
    print(dataset.__getitem__(0))
    print()
    print(dataset.ch_index_data[0])
