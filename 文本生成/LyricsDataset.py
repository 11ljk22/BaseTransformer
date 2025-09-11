import torch
import pandas as pd
from torch.nn.functional import pad
from torch.utils.data import Dataset


class LyricsDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        separator: str = "，",
        nrows: int = 300,
        batch_size: int = 32,
    ):
        super().__init__()
        self.separator = separator
        self.nrows = nrows
        self.batch_size = batch_size

        self.token_to_index, self.index_to_token, self.chars, self.data = (
            self.get_dictionary(file_path)
        )

        self.index_data = self.get_index_data()
        self.batch_max_length = self.get_batch_max_length()

    def get_batch_max_length(self):
        num_samples = self.__len__()
        batch_length = []
        if num_samples % self.batch_size == 0:
            num_batch = num_samples // self.batch_size
        else:
            num_batch = (num_samples // self.batch_size) + 1

        for batch in range(num_batch):
            batch_df = self.df.iloc[
                batch * self.batch_size : (batch + 1) * self.batch_size, :
            ]
            batch_max_length = len(max(line for line in batch_df.values).item())
            batch_length.append(batch_max_length)
        return batch_length

    def get_index_data(self):
        index_data = []
        for tokens in self.data:
            # 确保 tokens 是列表形式
            tokens = list(tokens)

            tmp = [self.token_to_index["<bos>"]]
            for token in tokens:
                # 添加对未知token的处理
                token_idx = self.token_to_index.get(token)
                if token_idx is None:
                    token_idx = self.token_to_index["<unk>"]
                tmp.append(token_idx)
            tmp.append(self.token_to_index["<eos>"])
            index_data.append(tmp)
        return index_data

    def read_file(self, file_path):
        if self.nrows >= 0:
            df = pd.read_csv(file_path, nrows=self.nrows)
        else:
            df = pd.read_csv(file_path)
        self.df = df
        # 返回字符列表和原始行
        data = df.values.reshape(-1)
        return [char for line in data for char in line], data

    def get_dictionary(self, file_path: str):
        chars, data = self.read_file(file_path)
        index_to_token = [
            "<unk>",
            "<bos>",
            "<eos>",
            "<pad>",
            self.separator,
        ]

        token_to_index = {}
        # 先添加需要保留的token
        for idx, token in enumerate(index_to_token):
            token_to_index[token] = idx

        # 添加数据中的唯一token
        for char in chars:
            if char not in token_to_index:
                index_to_token.append(char)
                token_to_index[char] = len(index_to_token) - 1

        return token_to_index, index_to_token, chars, data

    def __getitem__(self, index):
        batch = index // self.batch_size
        batch_max_length = self.batch_max_length[batch]

        original_seq = self.index_data[index]
        # 输入序列：去掉 <eos>
        src_seq = original_seq[:-1]
        src_padded = pad(
            torch.tensor(src_seq, dtype=torch.long),
            (0, batch_max_length - len(src_seq)),
            value=self.token_to_index["<pad>"],
        )

        # 目标序列：去掉 <bos>
        tgt_seq = original_seq[1:]
        tgt_padded = pad(
            torch.tensor(tgt_seq, dtype=torch.long),
            (0, batch_max_length - len(tgt_seq)),
            value=self.token_to_index["<pad>"],
        )

        return src_padded, tgt_padded

    def __len__(self):
        return len(self.index_data)


if __name__ == "__main__":
    batch_size = 32
    dataset = LyricsDataset(
        "./data/generate/lyrics.csv", batch_size=batch_size, nrows=10000
    )
    from torch.utils.data import DataLoader

    print(dataset.batch_max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    for src, tgt in dataloader:
        print(src.shape, tgt.shape)
