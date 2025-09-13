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

    def get_index_data(self):
        index_data = []
        for tokens in self.data:
            # 确保 tokens 是列表形式
            tokens = list(tokens)

            tmp = [self.token_to_index["<bos>"]]
            for token in tokens:
                # 添加对未知token的处理
                token_idx = self.token_to_index.get(token, self.token_to_index["<unk>"])
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
        # 我们使用collate_fn将批次中的所有样本补充到同一个长度
        original_seq = self.index_data[index]
        src_seq = original_seq[:-1]
        tgt_seq = original_seq[1:]
        return torch.tensor(src_seq, dtype=torch.long), torch.tensor(
            tgt_seq, dtype=torch.long
        )

    def __len__(self):
        return len(self.index_data)

    def collate_fn(self, batch):
        """动态批次长度，这个函数可被指定为DataLoader数据加载器的collate_fn参数"""
        batch_max_length = max(max(len(src), len(tgt)) for src, tgt in batch)
        batch_src, batch_tgt = [], []
        pad_idx = self.token_to_index["<pad>"]
        for src, tgt in batch:
            batch_src.append(pad(src, (0, batch_max_length - len(src)), value=pad_idx))
            batch_tgt.append(pad(tgt, (0, batch_max_length - len(tgt)), value=pad_idx))
        return torch.stack(batch_src), torch.stack(batch_tgt)


if __name__ == "__main__":
    batch_size = 32
    dataset = LyricsDataset(
        "./data/generate/lyrics.csv", batch_size=batch_size, nrows=100
    )
    from torch.utils.data import DataLoader

    pad_idx = dataset.token_to_index["<pad>"]
    print(pad_idx)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=dataset.collate_fn,
    )
    for src, tgt in dataloader:
        print(src.shape, tgt.shape)
