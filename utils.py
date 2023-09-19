# coding: UTF-8
import torch
from tqdm import tqdm
import time
from datetime import timedelta

PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号


"""
加载文本数据
对文本数据进行分词tokenizer
并为分词后的词条创建唯一的id
根据pad_size对token_ids进行padding和裁剪
"""
def build_dataset(config):

    def load_dataset(path, pad_size=32):
        contents = []
        with open(path, 'r', encoding='UTF-8') as f: 
            for line in tqdm(f):     # `for line in tqdm(f):` 使用`tqdm`包装文件对象以显示一个进度条。
                lin = line.strip()   #`lin = line.strip()` 移除每行文本前后的空白。
                if not lin:
                    continue
                content, label = lin.split('\t') #. `content, label = lin.split('\t')` 通过制表符（'\t'）tab将每行拆分为内容（`content`）和标签（`label`）。
                token = config.tokenizer.tokenize(content) #使用在`config`中指定的分词器将内容分词。
                token = [CLS] + token #在分词结果前添加`CLS`标记，它常用于BERT等模型，表示一个句子的开始。
                seq_len = len(token) #`seq_len = len(token)` 计算分词后的序列长度。
                mask = [] #`mask = []` 初始化一个空的mask列表，这在某些模型（如BERT）中用于忽略（或“掩盖”）输入中的某些元素。
                token_ids = config.tokenizer.convert_tokens_to_ids(token) #将词条转化为它们对应的id
                #在一些NLP任务中，如文本分类、情感分析、命名实体识别、问答系统等，首先需要对文本进行预处理，这通常就包括分词和转换词条为ID这两步。分词是将文本划分为独立的词条，
                # 而将词条转化为它们对应的id，是因为机器学习模型不能直接处理文本数据，需要将文本数据转换为可以输入模型的数值形式。在这个过程中，每个独立的词条被赋予一个独特的id。
                # BERT模型通常使用WordPiece分词器，该分词器能有效处理各种语言中未知的单词。WordPiece分词会尝试将未知单词分解成已知的子词
                
                """
                在许多NLP模型（包括BERT在内）中，输入序列需要有相同的长度。然而实际上不同的文本长度各异，我们需要进行一些处理使得它们长度一致。
                这个操作就叫做填充(padding)。填充就是在短的序列后边添加0，使其长度达到我们设定的最大长度(`pad_size`)
                如果序列长度小于`pad_size`，我们在`token_ids`后面添加0，同时对应的mask中也添加0（在自然语言处理中，通常用1表示真实的词条，用0表示填充部分）。
                然而，如果序列长度大于我们设定的最大长度`pad_size`，我们则需要裁剪这个序列，只保留前`pad_size`个词条。相应的，mask也只保留前`pad_size`个1：
                """
                if pad_size:
                    if len(token) < pad_size:
                        mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                        token_ids += ([0] * (pad_size - len(token)))
                    else:
                        mask = [1] * pad_size
                        token_ids = token_ids[:pad_size]
                        seq_len = pad_size
                contents.append((token_ids, int(label), seq_len, mask))
        return contents
    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    return train, dev, test


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        return (x, seq_len, mask), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))
