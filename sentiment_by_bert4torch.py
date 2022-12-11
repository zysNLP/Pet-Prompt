#! -*- coding:utf-8 -*-
# 情感分析例子，利用MLM做 Zero-Shot/Few-Shot/Semi-Supervised Learning
# 参考项目：https://github.com/bojone/Pattern-Exploiting-Training
# 增加FocalLoss前：
# semi-sup:   0.9024/0.8948


import torch
import torch.nn as nn
import numpy as np
from torchinfo import summary
from bert4torch.tokenizers import Tokenizer
from bert4torch.models import build_transformer_model
from torch.optim import Adam
import torch.nn.functional as F
from bert4torch.snippets import sequence_padding, ListDataset, Callback
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModelForSequenceClassification


num_classes = 2
maxlen = 128
batch_size = 16


# doc_path = "/home/ys/Documents/pretrained_models/torch/bert-base-chinese/"
doc_path = "/home/ys/Documents/pretrained_models/torch/chinese-roberta-wwm-ext/"
# doc_path = "/home/ys/Documents/pretrained_models/torch/PromptCLUE-base-v1-5/"
config_path = doc_path + 'config.json'
checkpoint_path = doc_path + 'pytorch_model.bin'
dict_path = doc_path + 'vocab.txt'

# tokenizer = AutoTokenizer.from_pretrained(doc_path)
# model = AutoModelForSequenceClassification.from_pretrained(doc_path)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
# choice = "zero-shot1"
choice = 'semi-sup'  # zero-shot1, zero-shot2, few-shot, semi-sup

def load_data(filename):
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            text, label = l.strip().split('\t')
            D.append((text, int(label)))
    return D

# 加载数据集
train_data = load_data('/home/ys/Documents/prompt/data/sentiment/sentiment.train.data')
valid_data = load_data('/home/ys/Documents/prompt/data/sentiment/sentiment.valid.data')
test_data = load_data('/home/ys/Documents/prompt/data/sentiment/sentiment.test.data')

# 模拟标注和非标注数据
train_frac = 0.01  # 标注数据的比例
num_labeled = int(len(train_data) * train_frac)
unlabeled_data = [(t, 2) for t, l in train_data[num_labeled:]]

if choice == 'zero-shot2':
    train_data = unlabeled_data  # 仅使用无监督数据继续mlm预训练
elif choice == 'few-shot':
    train_data = train_data[:num_labeled]  # 仅使用少量监督数据
elif choice == 'semi-sup':  # 少量监督数据和全量无监督数据做半监督
    train_data = train_data[:num_labeled]
    train_data = train_data + unlabeled_data

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

# 对应的任务描述
prefix = u'很满意。'
mask_idx = 1
pos_id = tokenizer.token_to_id(u'很')
neg_id = tokenizer.token_to_id(u'不')


def setup_seed(seed=0):
    import torch
    import os
    import numpy as np
    import random
    torch.manual_seed(seed)  # 为CPU设置随机种子
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    if torch.cuda.is_available():
        # torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子

setup_seed(seed=0)

def random_masking(token_ids):
    """对输入进行随机mask
    """
    rands = np.random.random(len(token_ids))
    source, target = [], []
    for r, t in zip(rands, token_ids):
        if r < 0.15 * 0.8:
            source.append(tokenizer._token_mask_id)
            target.append(t)
        elif r < 0.15 * 0.9:
            source.append(t)
            target.append(t)
        elif r < 0.15:
            source.append(np.random.choice(tokenizer._vocab_size - 1) + 1)
            target.append(t)
        else:
            source.append(t)
            target.append(0)
    return source, target

class MyDataset(ListDataset):
    def collate_fn(self, batch):
        batch_token_ids, batch_segment_ids, batch_output_ids = [], [], []
        for text, label in batch:
            if label != 2:
                text = prefix + text
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            if self.kwargs['random']:
                source_ids, target_ids = random_masking(token_ids)
            else:
                source_ids, target_ids = token_ids[:], token_ids[:]
            if label == 0:
                source_ids[mask_idx] = tokenizer._token_mask_id
                target_ids[mask_idx] = neg_id
            elif label == 1:
                source_ids[mask_idx] = tokenizer._token_mask_id
                target_ids[mask_idx] = pos_id
            batch_token_ids.append(source_ids)
            batch_segment_ids.append(segment_ids)
            batch_output_ids.append(target_ids)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids), dtype=torch.long, device=device)
        batch_segment_ids = torch.tensor(sequence_padding(batch_segment_ids), dtype=torch.long, device=device)
        batch_output_ids = torch.tensor(sequence_padding(batch_output_ids), dtype=torch.long, device=device)
        return [batch_token_ids, batch_segment_ids], batch_output_ids

# 加载数据集
train_dataset = MyDataset(data=train_data, random=True)
valid_dataset = MyDataset(data=valid_data, random=False)
test_dataset = MyDataset(data=test_data, random=False)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=valid_dataset.collate_fn)
test_dataloader = DataLoader(test_dataset,  batch_size=batch_size, collate_fn=test_dataset.collate_fn)

# 加载预训练模型
model = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, with_mlm=True, add_trainer=True).to(device)

class MyLoss(nn.CrossEntropyLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, y_preds, y_true):
        # y_true.shape: (batch_size, max_len)=torch.Size([16, 128])
        # y_true相当于一组batch中的所有字的tokenizer
        # y_pred=y_preds[1].shape=[16, 128, 21128]=[b,m,v]   # y_preds[0].shape=[16, 128, 768]没啥用
        # y_pred相当于对一组batch中的所有字m都预测了一个概率分布v; 首先取出来
        y_pred = y_preds[1]
        # 然后y_pred转成[b*m, v] = [2048, 21128]，相当于这组batch中所有字的概率分布矩阵
        y_pred = y_pred.reshape(-1, y_pred.shape[-1])
        loss = super().forward(y_pred, y_true.flatten())
        return loss

class MyCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(MyCrossEntropyLoss, self).__init__()

    def forward(self, y_preds, y_true):

        y_pred = y_preds[1]        # [batch_size, max_len, vocab_size]
        y_pred = y_pred.reshape(-1, y_pred.shape[-1])  # [batch_size*max_len, vocab_size]
        y_true = y_true.flatten()  # [batch_size*max_len, ]

        # 先经过softmax函数，求出每个类别的概率值，取值到0-1之间
        softmax = nn.Softmax()
        y_pred = softmax(y_pred)
        # 再经过log函数，取对数，原来的变化趋势保持不变，但所有值都会变成负的，
        # 原来概率大的，成为负值也大，但是它取绝对值后就是最小的，我们想要的是最小损失，正好贴合
        y_pred = torch.log(y_pred)

        # 取出每一个样本标签值处的概率
        loss = y_pred[range(len(y_pred)), y_true]
        # 求每个样本的标签处的预测值之和，然后取平均，变为正数
        loss = abs(sum(loss) / len(y_pred))
        return loss

class MyFocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None, ignore_index=-100):
        super(MyFocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, y_preds, y_true):
        y_pred = y_preds[1]
        y_pred = y_pred.reshape(-1, y_pred.shape[-1])
        y_true = y_true.flatten()

        # 先经过softmax函数，求出每个类别的概率值，取值到0-1之间
        softmax = nn.Softmax()
        y_pred = softmax(y_pred)
        y_pred_log = torch.log(y_pred)

        alpha = self.weight
        if alpha:
            y_pred = alpha * (1 - y_pred)**self.gamma * y_pred_log
        else:
            y_pred = (1 - y_pred)**self.gamma * y_pred_log

        loss = y_pred[range(len(y_pred)), y_true]
        loss = abs(sum(loss) / len(y_pred))
        return loss

class FocalLoss(nn.Module):
    def __init__(self,  gamma=2, weight=None, ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, y_preds, y_true):

        y_pred = y_preds[1]
        y_pred = y_pred.reshape(-1, y_pred.shape[-1])
        y_true = y_true.flatten()

        # 先经过softmax函数，求出每个类别的概率值，取值到0-1之间
        softmax = nn.Softmax()
        y_pred = softmax(y_pred)
        y_pred_log = torch.log(y_pred)

        y_pred = (1 - y_pred)**self.gamma * y_pred_log
        loss = F.nll_loss(y_pred, y_true, self.weight, ignore_index=self.ignore_index)
        return loss

# 输入shape=[batch_size, max_len]
# input_data = [next(iter(train_dataloader))[0]]
# summary(model, input_data=input_data)

class Evaluator(Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, global_step, epoch, logs=None):
        val_acc = self.evaluate(valid_dataloader)
        test_acc = self.evaluate(test_dataloader)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            model.save_weights('best_model.pt')
        print(f'[{choice}]  valid_acc: {val_acc:.4f}, test_acc: {test_acc:.4f}, best_val_acc: {self.best_val_acc:.4f}\n')

    @staticmethod
    def evaluate(data):
        # total: 记录所有句子中mask的总个数，因为每个句子中只有一个mask，循环遍历完成后，等于所有句子的个数
        # right: 记录所有句子中的mask预测正确的总个数
        total, right = 0., 0.
        for x_true, y_true in data:
            y_pred = F.softmax(model.predict(x_true)[1], dim=-1)
            # 经过softmax之后，y_pred的shape=(batch, max_len, vocab_size)=(16, 128, 21128)
            # 每组batch中有16个句子，每个句子有128个token，每个token的概率分布是vocab_size种可能
            # 预测时，对每个句子，取mask位置的token，找出它被预测为neg_id（不）和 pos_id（很）的单词的概率
            # 由于每个句子里只有一个mask，每个mask都有2种可能，argmax之前得到的维度是(batch_size, 2)
            # argmax(axis=1)，则是将预测为pos_id（很）的概率更大则变成1，否则为0
            y_pred = y_pred[:, mask_idx, [neg_id, pos_id]].argmax(axis=1)
            # 同样，y_true则是从所有 batch_size 个句子中，找到mask的预测为pos_id（很）的标签1
            y_true = (y_true[:, mask_idx] == pos_id).long()
            total += len(y_true)
            right += (y_true == y_pred).sum().item()
        return right / total

def params_adjust(model, train_dataloader):
    gammas = [1, 2]
    # 定义使用的loss和optimizer，这里支持自定义
    for gamma in gammas:
        print(f"gamma: {gamma} ==============")
        model.compile(
            # loss=MyLoss(ignore_index=-100),
            # loss=MyCrossEntropyLoss(),
            loss=FocalLoss(gamma=gamma, weight=None, ignore_index=0),
            # loss=MyFocalLoss(gamma=5, weight=None, ignore_index=0),
            optimizer=Adam(model.parameters(), lr=2e-5),
        )
        model.fit(train_dataloader, epochs=10, steps_per_epoch=None, callbacks=[evaluator])


if __name__ == '__main__':

    evaluator = Evaluator()
    if choice == 'zero-shot1':
        valid_acc = evaluator.evaluate(valid_dataloader)
        test_acc = evaluator.evaluate(test_dataloader)
        print(f'[{choice}]  valid_acc: {valid_acc:.4f}, test_acc: {test_acc:.4f}')
    else:
        # model.fit(train_dataloader, epochs=10, steps_per_epoch=None, callbacks=[evaluator])
        params_adjust(model=model, train_dataloader=train_dataloader)
else:
    model.load_weights('best_model.pt')
