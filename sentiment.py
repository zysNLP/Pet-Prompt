#! -*- coding:utf-8 -*-
# 情感分析例子，利用MLM做 Zero-Shot/Few-Shot/Semi-Supervised Learning
# 参考项目：https://github.com/bojone/Pattern-Exploiting-Training
# 指标如下，由于没有固定随机化因子，因此下述指标可能略有波动
# zero-shot1: 0.8517/0.8437
# zero-shot2: 0.8811/0.8707
# few-shot:   0.8896/0.8910
# semi-sup:   0.8839/0.8783  # batch_size=16; epoch=7; bert-base-chinese-tf

import numpy as np
from bert4keras.backend import keras, K
from bert4keras.layers import Loss
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from keras.layers import Lambda, Dense

num_classes = 2
maxlen = 128
batch_size = 16

doc_path = "/home/ys/Documents/nvidia-bert/tensorflow1.15/bojone/GlobalPointer/pretrained_models/chinese_L-12_H-768_A-12/"
config_path = doc_path + 'bert_config.json'
checkpoint_path = doc_path + 'bert_model.ckpt'
dict_path = doc_path + 'vocab.txt'


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
train_data = train_data[:num_labeled]
train_data = train_data + unlabeled_data

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

# 对应的任务描述
prefix = u'很满意。'
mask_idx = 1
pos_id = tokenizer.token_to_id(u'很')
neg_id = tokenizer.token_to_id(u'不')


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


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_output_ids = [], [], []
        for is_end, (text, label) in self.sample(random):
            if label != 2:
                text = prefix + text
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            if random:
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
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_output_ids = sequence_padding(batch_output_ids)
                yield [
                    batch_token_ids, batch_segment_ids, batch_output_ids
                ], None
                batch_token_ids, batch_segment_ids, batch_output_ids = [], [], []


class CrossEntropy(Loss):
    """交叉熵作为loss，并mask掉输入部分
    """
    def compute_loss(self, inputs, mask=None):
        y_true, y_pred = inputs
        y_mask = K.cast(K.not_equal(y_true, 0), K.floatx())
        accuracy = keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
        accuracy = K.sum(accuracy * y_mask) / K.sum(y_mask)
        self.add_metric(accuracy, name='accuracy')
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss


# 加载预训练模型
model = build_transformer_model(
    config_path=config_path, checkpoint_path=checkpoint_path, with_mlm=True
)
model.summary()

# 训练用模型
y_in = keras.layers.Input(shape=(None,))
outputs = CrossEntropy(1)([y_in, model.output])
print(model.inputs)

train_model = keras.models.Model(model.inputs + [y_in], outputs)
train_model.compile(optimizer=Adam(1e-5))
train_model.summary()

# 转换数据集
train_generator = data_generator(train_data, batch_size)
for batch in train_generator:
    print(batch)
    break
valid_generator = data_generator(valid_data, batch_size)
test_generator = data_generator(test_data, batch_size)


class Evaluator(keras.callbacks.Callback):
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        model.save_weights('mlm_model.weights')
        val_acc = evaluate(valid_generator)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            model.save_weights('best_model.weights')
        test_acc = evaluate(test_generator)
        print(
            u'val_acc: %.5f, best_val_acc: %.5f, test_acc: %.5f\n' %
            (val_acc, self.best_val_acc, test_acc)
        )


def evaluate(data):
    total, right = 0., 0.
    for x_true, _ in data:
        x_true, y_true = x_true[:2], x_true[2]
        y_pred = model.predict(x_true)
        y_pred = y_pred[:, mask_idx, [neg_id, pos_id]].argmax(axis=1)
        y_true = (y_true[:, mask_idx] == pos_id).astype(int)
        total += len(y_true)
        right += (y_true == y_pred).sum()
    return right / total


if __name__ == '__main__':

    evaluator = Evaluator()

    train_model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=1000,
        callbacks=[evaluator]
    )

else:

    model.load_weights('best_model.weights')
