# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif
from pytorch_pretrained.optimization import BertAdam


# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if len(w.size()) < 2:
                continue
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def train(config, model, train_iter, dev_iter, test_iter):
    start_time = time.time()
    model.train()
    param_optimizer = list(model.named_parameters()) #获取模型的所有参数。

    """
    在神经网络中，权重衰减（也叫作L2正则化）是一种常用的防止过拟合的技术。它通过在模型的损失函数中加入参数权重的平方项，来抑制模型参数值过大的问题。

    然而，并不是所有的参数都需要进行权重衰减。在这段代码中，作者指定了'bias', 'LayerNorm.bias', 'LayerNorm.weight'这三类参数不需要进行权重衰减。这是因为进行权重衰减可能会对这些参数的学习造成不必要的干扰。例如：

    - 'bias'参数：偏置参数通常不需要进行权重衰减，因为偏置并不会对输入数据进行放大，因此也不会引起过拟合。

    - 'LayerNorm.bias', 'LayerNorm.weight'参数：Layer Normalization是一种正则化技术，它本身就可以抑制模型过拟合，经常不需要额外的权重衰减。
    """
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight'] # 定义不需要进行权重衰减的参数。
    #。权重衰减就是在损失函数中加入一项，这项是模型权重的L2范数乘以一个系数，这个系数就是权重衰减参数。在更新模型参数（即权重）时，权重衰减起到一种“惩罚”的作用，防止权重变得过大。在上述代码中，
    # 权重衰减参数是0.01，即在计算权重的梯度时，会在原有的梯度基础上减去0.01倍的权重值，从而实现对权重的“衰减”。
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    """     
        - `optimizer_grouped_parameters`：这是一个包含模型中需要优化的参数的列表。
        
        - `lr=config.learning_rate`：设定优化器的学习率。学习率是一个很重要的超参数，它决定了模型在学习过程中权重更新的步长。如果学习率过大，可能导致模型难以收敛；如果学习率过小，可能导致训练过程十分缓慢。这里的学习率是从配置对象中获取的。

        - `warmup=0.05`：设定优化器的warmup比例。在训练初期，学习率会从很小逐渐增大，直到达到设定的学习率，这个过程就叫做warmup。这样做可以防止模型在训练初期由于学习率过大而导致的不稳定。

        - `t_total=len(train_iter) * config.num_epochs`：设定优化器的总训练步数。这个值等于训练数据的总批次数（每个epoch的批次数乘以总的epoch数）。这个值用在学习率衰减上。     
        "epoch数"或"num_epochs"通常表示训练过程中数据集需要被遍历的次数。这是一个重要的超参数，需要根据具体的任务和数据集来设定。如果epoch数过小，模型可能会欠拟合，即在训练集上的性能不佳；如果epoch数过大，
        模型可能会过拟合，即在训练集上性能好，但在测试集或未见过的新数据上性能差。
    """
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=config.learning_rate,
                         warmup=0.05,
                         t_total=len(train_iter) * config.num_epochs)
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    model.train()
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i, (trains, labels) in enumerate(train_iter):
            outputs = model(trains)
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            if total_batch % 100 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    test(config, model, test_iter)


def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)
