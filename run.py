# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif

"""
这是一段Python的argparse模块的代码，用于处理命令行参数
1. `--model`: 这是一个必需的参数，用户需要在命令行中提供它的值。用户可以选择'Bert'或'ERNIE'作为其值。
python script.py --model Bert
"""
parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: Bert, ERNIE')
args = parser.parse_args()


if __name__ == '__main__':
    dataset = 'THUCNews'  # 数据集

    """
    代码设置了随机种子以确保结果的可重复性。在实验中，我们经常需要设置随机种子以确保在不同运行中可以获取相同的随机数。
    `np.random.seed(1)`设置了NumPy的随机种子，
    `torch.manual_seed(1)`设置了PyTorch的CPU生成随机数的种子，
    而`torch.cuda.manual_seed_all(1)`则设置了PyTorch所有GPU生成随机数的种子。
    `torch.backends.cudnn.deterministic = True`这行代码强制使用确定性的算法，
    也就是每次运行网络时，同样的输入会得到完全相同的结果。这有助于保证每次运行实验时都能得到相同的结果，但可能会稍微降低运行速度。
    """
    model_name = args.model  # bert
    #`import_module`函数是Python的内建函数，它可以动态地加载一个模块
    x = import_module('models.' + model_name)
    config = x.Config(dataset)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样,确定性算法指的是一种给定确定的输入就会产生确定的输出，且每次运行过程都完全相同的算法。也就是说，同样的输入在任何时刻都会得到同样的输出结果。

    start_time = time.time()
    print("Loading data...")
    train_data, dev_data, test_data = build_dataset(config)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    model = x.Model(config).to(config.device)
    train(config, model, train_iter, dev_iter, test_iter)
