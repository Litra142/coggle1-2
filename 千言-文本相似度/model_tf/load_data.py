import pandas as pd
import numpy as np


def load_atec():
    """
    加载atec的数据集
    :return:
    """
    train = pd.read_table('input/atec/atec_nlp_sim_train.csv', header=None, encoding='utf-8-sig')  # (39346, 4)
    train.columns = ['line_num', 'q1', 'q2', 'label']

    train_add = pd.read_table('input/atec/atec_nlp_sim_train_add.csv', header=None,encoding='utf-8-sig')
    train_add.columns = ['line_num', 'q1', 'q2', 'label']
    train = pd.concat([train, train_add], axis=0)  # 按列 (102477, 4)
    return train


# load_atec()


def load_ccks():
    """
    读取ccks数据集
    :return:
    """
    train = []
    # 直接使用pandas读取train报错，估计是数据集里面分隔符不一致
    with open('input/ccks/task3_train.txt', 'r', encoding='utf-8') as train_file:
        for line in train_file:
            if len(line.strip().split('\t')) == 3:
                train.append(line.strip().split('\t'))
            else:
                print(line)
    train = pd.DataFrame(train)
    train.columns = ['q1', 'q2', 'label']
    train.label = train.label.astype('int')

    # 读取验证集
    dev = pd.read_table('input/ccks/task3_dev.txt', sep='\t',header=None)
    dev.columns = ['line', 'q1', 'q2']
    dev.drop(columns='line',inplace=True)
    dev.label = np.nan

    # 读取测试集
    test = pd.read_table('input/ccks/test_with_id.txt', sep='\t',header=None)
    test.columns = ['line', 'q1', 'q2']
    test.drop(columns='line',inplace=True)
    test['label'] = np.nan
    return train, dev, test  # (100000, 3) (9999, 3) (109999, 3)
