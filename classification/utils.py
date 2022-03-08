import os
import numpy as np
import pandas as pd
import gc
import matplotlib.pyplot as plt
from tqdm import tqdm

names = ('Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Near-full', 'Random', 'Scratch', 'none')

def oneHotEncoding(label):
    one_hot_vector = [[0] * len(names) for _ in range(len(label))]
    for i in range(len(label)):
        index = names.index(label[i][0])
        one_hot_vector[i][index] = 1
    one_hot_vector = np.array(one_hot_vector, dtype=np.float32)
    return one_hot_vector
    

def loadDataset(data_set, label_set):
    data_df = pd.read_pickle(data_set)
    label_df = pd.read_pickle(label_set)

    data = np.array(data_df)
    label = np.array(label_df)
    
    label = np.squeeze(label)
    for i, name in enumerate(names):
        label[label == name] = i
    label = np.array(label, dtype=np.int64)
    # label = oneHotEncoding(label)

    return data, label


if __name__ == "__main__":
    wafer_set = '../dataset/wafer.pkl'
    label_set = '../dataset/label.pkl'

    wafer, label = loadDataset(wafer_set, label_set)

    print(wafer.shape)
    print(label.shape)
    print(label[0])
    print(type(label[0]))

    # cases = np.unique(label)
    # for c in cases:
    #     print(f'{c}: {len(label[label == c])}')

    # gen_wafer = np.argmax(wafer, axis=3)
    # plt.imsave(f'{label[0][0]}.png', gen_wafer[0])