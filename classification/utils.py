import os
import numpy as np
import pandas as pd
import gc
import matplotlib.pyplot as plt
from tqdm import tqdm

def loadDataset(data_set, label_set):
    data_df = pd.read_pickle(data_set)
    label_df = pd.read_pickle(label_set)

    data = np.array(data_df, dtype=np.float32)
    label = np.array(label_df)

    return data, label


if __name__ == "__main__":
    wafer_set = '../dataset/wafer.pkl'
    label_set = '../dataset/label.pkl'

    wafer, label = loadDataset(wafer_set, label_set)

    cases = np.unique(label)
    for c in cases:
        print(f'{c}: {len(label[label == c])}')

    gen_wafer = np.argmax(wafer, axis=3)
    plt.imsave(f'{label[0][0]}.png', gen_wafer[0])