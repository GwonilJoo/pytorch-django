from cProfile import label
import numpy as np
import pandas as pd

def find_dim(x):
    dim0 = np.size(x, axis=0)
    dim1 = np.size(x, axis=1)
    return dim0, dim1


def getWaferDataset(dataset):
    # load dataset
    df = pd.read_pickle(dataset)  

    # find wafer dimension
    df['waferMapDim'] = df.waferMap.apply(find_dim)

    # 26 x 26인 wafer 추출
    df = df.loc[df['waferMapDim'] == (26, 26)]
    
    data = np.ones((1, 26, 26))
    label = list()

    for i in range(len(df)):
        if len(df.iloc[i,:]['failureType']) == 0:
            continue
        data = np.concatenate((data, df.iloc[i,:]['waferMap'].reshape(1, 26, 26)))
        label.append(df.iloc[i,:]['failureType'][0][0])

    data = data[1:]
    label = np.array(label).reshape((-1, 1))

    # add channel
    data = data.reshape((-1, 26, 26, 1))

    # make image
    image = np.zeros((len(data), 26, 26, 3), dtype=np.float32)
    for w in range(len(data)):
        for i in range(26):
            for j in range(26):
                image[w, i, j, int(data[w, i, j])] = 1

    return image, label