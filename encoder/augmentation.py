import pickle
import numpy as np

import torch
from utils import getWaferDataset
from dataloader import WaferDataset
from model import AutoEncoder


def generateData(wafer, label):
    global model
    dataset = WaferDataset(wafer)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(wafer), shuffle=False, drop_last=False)
    
    encoded_image = model.encoder(next(iter(dataloader)))
    gen_image = np.zeros((1, 26, 26, 3), dtype=np.float32)
    for _ in range((2000 // len(wafer)) + 1):
        noised_encoded_image = encoded_image + torch.normal(mean=0, std=0.1, size=(len(encoded_image), 64, 13, 13))
        noised_gen_image = model.decoder(noised_encoded_image)
        noised_gen_image = noised_gen_image.detach().numpy().reshape((-1, 26, 26, 3))
        print(f'\ngen_image.shape: {gen_image.shape}')
        print(f'noised_gen_image.shape: {noised_gen_image.shape}\n')
        gen_image = np.concatenate((gen_image, noised_gen_image), axis=0)
    
    gen_label = np.full((len(gen_image), 1), label)
    return gen_image[1:], gen_label[1:]


def generateDataset(dataset, autoencoder):
    global model
    images, labels = getWaferDataset(dataset)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = AutoEncoder().to(device)
    model = torch.load(autoencoder)
    model.eval()

    faulty_case = np.unique(labels)
    for f in faulty_case:
        if f == 'none':
            continue

        print(f)
        gen_image, gen_label = generateData(images[np.where(labels==f)[0]], f)
        images = np.concatenate((images, gen_image), axis = 0)
        labels = np.concatenate((labels, gen_label))
    
    # delete choiced 'none' image
    none_idx = np.where(labels == 'none')[0]
    none_length = len(none_idx)
    none_idx = none_idx[np.random.choice(none_length, size=11000, replace=False)]

    images = np.delete(images, none_idx, axis=0)
    labels = np.delete(labels, none_idx, axis=0)

    return images, labels


if __name__ == "__main__":
    dataset = '../dataset/LSWMD.pkl'
    autoencoder = 'conv_autoencoder.pkl'

    images, labels = generateDataset(dataset, autoencoder)
    
    with open('../dataset/wafer.pkl', 'wb') as f:
        pickle.dump(images, f)

    with open('../dataset/label.pkl', 'wb') as f:
        pickle.dump(labels, f)