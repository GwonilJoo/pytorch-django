import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from model import CNN
from dataloader import trainLoader, testLoader
from utils import loadDataset
import matplotlib.pyplot as plt
from tqdm import tqdm

def train(model, optimizer, train_loader, criterion, device):
    total_loss, total, correct = 0, 0, 0
    for image, label in tqdm(train_loader, desc="Train"):
        x = image.to(device)
        y = label.to(device)
        
        output = model(x)
        optimizer.zero_grad()
        
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        _, output_index = torch.max(output, 1)
        total_loss += loss.cpu().detach().numpy()
        total += label.size(0)

        correct += (output_index == y).sum().float()
    
    return total_loss / total, correct / total

def test(model, test_loader, criterion, device):
    with torch.no_grad():
        total_loss, total, correct = 0, 0, 0
        for image, label in tqdm(test_loader, desc="Test"):
            x = image.to(device)
            y = label.to(device)
        
            output = model.forward(x)
            loss = criterion(output, y)
            _, output_index = torch.max(output, 1)

            total_loss += loss.cpu().detach().numpy()
            total += label.size(0)
            correct += (output_index == y).sum().float()
    
    return total_loss / total, correct / total


if __name__ == "__main__":
    learning_rate = 0.001
    batch_size = 1024
    num_epoch = 15

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    image, label = loadDataset('../dataset/wafer.pkl', '../dataset/label.pkl')
    image_train, image_test, label_train, label_test = train_test_split(
        image, label, test_size = 0.33, random_state=42
    )
    
    train_loader = trainLoader(image_train, label_train, batch_size)
    test_loader = testLoader(image_test, label_test, batch_size)
    


    model = CNN().to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loss_arr, train_acc_arr = [], []
    test_loss_arr, test_acc_arr = [], []
    for epoch in range(num_epoch):
        train_loss, train_acc = train(model, optimizer, train_loader, loss_func, device)
        test_loss, test_acc = test(model, test_loader, loss_func, device)

        train_loss_arr.append(train_loss)
        train_acc_arr.append(train_acc)
        test_loss_arr.append(test_loss)
        test_acc_arr.append(test_acc)

        print(f'Epoch {epoch}/{num_epoch}')
        print(f'- loss: {train_loss} - acc: {train_acc}')
        print(f'- val_loss: {test_loss} - val_acc: {test_acc}')

        if test_acc == max(test_acc_arr):
            torch.save(model,'./best_acc.pkl')
        
        if test_loss == min(test_loss_arr):
            torch.save(model,'./best_loss.pkl')

    plt.plot(train_acc_arr)
    plt.plot(test_acc_arr)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'])
    plt.savefig('model_accuracy.png')

    plt.plot(train_loss_arr)
    plt.plot(test_loss_arr)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'])
    plt.savefig('model_loss.png')