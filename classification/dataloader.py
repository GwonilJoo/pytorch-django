from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torch

class WaferDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, image, label):
      self.image = image
      self.label = label
      self.transform = transforms.ToTensor()

    def __len__(self):
      return len(self.image)

    def __getitem__(self, idx):
      x = self.image[idx]
      x = self.transform(x)
      y = self.label[idx]
      return x, y

def trainLoader(image, label, batch_size):
    dataset = WaferDataset(image, label)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

def testLoader(image, label, batch_size):
    dataset = WaferDataset(image, label)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)