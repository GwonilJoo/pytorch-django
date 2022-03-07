from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

class WaferDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, image):
      self.image = image
      self.transform = transforms.ToTensor()

    def __len__(self):
      return len(self.image)

    def __getitem__(self, idx):
      x = self.image[idx]
      x = self.transform(x)
      return x

class WaferLoader:
    def __init__(self, train_data, test_data, batch_size):
        self.train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
        self.test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True)