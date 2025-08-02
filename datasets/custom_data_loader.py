import os
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def get_data_loaders(data_dir, img_size=128, batch_size=32, train_dir="train",validation_dir="validation",test_dir="test"):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
    ])
    train_ds = ImageFolder(os.path.join(data_dir, train_dir), transform=transform)
    val_ds   = ImageFolder(os.path.join(data_dir, validation_dir), transform=transform)
    test_ds  = ImageFolder(os.path.join(data_dir, test_dir), transform=transform)

    return (
      DataLoader(train_ds, batch_size=batch_size, shuffle=True),
      DataLoader(val_ds, batch_size=batch_size, shuffle=False),
      DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    )
