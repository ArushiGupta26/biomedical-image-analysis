from medmnist import PathMNIST
from torchvision import transforms
from torch.utils.data import DataLoader

def make_loaders(batch_size=128, download=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])
    train_ds = PathMNIST(split="train", transform=transform, download=download)
    val_ds = PathMNIST(split="val", transform=transform, download=download)
    test_ds = PathMNIST(split="test", transform=transform, download=download)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader
