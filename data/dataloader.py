from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import torch

def get_dataloaders(batch_size=32):
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    train_dataset = ImageFolder(root='./dataset/tiny-imagenet/tiny-imagenet-200/train', transform=transform)
    val_dataset = ImageFolder(root='./dataset/tiny-imagenet/tiny-imagenet-200/val', transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader
