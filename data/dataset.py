# dataset.py
import torchvision.transforms as transforms
from torchvision import datasets

def get_fashion_product_data(data_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    return dataset
