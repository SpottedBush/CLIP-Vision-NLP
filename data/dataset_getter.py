import os
import zipfile
import torchvision.transforms as transforms
from torchvision import datasets

def get_fashion_product_data(data_path):
    kaggle_dataset = "paramaggarwal/fashion-product-images-small"
    dataset_zip_path = os.path.join(data_path, "fashion-product-images-small.zip")
    dataset_folder_path = os.path.join(data_path, "fashion-product-images-small")

    if not os.path.exists(dataset_folder_path):
        print("Dataset not found. Downloading from Kaggle...")
        os.system(f"kaggle datasets download -d {kaggle_dataset} -p {data_path}")
        
        with zipfile.ZipFile(dataset_zip_path, "r") as zip_ref:
            zip_ref.extractall(data_path)
        print("Dataset downloaded and extracted.")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    # Load dataset
    dataset = datasets.ImageFolder(root=dataset_folder_path, transform=transform)
    return dataset