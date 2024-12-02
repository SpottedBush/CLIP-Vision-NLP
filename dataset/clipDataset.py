import os
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from torchvision import transforms


class CLIPDataset(Dataset):
    """
    Custom dataset for CLIP training that loads images and corresponding text descriptions.
    """
    def __init__(self, image_dir, descriptions):
        """
        Args:
            image_dir (str): Path to the directory containing images.
            descriptions (dict): A dictionary mapping image filenames to text descriptions.
            transform (callable, optional): Transform to apply to the images.
            tokenizer (callable, optional): Tokenizer to process text descriptions.
        """
        self.image_dir = image_dir
        self.descriptions = descriptions
        self.image_files = list(descriptions.keys())
        self.transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        ])
        self.tokenizer = AutoTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")
    
    def __call__(self, image, descriptions):
        return self.__init__(image, descriptions)
    
    def __len__(self):
        """
        Returns the size of the dataset.
        """
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Returns a single item from the dataset.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            tuple: (image, encoded_text), where:
                - image is the processed image tensor.
                - encoded_text is a dictionary with tokenized text inputs.
        """
        # Get image and its associated description
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        text = self.descriptions[self.image_files[idx]]
        encoded_text = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=77,
        )

        # Encoded text contains input_ids and attention_mask as keys
        return image, {key: value.squeeze(0) for key, value in encoded_text.items()}
