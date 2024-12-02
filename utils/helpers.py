import matplotlib.pyplot as plt
from torchvision.transforms import Resize, ToTensor, Normalize


def display_image(image):
    plt.imshow(image.permute(1, 2, 0))
    plt.show()

def preprocess_image(image):
    image = Resize((224, 224))(image)
    image = ToTensor()(image)
    image = Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])(image)
    return image

def tokenize_text(text, tokenizer):
    text = tokenizer(text, return_tensors="pt", padding="max_length", max_length=77, truncation=True)
    return text