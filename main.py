# main.py
from data.dataset_getter import get_fashion_product_data
from model.clip import load_clip_model
from training.train import train_clip
from evaluation.evaluate import evaluate_clip
import sys
import torch

DATA_PATH = 'data/'

if __name__ == "__main__":
    dataset = get_fashion_product_data(DATA_PATH)
    if len(sys.argv) != 1 and len(sys.argv) != 3:
        print("Usage: python main.py [img_model_path txt_model_path tokenizer_path] or python main.py")
    
    elif len(sys.argv) == 3:
        print("Loading model from", sys.argv[1], "and", sys.argv[2])
        img_model = torch.load(sys.argv[1])
        txt_model = torch.load(sys.argv[2])
        tokenizer = torch.load(sys.argv[3])
        train_clip(img_model, txt_model, tokenizer, dataset, num_epochs=5)
        
    else:
        print("Training model from scratch")
        img_model, txt_model, tokenizer = load_clip_model()
        train_clip(img_model, txt_model, tokenizer, dataset, num_epochs=5)

    evaluate_clip(img_model, txt_model, tokenizer, dataset)
