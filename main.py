# main.py
from data.dataset_getter import get_fashion_product_data
from model.clip import load_clip_model
from training.train import train_clip
from evaluation.evaluate import evaluate_clip

DATA_PATH = 'data/'

if __name__ == "__main__":

    dataset = get_fashion_product_data(DATA_PATH)
    img_model, txt_model, tokenizer = load_clip_model()
    train_clip(img_model, txt_model, tokenizer, dataset, num_epochs=5)
    evaluate_clip(img_model, txt_model, tokenizer, dataset)
