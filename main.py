# main.py
from data.dataset import get_fashion_product_data
from model.clip import load_clip_model
from training.train import train_clip
from evaluation.evaluate import evaluate_clip

DATA_PATH = 'data/fashion_product_data/'

if __name__ == "__main__":

    dataset = get_fashion_product_data(DATA_PATH)
    model, processor = load_clip_model()
    train_clip(model, dataset, processor, num_epochs=5)
    evaluate_clip(model, dataset, processor)
