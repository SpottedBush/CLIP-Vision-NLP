# main.py
from data.dataset import get_fashion_product_data
from model.clip import load_clip_model
from training.train import train_clip
from evaluation.evaluate import evaluate_clip

DATA_PATH = 'path_to_your_dataset'

if __name__ == "__main__":
    # Load dataset
    dataset = get_fashion_product_data(DATA_PATH)
    
    # Load CLIP model
    model, processor = load_clip_model()
    
    # Train the model
    train_clip(model, dataset, processor, num_epochs=5)
    
    # Evaluate the model
    evaluate_clip(model, dataset, processor)
