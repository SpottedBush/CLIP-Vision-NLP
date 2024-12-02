import torch
from torch.utils.data import DataLoader
from torch import optim
from models.ImageEncoderClass import ImageEncoder
from train.clipLoss import CLIPLoss
from dataset.clipDataset import CLIPDataset
from transformers import AutoModel
import numpy as np
from tqdm import tqdm

def train_and_eval_clip(image_dir, descriptions, epochs=1, batch_size=32, learning_rate=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset and DataLoader
    dataset = CLIPDataset(image_dir, descriptions)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Split dataset into train and test
    train_size = int(0.001 * len(dataset)) # 0.1% of the dataset, we tried with more but it was taking too long
    test_size = int(0.0001 * len(dataset))
    print(f"Train size: {train_size}, Test size: {test_size}")
    trash_size = len(dataset) - train_size - test_size
    train_dataset, test_dataset, trash_dataset = torch.utils.data.random_split(dataset, [train_size, test_size, trash_size])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Models
    image_encoder = ImageEncoder().to(device)
    text_encoder = AutoModel.from_pretrained("huawei-noah/TinyBERT_General_4L_312D", output_attentions=False).to(device)
    loss_fn = CLIPLoss().to(device)

    # Optimizer
    optimizer = optim.AdamW(list(image_encoder.parameters()) + list(text_encoder.parameters()), lr=learning_rate)

    # Training
    for epoch in tqdm(range(epochs), total=epochs):
        image_encoder.train()
        text_encoder.train()
        epoch_loss = 0
        for images, encoded_text in train_dataloader:
            images = torch.tensor(np.stack(images))
            images = images.to(device)
            input_ids = encoded_text["input_ids"].squeeze(1).to(device)
            attention_mask = encoded_text["attention_mask"].squeeze(1).to(device)

            optimizer.zero_grad()

            image_features = image_encoder(images)
            text_features = text_encoder(input_ids, attention_mask).last_hidden_state[:, 0, :]
            loss = loss_fn(image_features, text_features)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader)}")
        
    # Evaluation
    image_encoder.eval()
    text_encoder.eval()
    eval_loss = 0
    with torch.no_grad():
        for images, encoded_text in test_dataloader:
            images = torch.tensor(np.stack(images))
            images = images.to(device)
            input_ids = encoded_text["input_ids"].squeeze(1).to(device)
            attention_mask = encoded_text["attention_mask"].squeeze(1).to(device)

            image_features = image_encoder(images)
            text_features = text_encoder(input_ids, attention_mask).last_hidden_state[:, 0, :]
            loss = loss_fn(image_features, text_features)

            eval_loss += loss.item()

    print(f"Validation Loss: {eval_loss / len(test_dataloader)}")
    return image_encoder, text_encoder