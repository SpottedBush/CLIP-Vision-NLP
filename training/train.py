import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


def cosine_similarity(image_features, text_features):
    """
    Computes the cosine similarity between image and text features.
    """
    image_features = F.normalize(image_features, p=2, dim=1)  # Normalize to [batch_size, feature_dim]
    text_features = F.normalize(text_features, p=2, dim=1)

    return torch.matmul(image_features, text_features.T)  # [batch_size, batch_size]    

def clip_contrastive_loss(logits_per_image, logits_per_text):
    """
    Computes the symmetric contrastive loss for CLIP.
    
    logits_per_image: Cosine similarities between image embeddings and text embeddings.
    logits_per_text: Cosine similarities between text embeddings and image embeddings.
    """
    # The target is simply the indices of the images/texts since each image corresponds to a unique text and vice versa
    batch_size = logits_per_image.size(0)
    labels = torch.arange(batch_size, dtype=torch.long, device=logits_per_image.device)

    # Cross-entropy loss for both image->text and text->image
    loss_image_to_text = F.cross_entropy(logits_per_image, labels)
    loss_text_to_image = F.cross_entropy(logits_per_text, labels)

    # Return the mean of both losses
    return (loss_image_to_text + loss_text_to_image) / 2

def train_clip(img_model, txt_model, tokenizer, dataset, num_epochs=5, batch_size=32):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(
        list(img_model.parameters()) + list(txt_model.parameters()), lr=5e-5
    )

    img_model.train()
    txt_model.train()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_model.to(device)
    txt_model.to(device)

    img_linear_layer = torch.nn.Linear(768, 32).to(device)
    txt_linear_layer = torch.nn.Linear(312, 32).to(device)

    for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
        epoch_loss = 0
        for batch in tqdm(dataloader, desc="Training Batches", leave=False):
            images, labels = batch
            texts = [str(label) for label in labels]
            
            tokenized_texts = tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True, max_length=77
            )
            
            images = images.to(device)
            tokenized_texts = {key: val.to(device) for key, val in tokenized_texts.items()}

            image_features = img_model(images).last_hidden_state.mean(dim=1)
            text_features = txt_model(**tokenized_texts).last_hidden_state.mean(dim=1)
            image_features = img_linear_layer(image_features)
            text_features = txt_linear_layer(text_features)

            logits_per_image = cosine_similarity(image_features, text_features)
            logits_per_text = logits_per_image.T

            loss = clip_contrastive_loss(logits_per_image, logits_per_text)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1} completed. Loss: {epoch_loss / len(dataloader)}")