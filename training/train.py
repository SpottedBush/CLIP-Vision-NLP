import torch
import torch.nn.functional as F

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

def train_clip(model, dataset, processor, num_epochs=5, lr=1e-5):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        for image, _ in dataset:
            inputs = processor(images=image, return_tensors="pt", padding=True)
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            logits_per_text = outputs.logits_per_text

            loss = clip_contrastive_loss(logits_per_image, logits_per_text)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")
