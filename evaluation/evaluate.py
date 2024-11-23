# evaluate.py
import torch
from torch.utils.data import DataLoader

def evaluate_clip(img_model, txt_model, tokenizer, dataset, batch_size=32):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    img_model.eval()
    txt_model.eval()

    correct_image_to_text = 0
    correct_text_to_image = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            texts = [str(label) for label in labels]  # Convert labels to strings

            # Tokenize text inputs
            tokenized_texts = tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True
            )
            
            # Move inputs to GPU if available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            images = images.to(device)
            tokenized_texts = {key: val.to(device) for key, val in tokenized_texts.items()}
            img_model.to(device)
            txt_model.to(device)

            # Forward pass
            image_features = img_model(images)
            text_features = txt_model(**tokenized_texts)

            # Compute logits
            logits_per_image = torch.matmul(image_features, text_features.T)
            logits_per_text = logits_per_image.T  # Symmetry

            # Predictions
            predicted_text_idx = logits_per_image.argmax(dim=1)
            predicted_image_idx = logits_per_text.argmax(dim=1)
            
            # Ground truth indices
            ground_truth_indices = torch.arange(len(images)).to(device)

            # Accuracy calculations
            correct_image_to_text += (predicted_text_idx == ground_truth_indices).sum().item()
            correct_text_to_image += (predicted_image_idx == ground_truth_indices).sum().item()
            total += len(images)

    # Calculate accuracies
    image_to_text_acc = correct_image_to_text / total * 100
    text_to_image_acc = correct_text_to_image / total * 100

    print(f"Image-to-Text Accuracy: {image_to_text_acc:.2f}%")
    print(f"Text-to-Image Accuracy: {text_to_image_acc:.2f}%")

