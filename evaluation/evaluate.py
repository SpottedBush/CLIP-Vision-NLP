# evaluate.py
import torch
import numpy as np

def evaluate_clip(model, dataset, processor):
    model.eval()
    correct_image_to_text = 0
    correct_text_to_image = 0
    total = len(dataset)

    with torch.no_grad():
        for image, _ in dataset:
            # Preprocess images
            inputs = processor(images=image, return_tensors="pt", padding=True)
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            logits_per_text = outputs.logits_per_text
            
            predicted_text_idx = logits_per_image.argmax(dim=1)
            predicted_image_idx = logits_per_text.argmax(dim=1)

            correct_image_to_text += (predicted_text_idx == torch.arange(len(image)).to(predicted_text_idx.device)).sum().item()
            correct_text_to_image += (predicted_image_idx == torch.arange(len(image)).to(predicted_image_idx.device)).sum().item()

    image_to_text_acc = correct_image_to_text / total * 100
    text_to_image_acc = correct_text_to_image / total * 100

    print(f"Image-to-Text Accuracy: {image_to_text_acc:.2f}%")
    print(f"Text-to-Image Accuracy: {text_to_image_acc:.2f}%")
