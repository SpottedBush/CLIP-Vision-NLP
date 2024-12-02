from transformers import ViTModel
import torch.nn as nn


class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = ViTModel.from_pretrained("google/vit-base-patch16-224", output_attentions=False)
        self.fc = nn.Linear(768, 312)
        

    def forward(self, x):
        outputs = self.model(x)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return self.fc(cls_embedding)