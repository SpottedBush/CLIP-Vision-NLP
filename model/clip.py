# clip.py
from transformers import ViTModel, AutoModel, AutoTokenizer

def load_clip_model():
    img_model = ViTModel.from_pretrained("google/vit-base-patch16-224")

    model_name = "huawei-noah/TinyBERT_General_4L_312D"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    txt_model = AutoModel.from_pretrained(model_name)
    return img_model, txt_model, tokenizer
