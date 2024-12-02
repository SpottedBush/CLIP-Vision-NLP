# CLIP Implementation with Fashion Product Dataset

This project implements CLIP (Contrastive Language-Image Pretraining) from scratch using PyTorch and is tested on the Fashion Product Image Dataset.

## Installation

1. Clone this repository.
2. Install the required dependencies:

```
clip_project/
│
├── dataset/
│   └── dataset.py            # Dataset custom class and dataset_getter
│
├── models/
│   └── ImageEncoderClass.py  # The custom model for images (is also used to get the embeddings in the same dim as text features)
│
├── train/
│   └── clipLoss.py           # Loss functions
│   └── train_and_eval.py     # Training and evaluation function
│
├── utils/
│   └── helpers.py            # Helper functions (e.g., for preprocess)
│
├── main.ipynb                # Main notebook to understand how to use CLIP
├── requirements.txt          # List of dependencies
└── README.md                 # Project overview and instructions
```
