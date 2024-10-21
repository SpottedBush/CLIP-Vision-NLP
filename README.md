# CLIP Implementation with Fashion Product Dataset

This project implements CLIP (Contrastive Language-Image Pretraining) from scratch using PyTorch and is tested on the Fashion Product Image Dataset.

## Installation

1. Clone this repository.
2. Install the required dependencies:

clip_project/
│
├── data/
│   └── dataset.py            # Dataset loading and preprocessing
│
├── model/
│   └── clip.py               # CLIP model implementation
│
├── training/
│   └── train.py              # Training loop and loss functions
│
├── evaluation/
│   └── evaluate.py           # Evaluation functions
│
├── utils/
│   └── helpers.py            # Helper functions (e.g., for visualization)
│
├── main.py                   # Main script to run the project
├── requirements.txt          # List of dependencies
└── README.md                 # Project overview and instructions
