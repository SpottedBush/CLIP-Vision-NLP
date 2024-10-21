# helpers.py
import matplotlib.pyplot as plt

def display_image(image):
    plt.imshow(image.permute(1, 2, 0))
    plt.show()
