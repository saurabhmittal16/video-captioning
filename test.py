import os
import torch
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from build_vocab import Vocabulary
from torchvision import transforms
from model import EncoderCNN, DecoderRNN

# Image loader function
def load_image(image_path, transform=None):
    image = Image.open(image_path).convert("RGB")
    image = image.resize([224, 224], Image.LANCZOS)

    if transform is not None:
        image = transform(image).unsqueeze(0)

    return image


# Load vocabulary wrapper
with open("./data/vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

# Build models
encoder = EncoderCNN(256).eval()
decoder = DecoderRNN(256, 512, len(vocab), 1)

# Load the trained model parameters
encoder.load_state_dict(torch.load("./models/encoder-5-3000.pkl"))
decoder.load_state_dict(torch.load("./models/decoder-5-3000.pkl"))

for filename in os.listdir("./images/"):
    image_tensor = load_image("./images/" + filename, transform)

    # Generate an caption from the image
    feature = encoder(image_tensor)
    sampled_ids = decoder.sample(feature)
    sampled_ids = sampled_ids[0].cpu().numpy()

    # Convert word_ids to words
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == "<end>":
            break
    sentence = " ".join(sampled_caption[1:-2])

    print(filename, sentence)
