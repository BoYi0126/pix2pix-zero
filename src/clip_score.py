import torch
import clip
from PIL import Image
import os
import glob
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

def clip_text_score(image_path, text):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    text_token = clip.tokenize([text]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_token)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        similarity = (image_features @ text_features.T).item()

    return similarity

def clip_image_score(image_path1, image_path2):
    image1 = preprocess(Image.open(image_path1)).unsqueeze(0).to(device)
    image2 = preprocess(Image.open(image_path2)).unsqueeze(0).to(device)

    with torch.no_grad():
        feat1 = model.encode_image(image1)
        feat2 = model.encode_image(image2)

        feat1 = feat1 / feat1.norm(dim=-1, keepdim=True)
        feat2 = feat2 / feat2.norm(dim=-1, keepdim=True)

        similarity = (feat1 @ feat2.T).item()

    return similarity

def evaluate_folder(folder, target_text):
    scores = []

    for img_path in glob.glob(folder + "/*.png"):
        score = clip_text_score(img_path, target_text)
        scores.append(score)

    return np.mean(scores)
