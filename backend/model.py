import pandas as pd
import random
from diffusers import StableDiffusionPipeline
import torch
import os

# Load CSV
df = pd.read_csv("backend/dataset.csv")

# Load Stable Diffusion
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    use_safetensors=True
).to(device)

def get_letters():
    return sorted(df["Letter"].dropna().unique())

def get_words_by_letter(letter):
    words = df[df["Letter"] == letter]["Word"].dropna().unique().tolist()
    return words

def get_random_sentence(word):
    row = df[df["Word"] == word]
    if row.empty:
        return None
    sentences = row.iloc[0, 2:].dropna().tolist()
    return random.choice(sentences)

def generate_image(prompt):
    image = pipe(prompt).images[0]
    os.makedirs("static", exist_ok=True)
    output_path = "static/image.png"
    image.save(output_path)
    return output_path
