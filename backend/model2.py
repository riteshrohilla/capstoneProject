# import pandas as pd
# import random
# from diffusers import StableDiffusionPipeline
# import torch
# import os
# from pathlib import Path
# import logging

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Configuration
# BASE_DIR = Path(__file__).resolve().parent
# DATASET_PATH = BASE_DIR.parent / "dataset.csv"
# IMAGE_OUTPUT_DIR = BASE_DIR / "static" / "images"

# # Ensure output directory exists
# IMAGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# # Load dataset
# def load_dataset():
#     if not DATASET_PATH.exists():
#         logger.error(f"Dataset not found at {DATASET_PATH}")
#         raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
    
#     logger.info("Loading dataset...")
#     df = pd.read_csv(DATASET_PATH)
#     logger.info(f"Dataset loaded with {len(df)} records")
#     return df

# df = load_dataset()

# # Load Stable Diffusion model
# def load_model():
#     logger.info("Loading Stable Diffusion model...")
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     logger.info(f"Using device: {device}")
    
#     dtype = torch.float16 if device == "cuda" else torch.float32
#     logger.info(f"Using dtype: {dtype}")
    
#     pipe = StableDiffusionPipeline.from_pretrained(
#         "runwayml/stable-diffusion-v1-5",
#         torch_dtype=dtype,
#         use_safetensors=True
#     ).to(device)
    
#     logger.info("Model loaded successfully")
#     return pipe

# pipe = load_model()

# def get_letters():
#     """Get sorted list of unique starting letters from dataset"""
#     letters = df["Letter"].dropna().str.upper().unique()
#     return sorted(letters.tolist())

# def get_words_by_letter(letter: str):
#     """Get words starting with the specified letter"""
#     if len(letter) != 1 or not letter.isalpha():
#         raise ValueError("Input must be a single letter")
    
#     letter = letter.upper()
#     words = df[df["Letter"].str.upper() == letter]["Word"].dropna().unique()
#     return sorted([word for word in words if isinstance(word, str)])

# def get_random_sentence(word: str):
#     """Get a random sentence containing the specified word"""``
#     if not word or not isinstance(word, str):
#         return None
    
#     word = word.lower()
#     matching_rows = df[df["Word"].str.lower() == word]
    
#     if matching_rows.empty:
#         logger.warning(f"No matching rows found for word: {word}")
#         return None
    
#     # Get all non-empty sentences from the row
#     sentences = []
#     for col in df.columns[2:]:  # Skip Letter and Word columns
#         sentences.extend(matching_rows[col].dropna().tolist())
    
#     if not sentences:
#         logger.warning(f"No sentences found for word: {word}")
#         return None
    
#     return random.choice(sentences)

# def generate_image(prompt: str, filename: str):
#     """Generate image from text prompt and save to file"""
#     if not prompt or len(prompt) < 3:
#         raise ValueError("Prompt must be at least 3 characters long")
    
#     logger.info(f"Generating image for prompt: {prompt}")
    
#     # Generate image
#     image = pipe(prompt).images[0]
    
#     # Save with unique filename
#     output_path = IMAGE_OUTPUT_DIR / filename
#     image.save(output_path)
    
#     logger.info(f"Image saved to: {output_path}")
#     return output_path

import pandas as pd
import random
from diffusers import StableDiffusionPipeline
import torch
from pathlib import Path
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration - FLEXIBLE PATH HANDLING
def find_dataset():
    possible_paths = [
        Path.cwd() / "dataset.csv",            # Current directory
        Path(__file__).parent / "dataset.csv",  # Same as model.py
        Path.home() / "Desktop" / "dataset.csv",# Desktop
        Path.home() / "Downloads" / "dataset.csv" # Downloads
    ]
    
    for path in possible_paths:
        if path.exists():
            logger.info(f"Found dataset at: {path}")
            return path
            
    logger.error("Could not find dataset.csv in any common locations!")
    raise FileNotFoundError("Please place dataset.csv in the project directory")

# Set paths
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = find_dataset()
IMAGE_OUTPUT_DIR = BASE_DIR / "static" / "images"
IMAGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load dataset
def load_dataset():
    logger.info(f"Loading dataset from {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    logger.info(f"Dataset loaded with {len(df)} records")
    return df

try:
    df = load_dataset()
except Exception as e:
    logger.error(f"Error loading dataset: {str(e)}")
    df = pd.DataFrame(columns=["Letter", "Word"])  # Empty dataframe

# Load Stable Diffusion model
# def load_model():
#     logger.info("Loading Stable Diffusion model...")
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     logger.info(f"Using device: {device}")
    
#     dtype = torch.float16 if device == "cuda" else torch.float32
#     logger.info(f"Using dtype: {dtype}")
    
#     pipe = StableDiffusionPipeline.from_pretrained(
#         "runwayml/stable-diffusion-v1-5",
#         torch_dtype=dtype,
#         use_safetensors=True
#     ).to(device)
    
#     logger.info("Model loaded successfully")
#     return pipe

# try:
#     pipe = load_model()
# except Exception as e:
#     logger.error(f"Error loading model: {str(e)}")
#     pipe = None

def load_model():
    logger.info("Loading Stable Diffusion model...")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        dtype = torch.float16 if device == "cuda" else torch.float32
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=dtype,
            use_safetensors=True
        ).to(device)
        logger.info("Model loaded successfully!")
        return pipe
    except Exception as e:
        logger.error(f"MODEL LOAD FAILED: {str(e)}")  # Critical error
        return None

def get_letters():
    """Get sorted list of unique starting letters from dataset"""
    try:
        letters = df["Letter"].dropna().str.upper().unique()
        return sorted(letters.tolist())
    except:
        return ["A", "B", "C"]  # Fallback values

def get_words_by_letter(letter: str):
    """Get words starting with the specified letter"""
    try:
        if len(letter) != 1 or not letter.isalpha():
            raise ValueError("Input must be a single letter")
        
        letter = letter.upper()
        words = df[df["Letter"].str.upper() == letter]["Word"].dropna().unique()
        return sorted([word for word in words if isinstance(word, str)])
    except:
        return [f"{letter} Word 1", f"{letter} Word 2"]  # Fallback values

def get_random_sentence(word: str):
    """Get a random sentence containing the specified word"""
    try:
        if not word or not isinstance(word, str):
            return None
        
        word = word.lower()
        matching_rows = df[df["Word"].str.lower() == word]
        
        if matching_rows.empty:
            logger.warning(f"No matching rows found for word: {word}")
            return f"A sentence about {word}"
        
        # Get all non-empty sentences
        sentences = []
        for col in df.columns[2:]:  # Skip Letter and Word columns
            sentences.extend(matching_rows[col].dropna().tolist())
        
        if not sentences:
            logger.warning(f"No sentences found for word: {word}")
            return f"A descriptive sentence about {word}"
        
        return random.choice(sentences)
    except:
        return f"Example sentence containing {word}"

def generate_image(prompt: str, filename: str):
    """Generate image from text prompt and save to file"""
    if not prompt or len(prompt) < 3:
        raise ValueError("Prompt must be at least 3 characters long")
    
    if not pipe:
        raise RuntimeError("Model not loaded")
    
    logger.info(f"Generating image for prompt: {prompt}")
    
    # Generate image
    image = pipe(prompt).images[0]
    
    # Save with unique filename
    output_path = IMAGE_OUTPUT_DIR / filename
    image.save(output_path)
    
    logger.info(f"Image saved to: {output_path}")
    return output_path
