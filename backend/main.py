from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from model import get_letters, get_words_by_letter, get_random_sentence, generate_image

app = FastAPI()

# CORS: Allow frontend to access backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to frontend URL in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes

@app.get("/letters")
def get_all_letters():
    return {"letters": get_letters()}

@app.get("/words/{letter}")
def get_words(letter: str):
    return {"words": get_words_by_letter(letter.upper())}

@app.get("/sentence/{word}")
def get_sentence(word: str):
    sentence = get_random_sentence(word)
    if sentence:
        return {"sentence": sentence}
    return {"error": "No sentence found for this word."}

class PromptRequest(BaseModel):
    prompt: str

@app.post("/generate-image")
def generate(prompt_request: PromptRequest):
    image_path = generate_image(prompt_request.prompt)
    return FileResponse(image_path, media_type="image/png")