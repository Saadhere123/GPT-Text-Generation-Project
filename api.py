from fastapi import FastAPI
from transformers import pipeline

app = FastAPI()

generator = pipeline(
    "text-generation",
    model="gpt2"
)

@app.get("/generate")
def generate(prompt: str):

    output = generator(
        prompt,
        max_length=100
    )

    return {
        "generated_text": output[0]["generated_text"]
    }
