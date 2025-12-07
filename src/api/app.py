from fastapi import FastAPI
from pydantic import BaseModel
import torch

from src.inference.generation import load_model_and_tokenizer, generate


app = FastAPI(title="Mini LLM API")

device = "cuda" if torch.cuda.is_available() else "cpu"
model, tokenizer = load_model_and_tokenizer(device=device)


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 50
    temperature: float = 1.0
    top_k: int | None = None


class GenerateResponse(BaseModel):
    output: str


@app.post("/generate", response_model=GenerateResponse)
def generate_endpoint(req: GenerateRequest):
    text = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=req.prompt,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
        device=device,
    )
    return GenerateResponse(output=text)
