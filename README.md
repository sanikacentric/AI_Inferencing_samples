# Inference Options README (Groq API, Hugging Face Hosted, ONNX Runtime, vLLM)

This repository (or notes file) shows **multiple ways to run LLM inference**, depending on your constraints:

- **Hosted inference (no GPU needed)** → Hugging Face Inference API
- **High-speed hosted inference** → Groq (conceptual example)
- **Fast local CPU/GPU inference** → ONNX Runtime (e.g., Phi-3.5 mini)
- **Production-grade local inference server** → vLLM (OpenAI-compatible API)

> ✅ Pick the approach based on: **cost, speed, privacy, and deployment setup**.

---

## Table of Contents
- [1. Groq API Inference (Conceptual)](#1-groq-api-inference-conceptual)
- [2. Hugging Face Hosted Inference API (No GPU Needed)](#2-hugging-face-hosted-inference-api-no-gpu-needed)
- [3. ONNX Runtime (Fast Local Inference)](#3-onnx-runtime-fast-local-inference)
- [4. vLLM (Local OpenAI-Compatible Inference Server)](#4-vllm-local-openai-compatible-inference-server)
- [Recommended Usage](#recommended-usage)
- [Environment Variables](#environment-variables)
- [Troubleshooting](#troubleshooting)
- [Security Notes](#security-notes)

---

## 1. Groq API Inference (Conceptual)

Groq is known for **very fast inference** on supported models.  
The below script is **conceptual** and may not run as-is without:
- valid Groq API access
- correct endpoint + model name for your account

### File: `groq_inference_app.py` (Conceptual)

```python
import requests

def run_groq_inference(prompt):
    url = "https://api.groq.com/inference"
    headers = {
        "Authorization": "Bearer YOUR_GROQ_API_KEY",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "groq-llm",
        "prompt": prompt,
        "max_tokens": 50
    }
    response = requests.post(url, json=payload, headers=headers)
    return response.json()["generated_text"]

if __name__ == "__main__":
    prompt = "AI will revolutionize"
    output = run_groq_inference(prompt)
    print("Groq Inference Output:\n", output)
Notes

Replace YOUR_GROQ_API_KEY with your actual key.

Replace url and model with the ones Groq documents for your plan/account.

Add error handling in production (timeouts, retries, response validation).

2. Hugging Face Hosted Inference API (No GPU Needed)

✅ No GPU needed — Hugging Face runs inference for you.
Best for:

demos

small projects

quick testing

when you don’t want infra

Example: GPT-2 Inference via Hugging Face API
import requests

API_URL = "https://api-inference.huggingface.co/models/gpt2"
headers = {"Authorization": "Bearer YOUR_HF_TOKEN"}

def query(prompt):
    response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
    return response.json()[0]["generated_text"]

print(query("AI is revolutionizing the world because"))
Requirements

A Hugging Face token with access to inference

Internet access

3. ONNX Runtime (Fast Local Inference)

ONNX Runtime is great if you want:

local control

fast inference (often very optimized)

CPU-friendly execution (or GPU if configured)

to deploy without heavy PyTorch runtime in some cases

If you want ONNX + local control, Phi-3.5-mini-instruct or Phi-3-mini are strong options (ONNX-ready and openly accessible via ONNX community exports).

Install Dependencies
pip install onnxruntime transformers numpy
Example: Run an ONNX Model Locally
from transformers import AutoTokenizer
from onnxruntime import InferenceSession
import numpy as np

# Load ONNX session
session = InferenceSession("phi3.5-mini-instruct-128k.onnx")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("onnx-community/Phi-3.5-mini-instruct-onnx-web")

# Tokenize input into NumPy tensors
inputs = tokenizer("Explain ONNX.", return_tensors="np")

# Run inference (input name depends on the ONNX model graph)
outputs = session.run(None, {session.get_inputs()[0].name: inputs["input_ids"]})

print("ONNX output:", outputs[0])
Notes

The ONNX file must exist locally: phi3.5-mini-instruct-128k.onnx

Input/output names vary across ONNX exports

For full chat-style generation, you usually need:

attention masks

past_key_values caching

a generation loop
(the above is a minimal “raw forward pass” example)

4. vLLM (Local OpenAI-Compatible Inference Server)

vLLM is great for:

running models you own (local or from HF)

serving multiple requests efficiently

OpenAI-compatible API endpoint (easy integration)

Step 1: Launch the vLLM server
python -m vllm.entrypoints.openai.api_server \
  --model mistralai/Mistral-7B-Instruct-v0.2

This starts a server at:

http://localhost:8000/v1

Step 2: Call it like OpenAI (OpenAI-Compatible)

Below is the same idea as OpenAI Chat Completions, but pointed at your local server.

import openai

openai.api_base = "http://localhost:8000/v1"
openai.api_key = "dummy"

response = openai.ChatCompletion.create(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    messages=[
        {"role": "user", "content": "Explain ONNX in simple terms."}
    ]
)

print(response.choices[0].message["content"])
Notes

api_key is not required locally (set a dummy value to satisfy the client).

Ensure you have enough RAM/VRAM depending on model size.

Recommended Usage

Use this quick decision guide:

✅ Hugging Face Inference API

Use when:

you want zero infra

you want quick demo/testing

you don’t have GPU

✅ Groq (Hosted, Very Fast)

Use when:

you want speed and hosted inference

you have Groq access and supported models

✅ ONNX Runtime

Use when:

you want fast local inference

you want local control + optimized runtime

you are okay handling generation loop complexity for chat-style models

✅ vLLM

Use when:

you want a local inference server

you want OpenAI-compatible endpoint

you want scalable multi-request serving

Environment Variables

Create a .env file (recommended):

HF_TOKEN=your_huggingface_token_here
GROQ_API_KEY=your_groq_api_key_here

And in Python, load them (optional):

import os
HF_TOKEN = os.getenv("HF_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
Troubleshooting
Hugging Face errors

401 Unauthorized → token missing/invalid

403 Forbidden → gated model or no access

Slow response → model cold start or rate limits

ONNX issues

Invalid Graph / shape mismatch → wrong model export or missing inputs

Output not “text” → you’re doing raw logits inference, not generation
(need decoding loop)

vLLM issues

Out of memory → use smaller model or quantization

Model download issues → confirm model name and HF auth (if gated)

Security Notes

Never hardcode API keys in code committed to Git.

Use environment variables or secret managers.

Rotate tokens if accidentally exposed.

License

Use freely for learning and internal demos. Add a license if publishing publicly.
