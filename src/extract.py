import pdfplumber
import torch
import pickle
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = '../models/bert-clauses-weighted'

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=8,
    ignore_mismatched_sizes=True
)

# Load the fine-tuned weights
import torch
import os

# Find the checkpoint folder
checkpoints = [d for d in os.listdir(MODEL_PATH) if d.startswith('checkpoint')]
if checkpoints:
    latest = sorted(checkpoints)[-1]
    weights_path = os.path.join(MODEL_PATH, latest, 'model.safetensors')
else:
    weights_path = os.path.join(MODEL_PATH, 'model.safetensors')

from safetensors.torch import load_file
state_dict = load_file(weights_path)
model.load_state_dict(state_dict, strict=False)
print(f"Loaded weights from {weights_path}")

with open('../models/label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

with open('../models/class_thresholds.pkl', 'rb') as f:
    thresholds = pickle.load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if len(chunk.strip()) > 50:
            chunks.append(chunk)
    return chunks

def classify_chunks(chunks: list) -> list:
    results = []
    for chunk in chunks:
        inputs = tokenizer(
            chunk,
            truncation=True,
            max_length=256,
            return_tensors='pt',
            padding=True
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]

        best_class_idx = np.argmax(probs)
        best_class = le.classes_[best_class_idx]
        confidence = float(probs[best_class_idx])
        thresh = thresholds.get(best_class, 0.3)

        if confidence >= thresh:
            results.append({
                'clause_type': best_class,
                'text': chunk,
                'confidence': round(confidence, 4)
            })

    return results

def deduplicate_clauses(clauses: list) -> list:
    seen = {}
    for clause in clauses:
        ct = clause['clause_type']
        if ct not in seen or clause['confidence'] > seen[ct]['confidence']:
            seen[ct] = clause
    return list(seen.values())

def extract_clauses(pdf_path: str) -> list:
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)
    clauses = classify_chunks(chunks)
    return deduplicate_clauses(clauses)