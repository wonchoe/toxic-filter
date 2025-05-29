from fastapi import FastAPI
from pydantic import BaseModel
import torch
import joblib
import numpy as np
import re
from spellchecker import SpellChecker  # 🔧

# Клас моделі
class ToxicClassifier(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ToxicClassifier, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(512, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

# Параметри
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS = ['toxic', 'severe_toxicity', 'obscene', 'insult', 'threat', 'identity_attack']

# Завантажити модель і TF-IDF
model = ToxicClassifier(input_dim=30000, output_dim=len(LABELS))
model.load_state_dict(torch.load("ultra_tox_model_final.pt", map_location=DEVICE))
model.to(DEVICE)
model.eval()

vectorizer = joblib.load("tfidf_vectorizer.joblib")
spell = SpellChecker()  # 🧠 Ініціалізація spell checker

def load_blacklist(path="custom_blacklist.txt"):
    try:
        with open(path, "r") as f:
            return set(line.strip().lower() for line in f if line.strip())
    except FileNotFoundError:
        return set()

CUSTOM_TOXIC_WORDS = load_blacklist()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'1', 'i', text)
    text = re.sub(r'3', 'e', text)
    text = re.sub(r'0', 'o', text)
    text = re.sub(r'5', 's', text)
    text = re.sub(r'@', 'a', text)
    text = re.sub(r'\$', 's', text)
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    corrected = [spell.correction(w) or w for w in words]
    return ' '.join(corrected)

# FastAPI
app = FastAPI()

class TextIn(BaseModel):
    text: str

def check_text(data: TextIn):
    try:
        cleaned = clean_text(data.text)
        vec = vectorizer.transform([cleaned]).toarray()
        x_tensor = torch.tensor(vec, dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            output = torch.sigmoid(model(x_tensor)).cpu().numpy()[0]
            result = {label: bool(score > 0.5) for label, score in zip(LABELS, output)}

            # -- BLACKLIST CHECK (word, phrase, substring, joined variant)
            tokens = cleaned.split()
            cleaned_joined = ''.join(tokens)
            lower_original = data.text.lower()
            lower_original_joined = ''.join(lower_original.split())

            # 1. Перевіряє кожне слово
            if any(word in CUSTOM_TOXIC_WORDS for word in tokens):
                result['toxic'] = True
                result['identity_attack'] = True
            # 2. Перевіряє всю фразу (почистили)
            if cleaned in CUSTOM_TOXIC_WORDS or cleaned_joined in CUSTOM_TOXIC_WORDS:
                result['toxic'] = True
                result['identity_attack'] = True
            # 3. Перевіряє всю фразу як сабстрінг (навіть у середині)
            if any(bad in cleaned for bad in CUSTOM_TOXIC_WORDS):
                result['toxic'] = True
                result['identity_attack'] = True
            # 4. Перевіряє raw-text теж (на випадок обфускацій)
            if any(bad in lower_original for bad in CUSTOM_TOXIC_WORDS):
                result['toxic'] = True
                result['identity_attack'] = True
            if any(bad in lower_original_joined for bad in CUSTOM_TOXIC_WORDS):
                result['toxic'] = True
                result['identity_attack'] = True

        return result
    except Exception as e:
        return {"error": str(e)}