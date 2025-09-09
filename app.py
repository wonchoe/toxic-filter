from fastapi import FastAPI
from pydantic import BaseModel
import torch
import joblib
import numpy as np
import re
from spellchecker import SpellChecker
from better_profanity import profanity

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


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS = ['toxic', 'severe_toxicity', 'obscene', 'insult', 'threat', 'identity_attack']


model = ToxicClassifier(input_dim=30000, output_dim=len(LABELS))
model.load_state_dict(torch.load("ultra_tox_model_final.pt", map_location=DEVICE))
model.to(DEVICE)
model.eval()

vectorizer = joblib.load("tfidf_vectorizer.joblib")
spell = SpellChecker()

def load_blacklist(path="custom_blacklist.txt"):
    try:
        with open(path, "r") as f:
            return set(line.strip().lower() for line in f if line.strip())
    except FileNotFoundError:
        return set()

CUSTOM_TOXIC_WORDS = load_blacklist()

def clean_text(text):
    text = text.lower()
    replacements = {
        '1': 'i', '3': 'e', '0': 'o', '5': 's', '@': 'a', '$': 's', '!': 'i',
        '|': 'i', '7': 't', '4': 'a', '8': 'b'
    }
    for src, tgt in replacements.items():
        text = text.replace(src, tgt)
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    #corrected = [spell.correction(w) or w for w in words]
    return ' '.join(words)

# FastAPI
app = FastAPI()

class TextIn(BaseModel):
    text: str

profanity.load_censor_words()

def check_text(data: TextIn):
    try:
        cleaned = clean_text(data.text)
        if profanity.contains_profanity(data.text) or profanity.contains_profanity(cleaned):
            print(f"[PROFANITY FILTER] BLOCKED: '{data.text}' (cleaned: '{cleaned}')")
            return {
                'toxic': True,
                'severe_toxicity': False,
                'obscene': True,
                'insult': True,
                'threat': False,
                'identity_attack': True,
                'detected_by': 'profanity'
            }

        print("[DEBUG] CLEANED:", cleaned)
        vec = vectorizer.transform([cleaned]).toarray()
        print("[DEBUG] VEC:", vec)
        x_tensor = torch.tensor(vec, dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            output = torch.sigmoid(model(x_tensor)).cpu().numpy()[0]
            print("[DEBUG] AI OUTPUT:", output)
            result = {label: bool(score > 0.5) for label, score in zip(LABELS, output)}
            print("[DEBUG] RESULT BEFORE BLACKLIST:", result)

            # -- BLACKLIST CHECK (word, phrase, substring, joined variant)
            tokens = cleaned.split()
            cleaned_joined = ''.join(tokens)
            lower_original = data.text.lower()
            lower_original_joined = ''.join(lower_original.split())

            if any(word in CUSTOM_TOXIC_WORDS for word in tokens):
                print(f"[BLACKLIST] TOKEN MATCH: {tokens}")
                result['toxic'] = True
                result['identity_attack'] = True
                result['detected_by'] = 'blacklist_word'
            if cleaned in CUSTOM_TOXIC_WORDS or cleaned_joined in CUSTOM_TOXIC_WORDS:
                print(f"[BLACKLIST] PHRASE MATCH: {cleaned}, {cleaned_joined}")
                result['toxic'] = True
                result['identity_attack'] = True
                result['detected_by'] = 'blacklist_phrase'
            if any(bad in cleaned for bad in CUSTOM_TOXIC_WORDS):
                print(f"[BLACKLIST] SUBSTRING CLEANED: {cleaned}")
                result['toxic'] = True
                result['identity_attack'] = True
                result['detected_by'] = 'blacklist_substr_cleaned'
            if any(bad in lower_original for bad in CUSTOM_TOXIC_WORDS):
                print(f"[BLACKLIST] SUBSTRING RAW: {lower_original}")
                result['toxic'] = True
                result['identity_attack'] = True
                result['detected_by'] = 'blacklist_substr_raw'
            if any(bad in lower_original_joined for bad in CUSTOM_TOXIC_WORDS):
                print(f"[BLACKLIST] SUBSTRING JOINED RAW: {lower_original_joined}")
                result['toxic'] = True
                result['identity_attack'] = True
                result['detected_by'] = 'blacklist_substr_joined_raw'

        return result
    except Exception as e:
        return {"error": str(e)}

    
@app.post("/check")
def check_endpoint(data: TextIn):
    return check_text(data)
