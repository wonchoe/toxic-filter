
# 🧪 Toxic Filter API

**AI-powered REST API for real-time toxic content detection in chat messages, usernames, and more.**

- 🚀 Built with FastAPI, PyTorch, scikit-learn, better-profanity, and spellchecker
- 🔥 Detects insults, profanity, threats, hate speech, and obfuscated offensive words
- 🛡️ Customizable blacklist for multi-language & slang coverage
- 🏃‍♂️ Super-fast (Docker-ready), perfect for chat, moderation, or integration with any project

---

## Features

- **Hybrid Moderation**: AI model + TF-IDF vectorizer for advanced toxic phrase detection
- **Profanity Filter**: Uses [`better-profanity`](https://github.com/snguyenthanh/better_profanity) for instant detection (including obfuscated words: "sh1t", "f*ck", etc.)
- **Custom Blacklist**: Add any extra words/phrases (e.g., slang, other languages) via `custom_blacklist.txt`
- **Spellchecker Support**: Handles common typos and character swaps
- **Multi-label Output**: Toxic, obscene, insult, identity attack, threat, severe toxicity
- **API-Ready**: Easy HTTP/JSON interface for your backend/frontend
- **Docker Support**: One-command launch for production or local dev

---

## Quick Start

### 1. Clone & Prepare

```bash
git clone https://github.com/wonchoe/toxic-filter.git
cd toxic-filter-api
```

### 2. Install Dependencies

**Python 3.10 recommended**

```bash
pip install -r requirements.txt
```

### 3. 3. Model Files — Already Included
The required files ultra_tox_model_final.pt (PyTorch model) and tfidf_vectorizer.joblib (TF-IDF vectorizer) are already included in this repository.

These models were pre-trained on a massive dataset of 7 million chat messages/comments, and internet “meme/trash talk” content for real-world performance.

(Optional): Update your blacklist for extra words or memes: custom_blacklist.txt

### 4. Run Locally

```bash
uvicorn app:app --host 0.0.0.0 --port 8002
```

or with Docker:

```bash
docker build -t toxic .
docker run -p 8002:8002 toxic
```

---

## API Usage

### **POST /check**

Checks a message for toxic or inappropriate content.

**Request:**
```json
{
  "text": "your message here"
}
```

**Response:**
```json
{
  "toxic": true,
  "severe_toxicity": false,
  "obscene": true,
  "insult": false,
  "threat": false,
  "identity_attack": false
}
```

**If the message is blocked by the profanity filter:**
```json
{
  "blocked_by": "profanity",
  "reason": "Profanity detected"
}
```

---

## Configuration

- **custom_blacklist.txt**:  
  One word/phrase per line, lowercase. Used for extra language and slang.

- **Profanity Filter**:  
  Uses English by default. To add words:
  ```python
  from better_profanity import profanity
  profanity.load_censor_words_from_file("my_extra_profanity.txt")
  ```

- **Spellchecker**:  
  Auto-fixes common typos and obfuscated words (e.g., "n1gga" → "nigga").

---

## Example CURL Requests

```bash
curl -X POST http://localhost:8002/check -H "Content-Type: application/json" -d '{"text":"n1gga"}'
curl -X POST http://localhost:8002/check -H "Content-Type: application/json" -d '{"text":"kill yourself"}'
curl -X POST http://localhost:8002/check -H "Content-Type: application/json" -d '{"text":"hello world"}'
```

---

## Why Use Both AI & Blacklist?

- **AI model** detects *context*, *phrases*, *new toxic memes*, and *unusual combinations*.
- **Blacklist/Profanity** instantly catches **obfuscated**, **slang** abuse (including new/rare cases).

**No more "f.u.c.k", "n1gga" slipping through!**

---

## Roadmap

- [x] API endpoint
- [x] Docker image
- [x] Multi-language support via custom blacklist
- [x] Profanity/obfuscation detection

---

## License

MIT

---

## Credits

- [PyTorch](https://pytorch.org/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [better-profanity](https://github.com/snguyenthanh/better_profanity)
- [scikit-learn](https://scikit-learn.org/)
- [pyspellchecker](https://github.com/barrust/pyspellchecker)

---

**Questions or contributions? PRs and issues are welcome!**
