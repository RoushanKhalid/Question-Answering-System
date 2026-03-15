# 🤖 Semantic Question Answering System

<div align="center">

**An intelligent QA system that understands *meaning*, not just keywords.**

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.x-000000?style=flat-square&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Sentence Transformers](https://img.shields.io/badge/Sentence%20Transformers-MPNet-FF6B6B?style=flat-square)](https://www.sbert.net/)

*Semantic search over your knowledge base — ask in your own words and get accurate answers.*

</div>

---

## 📖 Overview

This project is a **semantic question-answering system** that uses state-of-the-art sentence embeddings to match user questions to a curated knowledge base. Instead of brittle keyword matching, it uses the **all-mpnet-base-v2** model (via [Sentence-Transformers](https://www.sbert.net/)) to encode both questions and answers into dense vectors, then finds the best match using **cosine similarity**. That means you can rephrase questions, use synonyms, or ask in a different style and still get the right answer.

| Aspect | Details |
|--------|--------|
| **Model** | `sentence-transformers/all-mpnet-base-v2` (MPNet, 768-dim) |
| **Backend** | Flask (Python) |
| **Data** | CSV with `question` and `answer` columns |
| **UI** | Single-page web app with confidence scores and modern dark theme |

---

## ✨ Features

- **Semantic understanding** — Matches by meaning, so "What's France's capital?" and "Capital of France?" both map to the same answer.
- **Confidence scoring** — Every response comes with a 0–100% match score so you know how reliable the answer is.
- **Rich web UI** — Dark theme, gradients, smooth animations, and a clean "Ask AI" experience.
- **Easy to extend** — Drop in a new CSV; the system indexes it at startup. No retraining required.
- **GPU-ready** — Uses PyTorch; if CUDA is available, encoding runs on GPU for faster responses.
- **Fallback data** — If the CSV is missing or invalid, a small built-in demo dataset keeps the app runnable.

---

## 🏗️ How It Works

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│  Your question  │ ──► │  MPNet encoder   │ ──► │  Cosine similarity   │
│  "Capital of    │     │  (sentence       │     │  vs. all stored      │
│   France?"      │     │   transformers)  │     │  question vectors    │
└─────────────────┘     └──────────────────┘     └──────────┬──────────┘
                                                             │
                                                             ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│  Answer +       │ ◄── │  Return answer   │ ◄── │  Best-matching      │
│  confidence %    │     │  for best match  │     │  question index     │
└─────────────────┘     └──────────────────┘     └─────────────────────┘
```

1. **Load** — On startup, the app reads the CSV and encodes every question into a vector.
2. **Query** — Your question is encoded with the same model.
3. **Match** — Cosine similarity finds the stored question most similar to your query.
4. **Respond** — The corresponding answer and similarity score are returned to the UI.

---

## 📦 Prerequisites

- **Python 3.11+** (3.11 or 3.12 recommended; the launcher will use what you have)
- **pip** (usually bundled with Python)
- **~500 MB disk** for the first run (model weights are downloaded once and cached)

---

## 🚀 Installation & Run

### 1. Clone or download the project

```bash
git clone https://github.com/your-username/Question-Answering-System.git
cd Question-Answering-System
```

### 2. (Optional) Create a virtual environment

```bash
py -3.11 -m venv venv
.\venv\Scripts\activate   # Windows
# source venv/bin/activate  # macOS/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

| Package | Purpose |
|---------|--------|
| `flask` | Web server and API |
| `pandas` | Load and handle CSV data |
| `torch` | Tensor ops and optional GPU |
| `sentence-transformers` | MPNet model and encoding utilities |

### 4. Run the app

**Option A — One-click (Windows)**  
Double-click **`run_app.bat`** in the project root.

**Option B — Command line**

```bash
py -3.11 app.py
# or
python app.py
```

Then open in your browser: **http://127.0.0.1:5000**

> ⏳ **First run:** The app will download the `all-mpnet-base-v2` model from Hugging Face (one-time). This can take 1–2 minutes depending on your connection.

---

## 📁 Project Structure

```
Question-Answering-System/
├── app.py                    # Flask app + SemanticQASystem logic
├── run_app.bat               # Windows one-click launcher
├── requirements.txt          # Python dependencies
├── 100_Unique_QA_Dataset.csv  # Default Q&A knowledge base (90 entries)
├── templates/
│   └── index.html            # Single-page UI (form + results + confidence)
├── static/                   # (optional) CSS/JS assets
└── README.md                 # This file
```

---

## 📄 Dataset Format

The system expects a CSV with exactly two columns:

| Column    | Description |
|-----------|-------------|
| `question` | The question or prompt (e.g. "What is the capital of France?") |
| `answer`   | The answer to return (e.g. "Paris") |

Example:

```csv
question,answer
What is the capital of France?,Paris
Who wrote 'To Kill a Mockingbird'?,Harper Lee
What is the largest planet in our solar system?,Jupiter
```

To use your own data, either replace `100_Unique_QA_Dataset.csv` or pass a custom path when instantiating `SemanticQASystem(csv_path="your_file.csv")` in `app.py`.

---

## 🔧 Configuration

| What | Where | Notes |
|------|--------|------|
| CSV path | `app.py` → `SemanticQASystem(csv_path=...)` | Default: `100_Unique_QA_Dataset.csv` |
| Embedding model | `app.py` → `self.model_name` | Default: `all-mpnet-base-v2`; other [sentence-transformers](https://www.sbert.net/docs/pretrained_models.html) models work too |
| Port | `app.py` → `app.run(port=5000)` | Change if 5000 is in use |
| Debug mode | `app.py` → `app.run(debug=...)` | Set `True` for auto-reload during development |

---

## 🌐 API

The app exposes a simple POST endpoint for programmatic use.

**`POST /predict`**

- **Content-Type:** `application/x-www-form-urlencoded`
- **Body:** `question=<your question>`
- **Response (JSON):**

```json
{
  "answer": "Paris",
  "confidence": 0.892
}
```

Example with `curl`:

```bash
curl -X POST http://127.0.0.1:5000/predict -d "question=What is the capital of France?"
```

---

## ❓ Troubleshooting

| Issue | What to do |
|-------|------------|
| **"No suitable Python runtime found"** | Install Python 3.11+ or edit `run_app.bat` to use `py` or `python` instead of `py -3.11`. |
| **`ModuleNotFoundError: No module named 'flask'`** | Run `pip install -r requirements.txt` in the project folder. |
| **First run is slow** | Normal. The MPNet model is downloaded once; later starts are fast. |
| **Low confidence on all answers** | Your question may be far from any in the CSV; try rephrasing or adding more Q&A pairs. |
| **Port 5000 already in use** | Change `port=5000` in `app.run(...)` in `app.py` to another port (e.g. 5001). |

---

## 📜 License & Credits

- **Sentence-Transformers** — [sentence-transformers](https://www.sbert.net/) (MPNet and utilities)
- **Model** — [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) on Hugging Face
- **Flask** — [pallets/flask](https://github.com/pallets/flask)

This project is provided as-is for learning and deployment. Adjust and extend as you like.

---

<div align="center">

**Built with Python · Flask · PyTorch · Sentence-Transformers**

*Ask in your own words — get the right answer.*

</div>
