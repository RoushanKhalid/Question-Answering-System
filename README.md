# 🤖 Semantic Question Answering System

An intelligent QA system powered by **BERT (Sentence-Transformers)** for high-accuracy semantic matching. This system can understand the meaning behind your questions, even if you change the wording slightly.

## ✨ Features
- **Semantic Understanding**: Uses the `all-MiniLM-L6-v2` BERT model to find answers based on meaning, not just keywords.
- **Premium UI**: Modern, responsive, and animated user interface with confidence scores.
- **Optimized for Stability**: Runs on **Python 3.12** to ensure all ML libraries function perfectly without hanging.
- **Zero Git Metadata**: Completely cleared and ready for fresh deployment.

## 🚀 How to Run

### Method 1: One-Click (Recommended)
Simply double-click the `run_app.bat` file in the root directory.

### Method 2: Command Line
Open your terminal in this folder and run:
```bash
py -3.12 app.py
```
Then visit: `http://127.0.0.1:5000`

## 📦 Requirements
- Python 3.12 (Auto-detected if installed)
- Dependencies: `flask`, `torch`, `sentence-transformers`, `pandas`

---
*Note: The first run may take a moment to download the BERT model weight files.*
