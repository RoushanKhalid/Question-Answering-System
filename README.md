# RNN-based Question Answering System

A minimal PyTorch + Flask web app for question answering using a simple RNN, trained on a small QA dataset.

## Features
- Answer questions using an RNN trained on your own dataset (CSV)
- Very simple, clean white-page frontend
- Works locally in your browser

## Project Structure
```
.
├── 100_Unique_QA_Dataset.csv       # QA dataset (customizable)
├── copy_of_pytorch_rnn_based_qa_system.py  # Core PyTorch model/code
├── app.py                         # Flask backend API and frontend
├── templates/
│   └── index.html                 # Minimal browser UI
├── static/                        # (optional) CSS/JS can go here
```

## Requirements
- Python 3.7+
- pip install torch pandas flask

## Setup
1. Clone this repo and place all files in a folder
2. (Ensure `100_Unique_QA_Dataset.csv` is present in the main directory)
3. Install dependencies:
    ```
    pip install torch pandas flask
    ```

## Usage
1. Run the Flask app:
    ```
    python app.py
    ```
2. Open your browser to [http://127.0.0.1:5000](http://127.0.0.1:5000)
3. Ask your question and get an answer instantly!

## Example
- **Q:** "What is the capital of France?"
- **A:** "Paris"

## Customizing
- Edit/add questions and answers in `100_Unique_QA_Dataset.csv` to grow your knowledge base!
- If you improve/train the model further, save and load weights as desired.

---

Made for demo and learning purposes. Contributions welcome!
