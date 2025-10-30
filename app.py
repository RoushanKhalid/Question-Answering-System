from flask import Flask, render_template, request, jsonify
import torch
import pandas as pd
import os
from copy_of_pytorch_rnn_based_qa_system import tokenize, text_to_indices, SimpleRNN, vocab

# Use absolute path for CSV
CSV_PATH = os.path.abspath("100_Unique_QA_Dataset.csv")
try:
    df = pd.read_csv(CSV_PATH)
except Exception as e:
    print(f"Error loading the dataset at {CSV_PATH}: {e}")
    df = None

# Build a simple lookup for direct question->answer, for deterministic answers
qa_lookup = {row['question'].strip().lower(): row['answer'] for i, row in (df.iterrows() if df is not None else [])}

# Load model as in .py file
model = SimpleRNN(len(vocab))
# You may want to load trained weights here if saving them later
model.eval()

app = Flask(__name__, template_folder="templates", static_folder="static")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if df is None:
        return jsonify({"answer": "Dataset not loaded."})
    question = request.form.get("question", "")
    qkey = question.strip().lower()
    # Direct lookup first (answers exactly from CSV)
    if qkey in qa_lookup:
        return jsonify({"answer": qa_lookup[qkey].replace('-', ' ')})
    # Otherwise, run through the model for most similar answer
    numerical_question = text_to_indices(question, vocab)
    question_tensor = torch.tensor(numerical_question).unsqueeze(0)
    output = model(question_tensor)
    probs = torch.nn.functional.softmax(output, dim=1)
    value, predicted_index = torch.max(probs, dim=1)
    # Map back to an answer string if possible (find all answers matching this vocab index)
    pred_token = list(vocab.keys())[predicted_index.item()]
    possible_answers = [row['answer'] for _, row in df.iterrows() if pred_token in tokenize(row['answer'])]
    if possible_answers:
        return jsonify({"answer": possible_answers[0].replace('-', ' ')})
    else:
        return jsonify({"answer": "Sorry, answer not found."})

if __name__ == "__main__":
    app.run(debug=True)
