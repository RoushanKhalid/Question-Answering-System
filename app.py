import os
import pandas as pd
import numpy as np
import torch
from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer, util
import warnings
import re

# Suppress warnings for a clean console experience
warnings.filterwarnings('ignore')

app = Flask(__name__, template_folder="templates", static_folder="static")

class SemanticQASystem:
    """
    Optimized QA System using Semantic Embeddings.
    Designed for high accuracy even when questions are phrased differently.
    """
    
    def __init__(self, csv_path="100_Unique_QA_Dataset.csv"):
        self.csv_path = csv_path
        # Using a very high-quality embedding model (mpnet is more accurate than MiniLM)
        self.model_name = 'all-mpnet-base-v2'
        self.questions = []
        self.answers = []
        self.question_embeddings = None
        self.model = None
        
        self.initialize()

    def initialize(self):
        """Prepare the data and embedding model"""
        print(f"[*] Initializing Semantic Engine...")
        
        # Load Dataset
        if os.path.exists(self.csv_path):
            try:
                df = pd.read_csv(self.csv_path)
                # Ensure we have the right columns
                if 'question' in df.columns and 'answer' in df.columns:
                    self.questions = df['question'].tolist()
                    self.answers = df['answer'].tolist()
                    print(f"[+] Successfully indexed {len(self.questions)} knowledge points.")
                else:
                    print(f"[-] Error: CSV missing 'question' or 'answer' columns.")
                    self._load_fallback_data()
            except Exception as e:
                print(f"[-] Load Error: {e}")
                self._load_fallback_data()
        else:
            print(f"[!] {self.csv_path} not found. Loading demo data.")
            self._load_fallback_data()

        # Load Embedding Model
        try:
            print(f"[*] Loading Embedding Model: {self.model_name}...")
            # This handles downloading and caching automatically
            self.model = SentenceTransformer(self.model_name)
            
            print("[*] Generating Knowledge Embeddings...")
            # Convert-to-tensor enables GPU acceleration if available
            self.question_embeddings = self.model.encode(self.questions, convert_to_tensor=True)
            print("[✓] Semantic Engine Online.")
        except Exception as e:
            print(f"[-] Semantic Engine Failed to Start: {e}")
            self.model = None

    def _load_fallback_data(self):
        """Default dataset if CSV is missing or broken"""
        self.questions = ["What is the capital of France?", "How many continents are there?"]
        self.answers = ["The capital of France is Paris.", "There are 7 continents on Earth."]

    def get_response(self, user_query):
        """Semantic search with confidence scoring"""
        if not self.model or self.question_embeddings is None:
            return "Knowledge base is offline.", 0.0

        try:
            # 1. Clean query
            query = user_query.strip()
            
            # 2. Generate embedding for user query
            query_embedding = self.model.encode(query, convert_to_tensor=True)
            
            # 3. Compute semantic similarity (Cosine Similarity)
            similarities = util.cos_sim(query_embedding, self.question_embeddings)[0]
            
            # 4. Find the best match
            best_idx = torch.argmax(similarities).item()
            score = similarities[best_idx].item()
            
            return self.answers[best_idx], score
            
        except Exception as e:
            print(f"[-] Search Error: {e}")
            return "Internal processing error.", 0.0

# Initialize the system
qa_engine = SemanticQASystem()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    user_query = request.form.get("question", "").strip()
    
    if not user_query:
        return jsonify({"answer": "I'm listening. Please ask something.", "confidence": 0.0})

    answer, confidence = qa_engine.get_response(user_query)
    
    # Return JSON for the modern UI
    return jsonify({
        "answer": answer,
        "confidence": round(float(confidence), 3)
    })

if __name__ == "__main__":
    print("\n" + "🚀 " * 10)
    print("  Semantic QA System is active at http://127.0.0.1:5000")
    print("🚀 " * 10 + "\n")
    app.run(debug=False, port=5000, threaded=True)
