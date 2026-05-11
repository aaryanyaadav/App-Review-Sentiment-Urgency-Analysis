import os

# Reduce PyTorch CPU usage
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
torch.set_num_threads(1)

import pickle
from pathlib import Path
import nltk

# Use Render NLTK data path
nltk.data.path.append("/opt/render/nltk_data")

from src.preprocessing import TextPreprocessor
from src.priority import PriorityScorer
from src.aspect import AspectDetector
from src.explain import Explainer

# Import model
from src.hybrid_model.hybrid_model import HybridModel


class MainPipeline:
    def __init__(self, model_path="models/hybrid/"):

        print("Initializing MainPipeline...")

        model_dir = Path(model_path)

        # -----------------------------
        # Load tokenizer & vectorizer
        # -----------------------------
        print("Loading tokenizer...")

        with open(model_dir / "tokenizer.pkl", "rb") as f:
            self.tokenizer = pickle.load(f)

        print("Tokenizer loaded.")

        print("Loading TF-IDF vectorizer...")

        with open(model_dir / "tfidf_vectorizer.pkl", "rb") as f:
            self.tfidf = pickle.load(f)

        print("TF-IDF loaded.")

        print("Loading encoders...")

        with open(model_dir / "sent_encoder.pkl", "rb") as f:
            self.sent_encoder = pickle.load(f)

        with open(model_dir / "urg_encoder.pkl", "rb") as f:
            self.urg_encoder = pickle.load(f)

        print("Encoders loaded.")

        # -----------------------------
        # Rebuild model architecture
        # -----------------------------
        tfidf_dim = len(self.tfidf.get_feature_names_out())
        vocab_size = len(self.tokenizer.word_index) + 1

        print("Building model architecture...")

        self.model = HybridModel(
            embedding_matrix=None,
            tfidf_dim=tfidf_dim,
            hidden_dim=128,
            num_sent=len(self.sent_encoder.classes_),
            num_urg=len(self.urg_encoder.classes_)
        )

        # Embedding layer
        self.model.embedding = torch.nn.Embedding(vocab_size, 100)

        # -----------------------------
        # Load model weights
        # -----------------------------
        print("Loading model weights...")

        self.model.load_state_dict(
            torch.load(
                model_dir / "hybrid_model.pt",
                map_location="cpu"
            )
        )

        self.model.eval()

        print("Model loaded successfully!")

        # -----------------------------
        # Initialize modules
        # -----------------------------
        self.preprocessor = TextPreprocessor(self.tokenizer)
        self.priority = PriorityScorer()
        self.aspect = AspectDetector()
        self.explainer = Explainer(self.tokenizer)

        print("Pipeline initialized successfully!")

    def predict(self, text):

        # Preprocessing
        seq, clean_text = self.preprocessor.text_to_sequence(text)

        # TF-IDF vector
        tfidf_vec = self.tfidf.transform([clean_text]).toarray()

        # Convert to tensors
        seq_tensor = torch.tensor(seq, dtype=torch.long)
        tfidf_tensor = torch.tensor(tfidf_vec, dtype=torch.float32)

        # Model inference
        with torch.no_grad():
            sent_out, urg_out, attn_weights = self.model(
                seq_tensor,
                tfidf_tensor
            )

        # Decode predictions
        sentiment = self.sent_encoder.inverse_transform(
            [sent_out.argmax().item()]
        )[0]

        urgency = self.urg_encoder.inverse_transform(
            [urg_out.argmax().item()]
        )[0]

        return sentiment, urgency, attn_weights, clean_text, seq

    def run(self, text):

        sentiment, urgency, attn, clean_text, seq = self.predict(text)

        # Priority scoring
        priority = self.priority.compute(
            sentiment,
            urgency,
            clean_text
        )

        # Aspect detection
        aspects = self.aspect.detect(clean_text)

        # Explainability
        explanation = self.explainer.explain(
            clean_text,
            attn,
            seq
        )

        return {
            "input_text": text,
            "processed_text": clean_text,
            "sentiment": sentiment,
            "urgency": urgency,
            "priority": priority,
            "aspects": aspects,
            "explanation": explanation
        }


if __name__ == "__main__":

    pipeline = MainPipeline()

    while True:

        text = input("Enter Review: ")

        if text.lower() == "exit":
            print("Exiting...")
            break

        result = pipeline.run(text)

        print(f"Sentiment : {result['sentiment']}")
        print(f"Urgency   : {result['urgency']}")
        print(f"Priority  : {result['priority']}")
        print(f"Aspects   : {result['aspects']}")

        print("\nImportant Words:")

        for w in result["explanation"]["top_words"]:
            print(f"  {w['word']} ({w['score']:.4f})")