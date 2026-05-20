import os

# Reduce PyTorch CPU usage
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
torch.set_num_threads(1)

import pickle
from pathlib import Path
import nltk

# Render NLTK path
if os.path.exists("/opt/render/nltk_data"):
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

        # Load Word Index
        print("Loading word index...")

        with open(model_dir / "word_index.pkl", "rb") as f:
            self.word_index = pickle.load(f)

        print("Word index loaded.")

        # Load TF-IDF
        print("Loading TF-IDF vectorizer...")

        with open(model_dir / "tfidf_vectorizer.pkl", "rb") as f:
            self.tfidf = pickle.load(f)

        print("TF-IDF loaded.")

        # Load Encoders
        print("Loading encoders...")

        with open(model_dir / "sent_encoder.pkl", "rb") as f:
            self.sent_encoder = pickle.load(f)

        with open(model_dir / "urg_encoder.pkl", "rb") as f:
            self.urg_encoder = pickle.load(f)

        print("Encoders loaded.")

        # Model Config
        tfidf_dim = len(
            self.tfidf.get_feature_names_out()
        )

        vocab_size = len(self.word_index) + 2

        print("Building model architecture...")

        # Create Model
        self.model = HybridModel(
            embedding_matrix=None,
            tfidf_dim=tfidf_dim,
            hidden_dim=128,
            num_sent=len(self.sent_encoder.classes_),
            num_urg=len(self.urg_encoder.classes_),
            vocab_size=vocab_size
        )

        # Load Weights
        print("Loading model weights...")

        self.model.load_state_dict(
            torch.load(
                model_dir / "hybrid_model.pt",
                map_location="cpu"
            )
        )

        self.model.eval()

        print("Model loaded successfully!")

        # Modules
        self.preprocessor = TextPreprocessor(
            self.word_index
        )

        self.priority = PriorityScorer()

        self.aspect = AspectDetector()

        self.explainer = Explainer(
            self.word_index
        )

        print("Pipeline initialized successfully!")

    # Predict
    def predict(self, text):

        # Preprocessing
        seq, clean_text = self.preprocessor.text_to_sequence(text)

        # TF-IDF
        tfidf_vec = self.tfidf.transform(
            [clean_text]
        ).toarray()

        # Tensor Conversion
        seq_tensor = torch.tensor(
            seq,
            dtype=torch.long
        )

        tfidf_tensor = torch.tensor(
            tfidf_vec,
            dtype=torch.float32
        )

        # Inference
        with torch.no_grad():

            sent_out, urg_out, attn_weights = self.model(
                seq_tensor,
                tfidf_tensor
            )

        # Decode
        sentiment = self.sent_encoder.inverse_transform(
            [sent_out.argmax().item()]
        )[0]

        urgency = self.urg_encoder.inverse_transform(
            [urg_out.argmax().item()]
        )[0]

        return (
            sentiment,
            urgency,
            attn_weights,
            clean_text,
            seq
        )


    # Full Pipeline
    def run(self, text):

        (
            sentiment,
            urgency,
            attn,
            clean_text,
            seq

        ) = self.predict(text)

        # Priority
        priority = self.priority.compute(
            sentiment,
            urgency,
            clean_text
        )

        # Aspect Detection
        aspects = self.aspect.detect(
            clean_text
        )

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


# Manual Testing
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

            print(
                f"  {w['word']} "
                f"({w['score']:.4f})"
            )