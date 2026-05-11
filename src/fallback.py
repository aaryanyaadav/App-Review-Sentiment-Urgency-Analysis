import pickle
from pathlib import Path
from typing import Any

from src.priority import PriorityScorer
from src.aspect import AspectDetector


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class BaselineFallback:
    def __init__(self, model_dir: str = "models/baseline_sen_+_urg") -> None:
        self.model_dir = Path(model_dir)
        if not self.model_dir.is_absolute():
            self.model_dir = PROJECT_ROOT / self.model_dir
        self.loaded = False

    def load(self) -> None:
        if self.loaded:
            return
        try:
            with open(self.model_dir / "tfidf_vectorizer.pkl", "rb") as f:
                self.tfidf = pickle.load(f)

            with open(self.model_dir / "sentiment_model.pkl", "rb") as f:
                self.sent_model = pickle.load(f)

            with open(self.model_dir / "urgency_model.pkl", "rb") as f:
                self.urg_model = pickle.load(f)

            # Optional encoders
            sent_enc = self.model_dir / "sentiment_encoder.pkl"
            urg_enc = self.model_dir / "urgency_encoder.pkl"
            self.sent_encoder = pickle.load(open(sent_enc, "rb")) if sent_enc.exists() else None
            self.urg_encoder = pickle.load(open(urg_enc, "rb")) if urg_enc.exists() else None

            self.priority = PriorityScorer()
            self.aspect = AspectDetector()

            self.loaded = True
        except Exception:
            self.loaded = False
            raise

    def predict(self, text: str) -> dict[str, Any]:
        if not self.loaded:
            self.load()

        clean = text.strip().lower()
        tfidf_vec = self.tfidf.transform([clean]).toarray()

        sent_pred = self.sent_model.predict(tfidf_vec)[0]
        urg_pred = self.urg_model.predict(tfidf_vec)[0]

        if self.sent_encoder is not None:
            sentiment = self.sent_encoder.inverse_transform([sent_pred])[0]
        else:
            sentiment = str(sent_pred)

        if self.urg_encoder is not None:
            urgency = self.urg_encoder.inverse_transform([urg_pred])[0]
        else:
            urgency = str(urg_pred)

        priority = self.priority.compute(sentiment, urgency, clean)
        aspects = self.aspect.detect(clean)

        explanation = {"method": "baseline", "top_words": []}

        return {
            "sentiment": sentiment,
            "urgency": urgency,
            "priority": priority,
            "aspects": aspects,
            "explanation": explanation,
            "used_fallback": "baseline"
        }
