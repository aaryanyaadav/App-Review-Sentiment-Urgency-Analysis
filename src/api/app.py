from fastapi import FastAPI, HTTPException
from pathlib import Path
import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import MainPipeline
from src.api.schemes import ReviewRequest
from src.fallback import BaselineFallback

port = int(os.environ.get("PORT", 10000))

app = FastAPI(
    title="Application Review Sentiment Analyzer",
    description="Hybrid Model For Prediction",
    version="1.0"
)

# Load pipeline once
pipeline = MainPipeline()
_baseline = BaselineFallback()


# Health Check
@app.get("/")
def home():
    return {"message": "API is running prefectly"}


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "Application Review Sentiment Analyzer",
        "message": "API is working properly"
    }

#prediction endpoint 

@app.post("/analyze")
def analyze_review(request: ReviewRequest):

    global pipeline, _baseline

    try:
        # Lazy load models only when needed
        if pipeline is None:
            pipeline = MainPipeline()

        if _baseline is None:
            _baseline = BaselineFallback()

        # Input validation
        if not request.text or len(request.text.strip()) == 0:
            raise HTTPException(status_code=400, detail="Empty input text")

        # Run pipeline
        result = pipeline.run(request.text)

        return {
            "success": True,
            "data": {
                "sentiment": result["sentiment"],
                "urgency": result["urgency"],
                "priority": result["priority"],
                "aspects": result["aspects"],
                "explanation": result["explanation"]
            }
        }

    except Exception as e:
        # Try baseline fallback
        try:
            if _baseline is None:
                _baseline = BaselineFallback()

            fb = _baseline.predict(request.text)

            return {
                "success": True,
                "used_fallback": "baseline",
                "data": fb
            }

        except Exception:
            return {
                "success": False,
                "error": str(e),
                "fallback": {
                    "sentiment": "Unknown",
                    "urgency": "Low",
                    "priority": "Low",
                    "aspects": ["General Issue"],
                    "explanation": {
                        "top_words": [],
                        "method": "fallback"
                    }
                }
            }


@app.post("/analyze/fallback")
def analyze_fallback(request: ReviewRequest):
    try:
        global _baseline

        if _baseline is None:
            _baseline = BaselineFallback()

        fb = _baseline.predict(request.text)
        return {"success": True, "used_fallback": "baseline", "data": fb}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fallback failed: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=port)



