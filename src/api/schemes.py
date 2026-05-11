from pydantic import BaseModel


class ReviewRequest(BaseModel):
    text: str


class ReviewResponse(BaseModel):
    sentiment: str
    urgency: str
    priority: str
    aspects: list
    explanation: dict