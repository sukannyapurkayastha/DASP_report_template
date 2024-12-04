from pydantic import BaseModel


class Review(BaseModel):
    reviewer: str
    rating: str | int
    soundness: str | int
    presentation: str | int
    contribution: str | int
    summary: str
    strengths: str
    weaknesses: str
    questions: str
    confidence: str
    forum: str | None = None
    id: str | None = None
    venue: str | None = None
    year: str | None = None
    type: str | None = None
    date: str | int | None = None
    flag_for_ethic_review: str | None = None
    content: str | None = None  # All the text in a single string
    sentences: list[str] | None = None  # Output of spacy sentencizer => This is used for the model classification
