from dataclasses import dataclass, fields


@dataclass
class Review:
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

    @classmethod
    def from_dict(cls, data: dict):
        # Collect field names from the dataclass
        field_names = {field.name for field in fields(cls)}
        # Filter the data dict to only include keys that match the class fields
        init_args = {key: data.get(key) for key in field_names if key in data}
        # Create an instance of Review with the filtered data
        return cls(**init_args)
