from typing import TypedDict, Literal, Any


class StudyState(TypedDict, total=False):
    question: str
    intent: Literal["tutor", "quiz"]
    context: str
    sources: list[dict[str, Any]]
    answer: str
