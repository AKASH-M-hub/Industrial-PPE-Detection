import re
from typing import Optional
from pydantic import BaseModel, Field
from loguru import logger
from src.config import settings


class IntentDecision(BaseModel):
    """Result of Intent Routing step."""
    needs_detection: bool = Field(
        ...,
        description="True if answering requires inspecting objects in the image"
    )
    intent_category: str = Field(
        ...,
        description="'visual_inspection', 'general_knowledge', 'irrelevant', or 'chit_chat'"
    )
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = Field(..., description="Explanation of why this routing path was selected")


class IntentRouter:
    """Routes user queries between visual inspection and general dialogue."""

    VISUAL_KEYWORDS = [
        r"\bhow many\b", r"\bcount\b", r"\bpeople\b", r"\bperson\b", r"\bworker\b",
        r"\bworkers\b", r"\bhelmet\b", r"\bhelmets\b", r"\bhard-?hat\b", r"\bhard-?hats\b",
        r"\bvest\b", r"\bvests\b", r"\bsafety\b", r"\bviolation\b", r"\bviolating\b",
        r"\bwear(ing)?\b", r"\bppe\b", r"\bwhere\b", r"\blocation\b", r"\bsee\b",
        r"\bdetect\b", r"\bimage\b", r"\bphoto\b", r"\bpicture\b", r"\blook\b",
        r"\bstanding\b", r"\bequipment\b", r"\bcompliant\b", r"\bcompliance\b"
    ]

    GENERAL_KEYWORDS = [
        r"\bwhat is the capital\b", r"\bwho wrote\b", r"\bweather\b", r"\btemperature\b",
        r"\bpresident\b", r"\bformula\b", r"\bmath\b", r"\btell me a joke\b",
        r"\bwho is\b", r"\bdefine\b", r"\bhistory of\b", r"\btranslate\b",
        r"\bhow to code\b", r"\bpython\b"
    ]

    def __init__(self, groq_client: Optional[object] = None):
        self.client = groq_client

    def route(self, question: str) -> IntentDecision:
        cleaned = question.strip().lower()

        # Check visual keywords
        for pat in self.VISUAL_KEYWORDS:
            if re.search(pat, cleaned):
                return IntentDecision(
                    needs_detection=True,
                    intent_category="visual_inspection",
                    confidence=0.98,
                    reasoning=f"Query matches visual inspection pattern: '{pat}'."
                )

        # Fast path 2: Known general knowledge intent
        for pat in self.GENERAL_KEYWORDS:
            if re.search(pat, cleaned):
                return IntentDecision(
                    needs_detection=False,
                    intent_category="general_knowledge",
                    confidence=0.95,
                    reasoning=f"Query is non-visual general knowledge, matching '{pat}'."
                )

        # Ambiguous query path: If Groq client is available, ask LLM for single-word classification
        if self.client and settings.GROQ_API_KEY:
            try:
                system_prompt = (
                    "You are a strict query classifier. Decide if the user question requires "
                    "looking at an image to detect objects, people, or safety equipment.\n"
                    "Respond with ONLY JSON: {\"needs_detection\": true|false, \"category\": \"visual_inspection\"|\"general_knowledge\", \"reason\": \"<1 sentence>\"}"
                )
                response = self.client.chat.completions.create(
                    model=settings.GROQ_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question}
                    ],
                    temperature=0.0,
                    max_tokens=80,
                    response_format={"type": "json_object"}
                )
                import json
                parsed = json.loads(response.choices[0].message.content)
                return IntentDecision(
                    needs_detection=bool(parsed.get("needs_detection", True)),
                    intent_category=str(parsed.get("category", "visual_inspection")),
                    confidence=0.90,
                    reasoning=str(parsed.get("reason", "LLM zero-shot intent routing."))
                )
            except Exception as e:
                logger.warning(f"Groq routing fallback error: {e}. Defaulting to image inspection.")

        # Default fallback: When in doubt, perform detection rather than dropping user request
        return IntentDecision(
            needs_detection=True,
            intent_category="visual_inspection",
            confidence=0.75,
            reasoning="Query context is ambiguous; defaulted to object detection to prevent information loss."
        )
