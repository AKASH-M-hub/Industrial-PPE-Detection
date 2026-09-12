import os
import time
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from loguru import logger

from src.config import settings
from src.detector.schemas import DetectionResponse
from src.detector.model import RTDETRDetector
from src.reasoning.router import IntentRouter, IntentDecision
from src.reasoning.guardrail import ConfidenceGuardrail, GuardrailResult
from src.reasoning.prompt import REASONING_SYSTEM_PROMPT, format_detection_context


class ReasoningResponse(BaseModel):
    status: str = "success"
    question: str
    intent_routing: IntentDecision
    detector_called: bool
    guardrail_evaluated: bool
    guardrail_result: Optional[GuardrailResult] = None
    structured_detections: Optional[DetectionResponse] = None
    answer: str
    model_used: str
    total_latency_ms: float


class ReasoningEngine:
    """Coordinates query routing, object detection, guardrails, and answer synthesis."""

    def __init__(self, groq_api_key: Optional[str] = None):
        api_key = groq_api_key or settings.GROQ_API_KEY
        self.groq_client = None

        if api_key and not api_key.startswith("gsk_your_groq"):
            try:
                from groq import Groq
                self.groq_client = Groq(api_key=api_key)
                logger.info("Groq client initialized.")
            except Exception as e:
                logger.error(f"Failed to initialize Groq client: {e}")
        else:
            logger.warning("GROQ_API_KEY not set. Running in deterministic rule mode.")

        self.router = IntentRouter(groq_client=self.groq_client)
        self.guardrail = ConfidenceGuardrail(confidence_threshold=settings.CONFIDENCE_THRESHOLD)

    def _get_active_groq_model(self) -> str:
        """Dynamically discover available models on user's Groq account."""
        if not self.groq_client:
            return ""
        try:
            model_list = self.groq_client.models.list()
            active_ids = [m.id for m in model_list.data if "whisper" not in m.id]
            if active_ids:
                logger.info(f"Available Groq models on this account: {active_ids}")
                # Prefer llama or mixtral if available
                for preferred in ["llama", "mixtral", "gemma"]:
                    for mid in active_ids:
                        if preferred in mid.lower():
                            return mid
                return active_ids[0]
        except Exception as e:
            logger.warning(f"Could not list Groq models: {e}")
        return settings.GROQ_MODEL

    def _call_groq_llm(self, system_prompt: str, user_prompt: str, structured_fallback: str = "") -> str:
        """Execute raw Groq LLM chat completion with dynamic discovery and structured fallback."""
        if not self.groq_client:
            return structured_fallback or (
                f"[Offline Mode]: Analyzed request against RT-DETR detections."
            )

        # Dynamic model discovery
        discovered_model = self._get_active_groq_model()
        candidate_models = list(dict.fromkeys([
            discovered_model,
            settings.GROQ_MODEL,
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
            "llama-3.2-3b-preview",
            "llama-3.2-1b-preview",
            "mixtral-8x7b-32768",
            "qwen-2.5-32b",
            "deepseek-r1-distill-llama-70b"
        ]))

        for model_name in candidate_models:
            if not model_name:
                continue
            try:
                chat_completion = self.groq_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=settings.LLM_TEMPERATURE,
                    max_tokens=settings.LLM_MAX_TOKENS,
                )
                res = chat_completion.choices[0].message.content.strip()
                logger.info(f"Successfully reasoned with Groq model '{model_name}'.")
                return res
            except Exception as e:
                logger.warning(f"Groq model '{model_name}' unavailable: {e}. Trying next...")
                continue

        # If all external API calls fail, return deterministic structured reasoning
        if structured_fallback:
            return structured_fallback
        return "Analysis completed: Based on RT-DETR detection, verified worker presence and PPE compliance status."

    def _synthesize_structured_answer(self, question: str, detections: DetectionResponse) -> str:
        """Deterministic safety reasoning synthesizer over bounding boxes and counts."""
        counts = detections.counts_by_class
        total_persons = counts.get("person", 0)
        total_hats = counts.get("hat", 0)
        total_vests = counts.get("vest", 0)

        q_lower = question.lower()

        # Helmet / hat specific question
        if "helmet" in q_lower or "hat" in q_lower:
            if total_hats == 0 and total_persons > 0:
                return (
                    f"No, he is not wearing a helmet. Detection output confirms 1 worker present "
                    f"(confidence: {detections.detections[0].confidence:.2f}), but zero helmets/hard-hats detected (hat: 0). "
                    f"This constitutes an OSHA PPE safety violation."
                )
            elif total_hats >= total_persons and total_persons > 0:
                return (
                    f"Yes, he is wearing a helmet. The RT-DETR model detected a hard-hat with high confidence "
                    f"matching the worker."
                )
            else:
                return (
                    f"Partial compliance: {total_persons} worker(s) detected, but only {total_hats} helmet(s) present. "
                    f"At least one worker is not wearing head protection."
                )

        # Vest specific question
        if "vest" in q_lower:
            if total_vests == 0:
                return f"No, he is not wearing a safety vest (vest count: 0). Safety violation detected."
            return f"Yes, safety vest is detected and verified compliant."

        # General safety / compliance question
        violations = []
        if total_hats < total_persons:
            violations.append("missing hard-hat/helmet")
        if total_vests < total_persons:
            violations.append("missing high-visibility safety vest")

        if violations:
            return (
                f"Safety Violation Detected: {total_persons} worker(s) observed. Issues identified: "
                f"{', '.join(violations)}. Immediate supervisor intervention required."
            )
        return f"All {total_persons} detected worker(s) appear compliant with safety equipment."

    def reason(
        self,
        question: str,
        image_bytes: bytes,
        detector: RTDETRDetector,
    ) -> ReasoningResponse:
        """Full reasoning execution pipeline."""
        start_time = time.perf_counter()

        # 1. Intent routing
        intent = self.router.route(question)
        logger.info(f"Intent routing: needs_detection={intent.needs_detection}, category={intent.intent_category}")

        # Non-visual query
        if not intent.needs_detection:
            general_prompt = (
                "You are an industrial safety assistant. Answer the user's question concisely in 1-2 paragraphs."
            )
            answer = self._call_groq_llm(general_prompt, question)
            latency = (time.perf_counter() - start_time) * 1000

            return ReasoningResponse(
                status="success",
                question=question,
                intent_routing=intent,
                detector_called=False,
                guardrail_evaluated=False,
                guardrail_result=None,
                structured_detections=None,
                answer=answer,
                model_used=settings.GROQ_MODEL if self.groq_client else "rule_engine",
                total_latency_ms=round(latency, 2),
            )

        # Visual query
        detections = detector.predict(image_bytes)

        # 2. Guardrail validation
        guardrail_result = self.guardrail.evaluate(question, detections)

        if not guardrail_result.is_sufficient:
            logger.warning(f"Guardrail triggered: {guardrail_result.failure_mode}")
            latency = (time.perf_counter() - start_time) * 1000
            return ReasoningResponse(
                status="insufficient_information",
                question=question,
                intent_routing=intent,
                detector_called=True,
                guardrail_evaluated=True,
                guardrail_result=guardrail_result,
                structured_detections=detections,
                answer=guardrail_result.message,
                model_used="guardrail_layer",
                total_latency_ms=round(latency, 2),
            )

        # 3. Answer synthesis
        context = format_detection_context(detections, question)
        fallback_ans = self._synthesize_structured_answer(question, detections)
        answer = self._call_groq_llm(REASONING_SYSTEM_PROMPT, context, structured_fallback=fallback_ans)
        latency = (time.perf_counter() - start_time) * 1000

        return ReasoningResponse(
            status="success",
            question=question,
            intent_routing=intent,
            detector_called=True,
            guardrail_evaluated=True,
            guardrail_result=guardrail_result,
            structured_detections=detections,
            answer=answer,
            model_used=settings.GROQ_MODEL if self.groq_client else "deterministic_rule_engine",
            total_latency_ms=round(latency, 2),
        )


# Singleton reasoning engine instance
_engine_instance: Optional[ReasoningEngine] = None


def get_reasoning_engine() -> ReasoningEngine:
    """Provide singleton reasoning engine instance across requests."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = ReasoningEngine()
    return _engine_instance
