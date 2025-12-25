"""
LLM service for context-constrained answer generation.

CRITICAL: The LLM is constrained to ONLY use provided context.
It must NEVER use prior knowledge or hallucinate answers.
"""

from typing import Optional
import structlog

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import Settings, get_settings

logger = structlog.get_logger(__name__)


# =============================================================================
# SYSTEM PROMPT: NO HALLUCINATION - CONTEXT ONLY
# =============================================================================

SYSTEM_PROMPT = """You are a precise document-based question answering assistant.

ABSOLUTE RULES - VIOLATION IS FORBIDDEN:

1. You may ONLY use information explicitly stated in the provided CONTEXT.
2. You must NEVER use your training data, prior knowledge, or make assumptions.
3. You must NEVER invent, guess, or infer information not in the CONTEXT.
4. If the CONTEXT does not contain enough information to answer, you MUST respond:
   "I don't know based on the provided documents."
5. If the CONTEXT is empty or completely irrelevant, you MUST respond:
   "Answer not found in documents."
6. Always cite which source(s) you used with [Source N] notation.
7. Be concise, accurate, and direct.
8. Never apologize or explain that you're an AI.

CORRECT BEHAVIOR:
- Context says "The policy allows 3 remote days" → Answer: "The policy allows 3 remote work days per week. [Source 1]"
- Context doesn't mention the topic → Answer: "I don't know based on the provided documents."
- Empty context → Answer: "Answer not found in documents."

INCORRECT BEHAVIOR (NEVER DO THIS):
- Making up statistics or dates not in context
- Providing general knowledge answers
- Saying "typically" or "usually" without context support
- Adding information "for completeness"

You exist ONLY to relay information FROM the documents. Nothing else."""


USER_PROMPT_TEMPLATE = """CONTEXT (Use ONLY this information):
{context}

SOURCES:
{sources}

QUESTION: {question}

Provide a precise answer using ONLY the context above. Cite sources with [Source N]."""


# =============================================================================
# LLM Service
# =============================================================================

class LLMService:
    """
    LLM service for generating answers constrained to provided context.
    
    Features:
    - Strict system prompt enforcing no hallucination
    - Context-only answer generation
    - Confidence scoring
    - Automatic retry with exponential backoff
    """
    
    # Phrases indicating LLM couldn't find answer
    NO_ANSWER_PHRASES = [
        "answer not found in documents",
        "i don't know",
        "not mentioned in",
        "no information",
        "cannot find",
        "cannot determine",
        "not specified in",
        "not stated in",
        "no relevant information",
        "insufficient information",
    ]
    
    def __init__(self, settings: Optional[Settings] = None):
        """Initialize the LLM service."""
        self._settings = settings or get_settings()
        self._client: Optional[AsyncOpenAI] = None
        
        logger.info(
            "Initialized LLMService",
            model=self._settings.llm_model,
            temperature=self._settings.llm_temperature,
        )
    
    @property
    def client(self) -> AsyncOpenAI:
        """Get or create AsyncOpenAI client."""
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self._settings.openai_api_key_value,
                organization=self._settings.openai_org_id,
                timeout=self._settings.llm_timeout,
            )
        return self._client
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def _call_llm(self, system: str, user: str) -> str:
        """Make LLM API call with retry logic."""
        response = await self.client.chat.completions.create(
            model=self._settings.llm_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=self._settings.llm_temperature,
            max_tokens=self._settings.llm_max_tokens,
        )
        return response.choices[0].message.content or ""
    
    def _format_context(
        self,
        context_texts: list[str],
        source_labels: list[str],
    ) -> tuple[str, str]:
        """Format context and sources for the prompt."""
        if not context_texts:
            return "", ""
        
        # Format context with source labels
        context_parts = []
        for i, (text, label) in enumerate(zip(context_texts, source_labels), 1):
            context_parts.append(f"[Source {i}] ({label}):\n{text}")
        
        formatted_context = "\n\n---\n\n".join(context_parts)
        
        # Format source list
        sources = ", ".join([
            f"[Source {i}]: {label}"
            for i, label in enumerate(source_labels, 1)
        ])
        
        return formatted_context, sources
    
    def _is_no_answer(self, response: str) -> bool:
        """Check if response indicates no answer was found."""
        response_lower = response.lower()
        return any(phrase in response_lower for phrase in self.NO_ANSWER_PHRASES)
    
    def _calculate_confidence(
        self,
        answer: str,
        context_count: int,
    ) -> float:
        """
        Calculate confidence score for the answer.
        
        Scoring:
        - 0.0: No answer found
        - 0.5-0.95: Answer found with varying confidence
        """
        if self._is_no_answer(answer):
            return 0.0
        
        confidence = 0.5
        
        # More context = higher confidence
        if context_count >= 2:
            confidence += 0.15
        if context_count >= 3:
            confidence += 0.1
        
        # Longer, more detailed answers = higher confidence
        if len(answer) > 100:
            confidence += 0.1
        if len(answer) > 200:
            confidence += 0.05
        
        # Has citations = higher confidence
        if "[source" in answer.lower():
            confidence += 0.1
        
        return min(confidence, 0.95)
    
    async def generate_answer(
        self,
        question: str,
        context_texts: list[str],
        source_labels: list[str],
    ) -> tuple[str, float]:
        """
        Generate an answer using ONLY the provided context.
        
        Args:
            question: User's question.
            context_texts: Retrieved document chunks.
            source_labels: Labels for each source (e.g., "doc.pdf, page 5").
            
        Returns:
            Tuple of (answer, confidence_score).
            
        IMPORTANT: If context is empty, returns "Answer not found in documents."
        """
        # Handle empty context - NO generation attempted
        if not context_texts:
            logger.info(
                "No context provided, returning no-answer",
                question=question[:100],
            )
            return "Answer not found in documents.", 0.0
        
        # Format prompt
        formatted_context, formatted_sources = self._format_context(
            context_texts, source_labels
        )
        
        user_prompt = USER_PROMPT_TEMPLATE.format(
            context=formatted_context,
            sources=formatted_sources,
            question=question,
        )
        
        # Call LLM
        answer = await self._call_llm(SYSTEM_PROMPT, user_prompt)
        answer = answer.strip()
        
        # Calculate confidence
        confidence = self._calculate_confidence(answer, len(context_texts))
        
        logger.info(
            "Generated answer",
            question_length=len(question),
            context_count=len(context_texts),
            answer_length=len(answer),
            confidence=confidence,
            is_no_answer=self._is_no_answer(answer),
        )
        
        return answer, confidence
    
    async def health_check(self) -> dict[str, str]:
        """Check if LLM service is healthy."""
        try:
            response = await self._call_llm(
                system="Respond with exactly: OK",
                user="Health check",
            )
            if response.strip():
                return {
                    "status": "healthy",
                    "model": self._settings.llm_model,
                }
            return {"status": "unhealthy", "error": "Empty response"}
        except Exception as e:
            logger.error("LLM health check failed", error=str(e))
            return {"status": "unhealthy", "error": str(e)}
