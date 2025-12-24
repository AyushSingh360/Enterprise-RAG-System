"""
LLM service for context-based answer generation.
Implements strict no-hallucination policies through system prompts.
"""

from typing import Optional
import structlog

from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage, MessageRole

from app.config import Settings, get_settings

logger = structlog.get_logger(__name__)


# System prompt that enforces no hallucination
NO_HALLUCINATION_SYSTEM_PROMPT = """You are a precise and accurate AI assistant for an Enterprise RAG system.

CRITICAL RULES - YOU MUST FOLLOW THESE EXACTLY:

1. ONLY use information from the provided context to answer questions.
2. NEVER use your prior knowledge or training data to answer.
3. NEVER make assumptions or inferences beyond what is explicitly stated in the context.
4. If the context does not contain sufficient information to answer the question, you MUST respond with exactly: "I don't know based on the provided documents."
5. If the context is empty or not relevant to the question, respond with: "Answer not found in documents."
6. Always cite which document(s) you used to formulate your answer.
7. Be concise and direct in your responses.
8. Do not add any information that is not present in the context.
9. If you are uncertain about any part of your answer, acknowledge the uncertainty.
10. Never fabricate facts, statistics, dates, names, or any other information.

Remember: Your primary purpose is to provide accurate information FROM THE DOCUMENTS ONLY.
Accuracy and honesty are more important than providing an answer."""


ANSWER_GENERATION_PROMPT = """Based ONLY on the following context, answer the user's question.

CONTEXT:
{context}

SOURCES PROVIDED:
{sources}

USER QUESTION: {question}

INSTRUCTIONS:
- Use ONLY the information from the context above
- If the context doesn't contain the answer, say "I don't know based on the provided documents."
- Cite the relevant source(s) in your answer
- Be concise and accurate

YOUR ANSWER:"""


class LLMService:
    """
    Service for LLM-based answer generation with strict context adherence.
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize the LLM service.
        
        Args:
            settings: Application settings.
        """
        self._settings = settings or get_settings()
        self._llm: Optional[OpenAI] = None
    
    @property
    def llm(self) -> OpenAI:
        """Get or create the LLM instance."""
        if self._llm is None:
            self._llm = OpenAI(
                model=self._settings.llm_model,
                api_key=self._settings.openai_api_key,
                temperature=self._settings.llm_temperature,
                max_tokens=self._settings.llm_max_tokens,
                system_prompt=NO_HALLUCINATION_SYSTEM_PROMPT,
            )
            logger.info(
                "Initialized LLM",
                model=self._settings.llm_model,
                temperature=self._settings.llm_temperature
            )
        return self._llm
    
    async def generate_answer(
        self,
        question: str,
        context_texts: list[str],
        source_citations: list[str]
    ) -> tuple[str, float]:
        """
        Generate an answer based on the provided context.
        
        Args:
            question: The user's question.
            context_texts: List of relevant context passages.
            source_citations: List of source identifiers.
            
        Returns:
            Tuple of (answer, confidence_score).
        """
        # Handle empty context
        if not context_texts:
            logger.info("No context provided, returning no-answer response")
            return "Answer not found in documents.", 0.0
        
        # Format context and sources
        formatted_context = "\n\n---\n\n".join([
            f"[Source {i+1}]: {text}"
            for i, text in enumerate(context_texts)
        ])
        
        formatted_sources = ", ".join([
            f"[{i+1}] {source}"
            for i, source in enumerate(source_citations)
        ])
        
        # Create the prompt
        user_prompt = ANSWER_GENERATION_PROMPT.format(
            context=formatted_context,
            sources=formatted_sources,
            question=question
        )
        
        # Generate response
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=NO_HALLUCINATION_SYSTEM_PROMPT),
            ChatMessage(role=MessageRole.USER, content=user_prompt)
        ]
        
        response = await self.llm.achat(messages)
        answer = response.message.content.strip()
        
        # Calculate confidence based on response characteristics
        confidence = self._calculate_confidence(answer, context_texts)
        
        logger.info(
            "Generated answer",
            question_length=len(question),
            context_count=len(context_texts),
            answer_length=len(answer),
            confidence=confidence
        )
        
        return answer, confidence
    
    def _calculate_confidence(
        self,
        answer: str,
        context_texts: list[str]
    ) -> float:
        """
        Calculate a confidence score for the answer.
        
        This is a heuristic-based calculation considering:
        - Whether the answer indicates uncertainty
        - Length and specificity of the answer
        - Presence of citations
        
        Args:
            answer: The generated answer.
            context_texts: The context used for generation.
            
        Returns:
            Confidence score between 0.0 and 1.0.
        """
        answer_lower = answer.lower()
        
        # Low confidence indicators
        low_confidence_phrases = [
            "i don't know",
            "not found in documents",
            "answer not found",
            "no information",
            "cannot determine",
            "unclear from the context",
            "not mentioned",
            "not specified"
        ]
        
        for phrase in low_confidence_phrases:
            if phrase in answer_lower:
                return 0.0
        
        # Base confidence starts at 0.5
        confidence = 0.5
        
        # Increase confidence if answer is substantive
        if len(answer) > 50:
            confidence += 0.1
        if len(answer) > 100:
            confidence += 0.1
        
        # Increase confidence if sources are cited
        if any(f"[{i+1}]" in answer or f"source {i+1}" in answer_lower 
               for i in range(len(context_texts))):
            confidence += 0.15
        
        # Increase confidence based on context quantity
        if len(context_texts) >= 2:
            confidence += 0.1
        if len(context_texts) >= 3:
            confidence += 0.05
        
        # Cap at 0.95 (never 100% confident)
        return min(confidence, 0.95)
    
    async def health_check(self) -> dict[str, str]:
        """Check if the LLM service is healthy."""
        try:
            # Simple test query
            messages = [
                ChatMessage(role=MessageRole.USER, content="Say 'OK' if you are operational.")
            ]
            response = await self.llm.achat(messages)
            
            if response.message.content:
                return {"status": "healthy", "model": self._settings.llm_model}
            return {"status": "unhealthy", "error": "Empty response from LLM"}
        except Exception as e:
            logger.error("LLM health check failed", error=str(e))
            return {"status": "unhealthy", "error": str(e)}
