import json
import logging
import os
from dotenv import load_dotenv
from openai import OpenAI

from core.models import EvaluationResult, EvaluationSample, RAGResponse

# Load environment variables
load_dotenv()

# Environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = "gpt-4o"

logger = logging.getLogger(__name__)

FAITHFULNESS_PROMPT = """You are a legal compliance auditor performing hallucination detection.

Given a RAG system's answer and the source context it retrieved, evaluate faithfulness:

1. Extract each factual claim from the ANSWER.
2. For each claim, determine if it is directly supported by the CONTEXT.
3. Return a score from 0.0 to 1.0 where:
   - 1.0 = every claim is grounded in the context
   - 0.0 = most claims are hallucinated or unsupported

Return ONLY JSON: {{"faithfulness": 0.95, "flagged_claims": ["claim1 is unsupported", ...]}}

CONTEXT:
{context}

ANSWER:
{answer}
"""

RELEVANCE_PROMPT = """Rate how well this ANSWER addresses the QUESTION on a scale 0.0-1.0.
Consider: Does it answer the actual question asked? Is it on-topic?

Return ONLY JSON: {{"answer_relevance": 0.87}}

QUESTION: {question}
ANSWER: {answer}
"""


class ComplianceAuditorAgent:
    """
    LLM-as-a-judge agent that detects hallucinations and scores RAG responses
    against the faithfulness and answer relevance metrics.
    """

    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = LLM_MODEL

    def evaluate(self, sample: EvaluationSample, rag_response: RAGResponse) -> dict:
        context = "\n\n".join(c.chunk.content for c in rag_response.retrieved_chunks)

        faithfulness, flagged = self._score_faithfulness(rag_response.answer, context)
        relevance = self._score_relevance(sample.question, rag_response.answer)

        if flagged:
            logger.warning(f"Hallucination flags for '{sample.question[:60]}': {flagged}")

        return {
            "faithfulness": faithfulness,
            "answer_relevance": relevance,
            "flagged_claims": flagged,
        }

    def _score_faithfulness(self, answer: str, context: str) -> tuple[float, list[str]]:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": FAITHFULNESS_PROMPT.format(
                            context=context[:4000], answer=answer
                        ),
                    }
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
            data = json.loads(response.choices[0].message.content)
            return data.get("faithfulness", 0.0), data.get("flagged_claims", [])
        except Exception as e:
            logger.error(f"Faithfulness scoring failed: {e}")
            return 0.0, []

    def _score_relevance(self, question: str, answer: str) -> float:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": RELEVANCE_PROMPT.format(
                            question=question, answer=answer
                        ),
                    }
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
            data = json.loads(response.choices[0].message.content)
            return data.get("answer_relevance", 0.0)
        except Exception as e:
            logger.error(f"Relevance scoring failed: {e}")
            return 0.0
