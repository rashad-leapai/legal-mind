import json
import re
import logging
import os
from dotenv import load_dotenv
from openai import OpenAI

from core.models import RAGResponse, RetrievedChunk

# Load environment variables
load_dotenv()

# Environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = "gpt-4o"

logger = logging.getLogger(__name__)

CONTEXT_PRECISION_PROMPT = """You are evaluating search result quality for legal research.

Given this QUESTION and the RETRIEVED CHUNKS (in order), score context precision 0.0-1.0:
- 1.0 = most relevant chunks are ranked highest
- 0.0 = relevant chunks are buried at the bottom or missing

Return ONLY JSON: {{"context_precision": 0.82, "reasoning": "top 2 chunks are highly relevant..."}}

QUESTION: {question}

RETRIEVED CHUNKS (ordered by rank):
{chunks}
"""


class ShepardizerAgent:
    """
    Validates citations in RAG responses: checks that every [DOC:id] reference
    exists in the retrieved source documents and measures context precision.
    """

    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = LLM_MODEL

    def validate(self, question: str, response: RAGResponse) -> dict:
        citation_result = self._validate_citations(response)
        precision = self._score_context_precision(question, response.retrieved_chunks)

        return {
            "citations_valid": citation_result["valid"],
            "broken_citations": citation_result["broken"],
            "context_precision": precision,
        }

    def _validate_citations(self, response: RAGResponse) -> dict:
        cited_ids = set(re.findall(r"\[DOC:([a-zA-Z0-9_\-]+)\]", response.answer))
        available_ids = {c.chunk.doc_id for c in response.retrieved_chunks}
        available_ids.update(response.source_ids)

        broken = cited_ids - available_ids
        valid = len(broken) == 0

        if broken:
            logger.warning(f"Broken citations detected: {broken}")

        return {"valid": valid, "broken": list(broken), "cited": list(cited_ids)}

    def _score_context_precision(
        self, question: str, chunks: list[RetrievedChunk]
    ) -> float:
        if not chunks:
            return 0.0

        chunks_text = "\n\n".join(
            f"Rank {i+1} [DOC:{c.chunk.doc_id}]: {c.chunk.content[:300]}"
            for i, c in enumerate(chunks)
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": CONTEXT_PRECISION_PROMPT.format(
                            question=question, chunks=chunks_text
                        ),
                    }
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
            data = json.loads(response.choices[0].message.content)
            return data.get("context_precision", 0.0)
        except Exception as e:
            logger.error(f"Context precision scoring failed: {e}")
            return 0.0
