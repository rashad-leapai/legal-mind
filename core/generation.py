import os
from dotenv import load_dotenv
from openai import OpenAI

from core.models import RAGResponse, RetrievedChunk

# Load environment variables
load_dotenv()

# Environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = "gpt-4o"

SYSTEM_PROMPT = """You are LegalMind, a precise legal research assistant. Your rules are absolute:

1. Answer ONLY from the provided context. Do not use any prior knowledge or make assumptions.
2. If the context does not contain enough information, respond exactly: "I don't know based on the provided documents."
3. Every factual claim MUST reference its source using [DOC:doc_id] inline notation.
4. Be precise and concise. Legal accuracy over verbosity.
5. If multiple documents conflict, explicitly note the discrepancy.
6. When information IS clearly stated in the context, provide a complete answer with proper citations.
7. Only refuse to answer when the specific information is truly missing from the provided context.

Format your response as:
ANSWER: <your answer with inline [DOC:doc_id] citations>
SOURCES: <comma-separated list of all doc_ids used>
"""


class GenerationLayer:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = LLM_MODEL

    def generate(self, query: str, chunks: list[RetrievedChunk]) -> RAGResponse:
        context = self._build_context(chunks)
        source_ids = list({c.chunk.doc_id for c in chunks})

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"CONTEXT:\n{context}\n\nQUESTION: {query}",
            },
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,  # Deterministic for legal accuracy
            max_completion_tokens=1500,
        )

        answer = response.choices[0].message.content
        return RAGResponse(
            answer=answer,
            source_ids=source_ids,
            retrieved_chunks=chunks,
        )

    def _build_context(self, chunks: list[RetrievedChunk]) -> str:
        parts = []
        for c in chunks:
            doc_id = c.chunk.doc_id
            filename = c.chunk.metadata.get("filename", "unknown")
            parts.append(f"[DOC:{doc_id}] ({filename})\n{c.chunk.content}")
        return "\n\n---\n\n".join(parts)
