import json
import logging
import os
from dotenv import load_dotenv
from openai import OpenAI

from core.models import DocumentChunk, EvaluationSample
from core.vector_store import QdrantVectorStore

# Load environment variables
load_dotenv()

# Environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = "gpt-4o"

logger = logging.getLogger(__name__)

ADVERSARIAL_PROMPT = """You are an adversarial legal expert generating complex benchmark questions.
Given these legal document excerpts, generate {n} challenging question-answer pairs that:
- Require multi-hop reasoning across clauses
- Test understanding of specific legal terminology  
- Include cross-document interactions where possible
- Are unambiguously answerable from the provided context

Return ONLY valid JSON in this exact format:
[
  {{
    "question": "...",
    "reference_context": "exact quote from the document that answers this",
    "expected_answer": "precise legal answer",
    "doc_ids": ["doc_id1", "doc_id2"]
  }}
]
"""


class AdversarialLawyerAgent:
    """
    Scans ingested documents and generates a 'Golden Dataset' of
    (question, context, expected_answer) triples for RAG benchmarking.
    """

    def __init__(self, vector_store: QdrantVectorStore):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = LLM_MODEL
        self.vector_store = vector_store

    def generate_golden_dataset(
        self, sample_chunks: list[DocumentChunk], n_questions: int = 50
    ) -> list[EvaluationSample]:
        batch_size = 5
        samples: list[EvaluationSample] = []

        for i in range(0, min(len(sample_chunks), n_questions * 2), batch_size):
            batch = sample_chunks[i : i + batch_size]
            context = "\n\n---\n\n".join(
                f"[DOC:{c.doc_id}]\n{c.content}" for c in batch
            )

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": ADVERSARIAL_PROMPT.format(n=batch_size)
                            + f"\n\nDOCUMENTS:\n{context}",
                        }
                    ],
                    temperature=0.7,
                    response_format={"type": "json_object"},
                )

                raw = response.choices[0].message.content
                try:
                    # Clean the response - sometimes LLMs add extra text
                    raw = raw.strip()
                    
                    # Try to extract JSON from the response if it's wrapped in text
                    if raw.startswith('```json'):
                        raw = raw.replace('```json', '').replace('```', '').strip()
                    elif raw.startswith('```'):
                        raw = raw.replace('```', '').strip()
                    
                    # Find JSON array or object in the text
                    start_idx = raw.find('[')
                    end_idx = raw.rfind(']')
                    if start_idx != -1 and end_idx != -1:
                        raw = raw[start_idx:end_idx+1]
                    
                    pairs = json.loads(raw)
                    
                    # Check for API error responses
                    if isinstance(pairs, dict) and "error" in pairs:
                        logger.warning(f"API error in batch {i}: {pairs.get('error')}")
                        continue
                    
                    # Handle different response formats
                    if isinstance(pairs, dict):
                        # If it's a dict, try to get the list from common keys
                        if "questions" in pairs:
                            pairs = pairs["questions"]
                        elif "samples" in pairs:
                            pairs = pairs["samples"]
                        elif "data" in pairs:
                            pairs = pairs["data"]
                        else:
                            # Take the first list value
                            values = list(pairs.values())
                            if values and isinstance(values[0], list):
                                pairs = values[0]
                            else:
                                logger.warning(f"Could not extract list from dict keys: {list(pairs.keys())}, raw: {raw[:200]}")
                                continue
                    
                    # Ensure pairs is a list
                    if not isinstance(pairs, list):
                        logger.warning(f"Expected list but got {type(pairs)}, raw response: {raw[:200]}")
                        continue

                    logger.info(f"Successfully parsed {len(pairs)} question-answer pairs from batch {i}")

                    for pair in pairs:
                        if not isinstance(pair, dict):
                            logger.warning(f"Expected dict but got {type(pair)}, skipping item")
                            continue
                            
                        # Validate required fields
                        required_fields = ["question", "reference_context", "expected_answer"]
                        if not all(key in pair for key in required_fields):
                            logger.warning(f"Missing required fields in pair: {pair.keys()}")
                            continue
                            
                        samples.append(
                            EvaluationSample(
                                question=pair["question"],
                                reference_context=pair["reference_context"],
                                expected_answer=pair["expected_answer"],
                                doc_ids=pair.get("doc_ids", [c.doc_id for c in batch]),
                            )
                        )

                except json.JSONDecodeError as e:
                    logger.warning(f"JSON decode error in batch {i}: {e}, raw response: {raw[:200]}")
                    continue

                if len(samples) >= n_questions:
                    break

            except Exception as e:
                logger.warning(f"Batch {i} generation failed: {e}")

        return samples[:n_questions]
