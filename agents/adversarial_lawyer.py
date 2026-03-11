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

ADVERSARIAL_PROMPT = """You are an expert legal analyst. Generate {n} challenging legal questions from the provided documents.

Focus on:
- Specific clauses and their implications
- Key terms and definitions
- Legal obligations and restrictions
- Important dates, parties, and procedures

For each document excerpt, create questions that can be answered directly from the text.

Return ONLY a JSON array in this exact format:
[
  {{
    "question": "What are the specific obligations under Section X?",
    "reference_context": "exact text from document that contains the answer",
    "expected_answer": "clear answer based on the reference context",
    "doc_ids": ["doc_id"]
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
        batch_size = 3  # Smaller batches for better quality
        samples: list[EvaluationSample] = []

        # Filter chunks to get meaningful content
        meaningful_chunks = [c for c in sample_chunks if len(c.content.split()) > 20]
        
        if not meaningful_chunks:
            logger.warning("No meaningful chunks found for question generation")
            return []

        for i in range(0, min(len(meaningful_chunks), n_questions * 3), batch_size):
            batch = meaningful_chunks[i : i + batch_size]
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
                        logger.info(f"API declined to generate questions for batch {i}: {pairs.get('error')}")
                        continue
                    
                    # Check for insufficient content response
                    if isinstance(pairs, dict) and any(key in str(pairs).lower() for key in ['not enough', 'insufficient', 'cannot generate']):
                        logger.debug(f"Insufficient content for batch {i}, skipping")
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
                                logger.debug(f"Could not extract list from dict keys: {list(pairs.keys())}, raw: {raw[:200]}")
                                continue
                    
                    # Ensure pairs is a list
                    if not isinstance(pairs, list):
                        logger.debug(f"Expected list but got {type(pairs)}, raw response: {raw[:200]}")
                        continue

                    logger.info(f"Successfully parsed {len(pairs)} question-answer pairs from batch {i}")

                    for pair in pairs:
                        if not isinstance(pair, dict):
                            # Try to parse string as JSON if it looks like a JSON object
                            if isinstance(pair, str) and pair.strip().startswith('{'):
                                try:
                                    pair = json.loads(pair.strip())
                                except json.JSONDecodeError:
                                    logger.debug(f"Skipping non-dict item: {type(pair)} - {str(pair)[:100]}")
                                    continue
                            else:
                                logger.debug(f"Skipping non-dict item: {type(pair)} - {str(pair)[:100]}")
                                continue
                            
                        # Validate required fields
                        required_fields = ["question", "reference_context", "expected_answer"]
                        if not all(key in pair for key in required_fields):
                            logger.debug(f"Missing required fields in pair: {pair.keys()}")
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
                    logger.debug(f"JSON decode error in batch {i}: {e}, raw response: {raw[:200]}")
                    continue

                if len(samples) >= n_questions:
                    break

            except Exception as e:
                logger.warning(f"Batch {i} generation failed: {e}")

        return samples[:n_questions]
