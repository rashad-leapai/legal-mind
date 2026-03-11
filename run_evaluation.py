#!/usr/bin/env python3
"""
LegalMind RAG Production Evaluation Script

This script runs the complete evaluation pipeline including:
1. Synthetic test data generation (Golden Dataset)
2. RAG Triad metrics (Faithfulness, Answer Relevance, Context Precision)
3. Citation validation and source attribution
4. Performance benchmarking

Usage:
    python run_evaluation.py --generate-golden  # Generate test dataset
    python run_evaluation.py --run-metrics      # Run evaluation metrics
    python run_evaluation.py --full             # Full evaluation pipeline
"""
import argparse
import json
import time
from pathlib import Path
from typing import List

from agents.adversarial_lawyer import AdversarialLawyerAgent
from agents.compliance_auditor import ComplianceAuditorAgent
from agents.shepardizer import ShepardizerAgent
from core.ingestion import IngestionPipeline
from core.models import DocumentChunk, EvaluationSample, RAGResponse
from core.pipeline import LegalMindRAG
from core.vector_store import QdrantVectorStore

# Evaluation thresholds (moved from config)
FAITHFULNESS_THRESHOLD = 0.9
RELEVANCE_THRESHOLD = 0.8
PRECISION_THRESHOLD = 0.85

def generate_golden_dataset(docs_path: Path, output_path: Path, n_questions: int = 50) -> List[EvaluationSample]:
    """Generate synthetic test data from legal documents."""
    print(f"🏗️  Generating Golden Dataset ({n_questions} questions)...")
    
    # Initialize components
    ingestion = IngestionPipeline()
    vector_store = QdrantVectorStore()
    adversarial_agent = AdversarialLawyerAgent(vector_store)
    
    # Ingest documents
    all_chunks = []
    doc_files = list(docs_path.glob("*.pdf")) + list(docs_path.glob("*.txt"))
    
    for doc_file in doc_files[:10]:  # Limit to 10 docs for speed
        print(f"📄 Processing {doc_file.name}")
        chunks = ingestion.ingest(doc_file)
        all_chunks.extend(chunks)
    
    # Generate test questions
    print("🤖 Generating adversarial questions...")
    golden_samples = adversarial_agent.generate_golden_dataset(all_chunks, n_questions)
    
    # Save results
    output_data = [
        {
            "question": s.question,
            "reference_context": s.reference_context,
            "expected_answer": s.expected_answer,
            "doc_ids": s.doc_ids
        }
        for s in golden_samples
    ]
    
    output_path.write_text(json.dumps(output_data, indent=2), encoding="utf-8")
    
    if len(golden_samples) == 0:
        print("⚠️  Warning: No questions were generated. This may be due to:")
        print("   - Insufficient document content")
        print("   - API rate limiting") 
        print("   - Document format issues")
        print(f"💾 Saved empty dataset to {output_path}")
    else:
        print(f"💾 Saved {len(golden_samples)} questions to {output_path}")
    
    return golden_samples

def run_evaluation_metrics(golden_dataset_path: Path) -> dict:
    """Run the RAG Triad evaluation metrics."""
    print("📊 Running RAG Evaluation Metrics...")
    
    # Load golden dataset
    golden_data = json.loads(golden_dataset_path.read_text(encoding="utf-8"))
    
    if not golden_data:
        print("❌ Golden dataset is empty. Run with --generate-golden first to create test data.")
        return {"evaluation_passed": False, "error": "Empty golden dataset"}
    
    golden_samples = [
        EvaluationSample(
            question=item["question"],
            reference_context=item["reference_context"], 
            expected_answer=item["expected_answer"],
            doc_ids=item["doc_ids"]
        )
        for item in golden_data
    ]
    
    # Initialize components
    rag_pipeline = LegalMindRAG()
    compliance_auditor = ComplianceAuditorAgent()
    shepardizer = ShepardizerAgent()
    
    # Track results
    results = {
        "total_samples": len(golden_samples),
        "faithfulness_scores": [],
        "relevance_scores": [],
        "precision_scores": [],
        "citation_accuracy": [],
        "failed_samples": []
    }
    
    for i, sample in enumerate(golden_samples):
        print(f"📝 Evaluating sample {i+1}/{len(golden_samples)}")
        
        try:
            # Get RAG response
            start_time = time.time()
            rag_response = rag_pipeline.query(sample.question)
            response_time = time.time() - start_time
            
            # Evaluate faithfulness and relevance
            audit_result = compliance_auditor.evaluate(sample, rag_response)
            faithfulness = audit_result["faithfulness"]
            relevance = audit_result["answer_relevance"]
            
            # Evaluate context precision and citations
            precision = compliance_auditor.evaluate_context_precision(sample, rag_response)
            shepard_result = shepardizer.validate(sample.question, rag_response)
            citations_valid = shepard_result["citations_valid"]
            
            # Record results
            results["faithfulness_scores"].append(faithfulness)
            results["relevance_scores"].append(relevance)
            results["precision_scores"].append(precision)
            results["citation_accuracy"].append(1.0 if citations_valid else 0.0)
            
            # Check thresholds
            if faithfulness < FAITHFULNESS_THRESHOLD:
                results["failed_samples"].append({
                    "sample_id": i,
                    "question": sample.question,
                    "reason": f"Faithfulness {faithfulness:.3f} < {FAITHFULNESS_THRESHOLD}",
                    "response_time": response_time
                })
                
        except Exception as e:
            print(f"❌ Error evaluating sample {i}: {e}")
            results["failed_samples"].append({
                "sample_id": i,
                "question": sample.question,
                "reason": f"Exception: {str(e)}",
                "response_time": 0
            })
    
    # Calculate aggregates with safety checks
    if results["faithfulness_scores"]:
        results["avg_faithfulness"] = sum(results["faithfulness_scores"]) / len(results["faithfulness_scores"])
        results["avg_relevance"] = sum(results["relevance_scores"]) / len(results["relevance_scores"])
        results["avg_precision"] = sum(results["precision_scores"]) / len(results["precision_scores"])
        results["citation_accuracy_rate"] = sum(results["citation_accuracy"]) / len(results["citation_accuracy"])
    else:
        print("❌ No evaluation samples were successfully processed")
        results["avg_faithfulness"] = 0.0
        results["avg_relevance"] = 0.0
        results["avg_precision"] = 0.0
        results["citation_accuracy_rate"] = 0.0
    
    # Pass/fail determination
    results["evaluation_passed"] = (
        results["avg_faithfulness"] >= FAITHFULNESS_THRESHOLD and
        results["avg_relevance"] >= RELEVANCE_THRESHOLD and
        results["avg_precision"] >= PRECISION_THRESHOLD and
        results["citation_accuracy_rate"] >= 0.95
    )
    
    return results

def print_evaluation_report(results: dict):
    """Print a comprehensive evaluation report."""
    print("\n" + "="*60)
    print("🏛️  LEGALMIND RAG EVALUATION REPORT")
    print("="*60)
    
    print(f"📊 METRICS SUMMARY")
    print(f"   Total Samples: {results['total_samples']}")
    print(f"   Faithfulness: {results['avg_faithfulness']:.3f} (threshold: {FAITHFULNESS_THRESHOLD})")
    print(f"   Answer Relevance: {results['avg_relevance']:.3f} (threshold: {RELEVANCE_THRESHOLD})")
    print(f"   Context Precision: {results['avg_precision']:.3f} (threshold: {PRECISION_THRESHOLD})")
    print(f"   Citation Accuracy: {results['citation_accuracy_rate']:.1%}")
    
    print(f"\n🎯 QUALITY GATES")
    faithfulness_status = "✅ PASS" if results['avg_faithfulness'] >= FAITHFULNESS_THRESHOLD else "❌ FAIL"
    relevance_status = "✅ PASS" if results['avg_relevance'] >= RELEVANCE_THRESHOLD else "❌ FAIL"
    precision_status = "✅ PASS" if results['avg_precision'] >= PRECISION_THRESHOLD else "❌ FAIL"
    citation_status = "✅ PASS" if results['citation_accuracy_rate'] >= 0.95 else "❌ FAIL"
    
    print(f"   Faithfulness (No Hallucinations): {faithfulness_status}")
    print(f"   Answer Relevance: {relevance_status}")
    print(f"   Context Precision: {precision_status}")
    print(f"   Citation Validation: {citation_status}")
    
    overall_status = "✅ EVALUATION PASSED" if results['evaluation_passed'] else "❌ EVALUATION FAILED"
    print(f"\n🏆 OVERALL: {overall_status}")
    
    if results['failed_samples']:
        print(f"\n⚠️  FAILED SAMPLES ({len(results['failed_samples'])})")
        for fail in results['failed_samples'][:5]:  # Show first 5
            print(f"   Sample {fail['sample_id']}: {fail['reason']}")
        if len(results['failed_samples']) > 5:
            print(f"   ... and {len(results['failed_samples'])-5} more")
    
    print("="*60 + "\n")

def main():
    parser = argparse.ArgumentParser(description="LegalMind RAG Evaluation Suite")
    parser.add_argument("--generate-golden", action="store_true", 
                       help="Generate synthetic test dataset")
    parser.add_argument("--run-metrics", action="store_true",
                       help="Run evaluation metrics")
    parser.add_argument("--full", action="store_true", 
                       help="Run full evaluation pipeline")
    parser.add_argument("--docs-path", type=Path, default=Path("data/sample_docs"),
                       help="Path to legal documents")
    parser.add_argument("--output", type=Path, default=Path("golden_dataset.json"),
                       help="Output path for golden dataset")
    parser.add_argument("--n-questions", type=int, default=50,
                       help="Number of test questions to generate")
    
    args = parser.parse_args()
    
    if args.full or args.generate_golden:
        if not args.docs_path.exists():
            print(f"❌ Documents path {args.docs_path} does not exist")
            return
            
        generate_golden_dataset(args.docs_path, args.output, args.n_questions)
    
    if args.full or args.run_metrics:
        if not args.output.exists():
            print(f"❌ Golden dataset {args.output} does not exist. Run with --generate-golden first.")
            return
            
        results = run_evaluation_metrics(args.output)
        print_evaluation_report(results)
        
        # Save detailed results
        results_path = Path("evaluation_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"💾 Detailed results saved to {results_path}")

if __name__ == "__main__":
    main()