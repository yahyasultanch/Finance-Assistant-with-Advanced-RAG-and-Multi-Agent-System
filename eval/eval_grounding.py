"""Evaluation for grounding quality and citation coverage."""
import re
from typing import Dict, List, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class GroundingEvaluator:
    """Evaluator for RAG grounding quality."""

    def __init__(self):
        """Initialize grounding evaluator."""
        pass

    def evaluate_summary(self, summary_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a single summary for grounding quality.
        
        Args:
            summary_result: Summary result from SummarizerAgent
            
        Returns:
            Dict with evaluation metrics
        """
        summary_text = summary_result.get("summary", "")
        citations = summary_result.get("citations", [])
        contexts = summary_result.get("contexts", [])
        
        # Extract bullet points
        bullets = self._extract_bullets(summary_text)
        
        # Calculate citation coverage
        bullets_with_citations = 0
        for bullet in bullets:
            if self._has_citation(bullet):
                bullets_with_citations += 1
        
        citation_coverage = (bullets_with_citations / len(bullets) * 100) if bullets else 0
        
        # Calculate citation count
        citation_ids = self._extract_citation_ids(summary_text)
        unique_citations = len(set(citation_ids))
        
        # Check for potential hallucinations
        hallucination_flags = self._check_hallucinations(summary_text, contexts)
        
        return {
            "total_bullets": len(bullets),
            "bullets_with_citations": bullets_with_citations,
            "citation_coverage_pct": citation_coverage,
            "unique_citations_used": unique_citations,
            "total_citations_available": len(citations),
            "hallucination_flags": hallucination_flags,
            "grounded": citation_coverage >= 90,  # Target: 90%+ grounded
            "quality_score": self._calculate_quality_score(
                citation_coverage, unique_citations, len(contexts)
            )
        }

    def evaluate_batch(
        self,
        summaries: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate multiple summaries.
        
        Args:
            summaries: Dict of query -> summary result
            
        Returns:
            Aggregated evaluation metrics
        """
        individual_results = {}
        
        for query, summary_result in summaries.items():
            individual_results[query] = self.evaluate_summary(summary_result)
        
        # Aggregate metrics
        total_bullets = sum(r["total_bullets"] for r in individual_results.values())
        total_with_citations = sum(r["bullets_with_citations"] for r in individual_results.values())
        
        overall_coverage = (total_with_citations / total_bullets * 100) if total_bullets > 0 else 0
        
        avg_quality_score = sum(r["quality_score"] for r in individual_results.values()) / len(individual_results) if individual_results else 0
        
        grounded_summaries = sum(1 for r in individual_results.values() if r["grounded"])
        grounding_rate = (grounded_summaries / len(individual_results) * 100) if individual_results else 0
        
        return {
            "individual_results": individual_results,
            "total_summaries": len(summaries),
            "total_bullets": total_bullets,
            "total_bullets_with_citations": total_with_citations,
            "overall_citation_coverage_pct": overall_coverage,
            "grounded_summaries": grounded_summaries,
            "grounding_rate_pct": grounding_rate,
            "avg_quality_score": avg_quality_score,
            "passes_target": overall_coverage >= 90  # 90% target
        }

    def manual_spot_check(
        self,
        summary_result: Dict[str, Any],
        annotations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Manual spot check with human annotations.
        
        Args:
            summary_result: Summary result
            annotations: List of annotations with bullet_idx and label (grounded/not_grounded/unsure)
            
        Returns:
            Manual evaluation metrics
        """
        summary_text = summary_result.get("summary", "")
        bullets = self._extract_bullets(summary_text)
        
        grounded_count = sum(1 for a in annotations if a.get("label") == "grounded")
        not_grounded_count = sum(1 for a in annotations if a.get("label") == "not_grounded")
        unsure_count = sum(1 for a in annotations if a.get("label") == "unsure")
        
        total_annotated = len(annotations)
        
        manual_grounding_pct = (grounded_count / total_annotated * 100) if total_annotated > 0 else 0
        
        return {
            "total_bullets": len(bullets),
            "annotated_bullets": total_annotated,
            "grounded": grounded_count,
            "not_grounded": not_grounded_count,
            "unsure": unsure_count,
            "manual_grounding_pct": manual_grounding_pct,
            "meets_target": manual_grounding_pct >= 90
        }

    def _extract_bullets(self, text: str) -> List[str]:
        """Extract bullet points from text."""
        # Match lines starting with -, *, or numbers
        bullets = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('-') or line.startswith('*') or re.match(r'^\d+\.', line):
                bullets.append(line)
        
        return bullets

    def _has_citation(self, bullet: str) -> bool:
        """Check if bullet has citation."""
        pattern = r'\[\d+:[^\]]+\]'
        return bool(re.search(pattern, bullet))

    def _extract_citation_ids(self, text: str) -> List[int]:
        """Extract citation IDs from text."""
        pattern = r'\[(\d+):[^\]]+\]'
        matches = re.findall(pattern, text)
        return [int(m) for m in matches]

    def _check_hallucinations(
        self,
        summary_text: str,
        contexts: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Check for potential hallucinations.
        
        Args:
            summary_text: Generated summary
            contexts: Retrieved contexts
            
        Returns:
            List of hallucination flags
        """
        flags = []
        
        # Check for specific numeric claims without citations
        # This is a simple heuristic
        numeric_pattern = r'\d+\.?\d*%|\$\d+\.?\d*[BMK]?|\d{4}'
        numeric_claims = re.findall(numeric_pattern, summary_text)
        
        if len(numeric_claims) > 5:
            flags.append("high_numeric_claim_count")
        
        # Check for absolute statements without citations
        absolute_words = ['always', 'never', 'all', 'none', 'every', 'must']
        for word in absolute_words:
            if word.lower() in summary_text.lower():
                # Check if near citation
                pattern = f'{word}[^\\[]*\\[\\d+:[^\\]]+\\]'
                if not re.search(pattern, summary_text, re.IGNORECASE):
                    flags.append(f"absolute_statement_without_citation:{word}")
        
        return flags

    def _calculate_quality_score(
        self,
        citation_coverage: float,
        unique_citations: int,
        total_contexts: int
    ) -> float:
        """
        Calculate overall quality score (0-100).
        
        Args:
            citation_coverage: Citation coverage percentage
            unique_citations: Number of unique citations used
            total_contexts: Total contexts available
            
        Returns:
            Quality score
        """
        # Weighted components
        coverage_weight = 0.6
        diversity_weight = 0.4
        
        # Coverage score
        coverage_score = citation_coverage
        
        # Diversity score (using multiple sources is good)
        diversity_score = min(unique_citations / max(total_contexts * 0.5, 1) * 100, 100)
        
        quality_score = (coverage_weight * coverage_score + 
                        diversity_weight * diversity_score)
        
        return round(quality_score, 2)


def generate_grounding_report(evaluation_results: Dict[str, Any]) -> str:
    """
    Generate markdown report for grounding evaluation.
    
    Args:
        evaluation_results: Results from evaluate_batch
        
    Returns:
        Markdown report
    """
    report = "# Grounding Quality Evaluation Report\n\n"
    
    report += "## Overall Metrics\n\n"
    report += f"- **Total Summaries Evaluated:** {evaluation_results['total_summaries']}\n"
    report += f"- **Total Bullet Points:** {evaluation_results['total_bullets']}\n"
    report += f"- **Bullets with Citations:** {evaluation_results['total_bullets_with_citations']}\n"
    report += f"- **Overall Citation Coverage:** {evaluation_results['overall_citation_coverage_pct']:.2f}%\n"
    report += f"- **Grounded Summaries:** {evaluation_results['grounded_summaries']} ({evaluation_results['grounding_rate_pct']:.2f}%)\n"
    report += f"- **Average Quality Score:** {evaluation_results['avg_quality_score']:.2f}/100\n"
    report += f"- **Passes Target (≥90%):** {'✅ Yes' if evaluation_results['passes_target'] else '❌ No'}\n\n"
    
    report += "## Individual Summary Results\n\n"
    report += "| Query | Bullets | With Citations | Coverage % | Quality Score |\n"
    report += "|-------|---------|----------------|------------|---------------|\n"
    
    for query, result in evaluation_results['individual_results'].items():
        report += f"| {query[:40]}... | {result['total_bullets']} | "
        report += f"{result['bullets_with_citations']} | "
        report += f"{result['citation_coverage_pct']:.1f}% | "
        report += f"{result['quality_score']:.1f} |\n"
    
    return report
