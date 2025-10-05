"""Evaluation for sentiment analysis accuracy."""
from datasets import load_dataset
from typing import Dict, List, Any, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
import logging

logger = logging.getLogger(__name__)


class SentimentEvaluator:
    """Evaluator for sentiment analysis accuracy."""

    def __init__(self):
        """Initialize sentiment evaluator."""
        pass

    def load_financial_phrasebank(
        self,
        split: str = "train",
        sample_size: int = 200
    ) -> List[Dict[str, Any]]:
        """
        Load Financial PhraseBank dataset for validation.
        
        Args:
            split: Dataset split
            sample_size: Number of samples to load
            
        Returns:
            List of samples with text and label
        """
        try:
            logger.info("Loading Financial PhraseBank dataset...")
            dataset = load_dataset("financial_phrasebank", "sentences_allagree", split=split)
            
            # Sample uniformly if dataset is large
            if len(dataset) > sample_size:
                indices = np.random.choice(len(dataset), sample_size, replace=False)
                dataset = dataset.select(indices)
            
            # Convert to list of dicts
            samples = []
            label_map = {0: "negative", 1: "neutral", 2: "positive"}
            
            for item in dataset:
                samples.append({
                    "text": item["sentence"],
                    "true_label": label_map[item["label"]]
                })
            
            logger.info(f"Loaded {len(samples)} samples from Financial PhraseBank")
            return samples
        except Exception as e:
            logger.error(f"Error loading Financial PhraseBank: {e}")
            return []

    def evaluate_model(
        self,
        sentiment_agent,
        test_samples: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate sentiment model on test samples.
        
        Args:
            sentiment_agent: SentimentAgent instance
            test_samples: List of test samples with 'text' and 'true_label'
            
        Returns:
            Evaluation metrics
        """
        if not test_samples:
            logger.error("No test samples provided")
            return {}
        
        # Extract texts and true labels
        texts = [sample["text"] for sample in test_samples]
        true_labels = [sample["true_label"] for sample in test_samples]
        
        # Predict
        logger.info(f"Predicting sentiment for {len(texts)} samples...")
        predictions = sentiment_agent.analyze_batch(texts)
        pred_labels = [pred["label"] for pred in predictions]
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, pred_labels)
        
        # Calculate per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels,
            pred_labels,
            labels=["positive", "neutral", "negative"],
            average=None,
            zero_division=0
        )
        
        # Macro-averaged F1
        macro_f1 = f1_score(
            true_labels,
            pred_labels,
            labels=["positive", "neutral", "negative"],
            average="macro",
            zero_division=0
        )
        
        # Weighted F1
        weighted_f1 = f1_score(
            true_labels,
            pred_labels,
            average="weighted",
            zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(
            true_labels,
            pred_labels,
            labels=["positive", "neutral", "negative"]
        )
        
        results = {
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "per_class_metrics": {
                "positive": {
                    "precision": float(precision[0]),
                    "recall": float(recall[0]),
                    "f1": float(f1[0]),
                    "support": int(support[0])
                },
                "neutral": {
                    "precision": float(precision[1]),
                    "recall": float(recall[1]),
                    "f1": float(f1[1]),
                    "support": int(support[1])
                },
                "negative": {
                    "precision": float(precision[2]),
                    "recall": float(recall[2]),
                    "f1": float(f1[2]),
                    "support": int(support[2])
                }
            },
            "confusion_matrix": cm.tolist(),
            "total_samples": len(test_samples)
        }
        
        logger.info(f"Evaluation complete - Accuracy: {accuracy:.4f}, Macro-F1: {macro_f1:.4f}")
        return results

    def compare_models(
        self,
        finbert_agent,
        finetuned_agent,
        test_samples: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compare zero-shot FinBERT vs fine-tuned model.
        
        Args:
            finbert_agent: Zero-shot FinBERT agent
            finetuned_agent: Fine-tuned agent
            test_samples: Test samples
            
        Returns:
            Comparison results
        """
        logger.info("Evaluating zero-shot FinBERT...")
        finbert_results = self.evaluate_model(finbert_agent, test_samples)
        
        logger.info("Evaluating fine-tuned model...")
        finetuned_results = self.evaluate_model(finetuned_agent, test_samples)
        
        comparison = {
            "finbert_zero_shot": finbert_results,
            "finetuned": finetuned_results,
            "improvement": {
                "accuracy": finetuned_results["accuracy"] - finbert_results["accuracy"],
                "macro_f1": finetuned_results["macro_f1"] - finbert_results["macro_f1"],
            },
            "winner": "finetuned" if finetuned_results["macro_f1"] > finbert_results["macro_f1"] else "finbert_zero_shot"
        }
        
        return comparison

    def evaluate_confidence_calibration(
        self,
        predictions: List[Dict[str, Any]],
        true_labels: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluate confidence calibration.
        
        Args:
            predictions: List of prediction dicts with 'label' and 'confidence'
            true_labels: List of true labels
            
        Returns:
            Calibration metrics
        """
        confidences = [pred["confidence"] for pred in predictions]
        pred_labels = [pred["label"] for pred in predictions]
        
        # Group predictions by confidence bins
        bins = np.linspace(0, 1, 11)  # 10 bins
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for i in range(len(bins) - 1):
            bin_mask = (np.array(confidences) >= bins[i]) & (np.array(confidences) < bins[i + 1])
            bin_count = np.sum(bin_mask)
            
            if bin_count > 0:
                bin_preds = [pred_labels[j] for j in range(len(pred_labels)) if bin_mask[j]]
                bin_trues = [true_labels[j] for j in range(len(true_labels)) if bin_mask[j]]
                bin_accuracy = accuracy_score(bin_trues, bin_preds)
                bin_avg_conf = np.mean([confidences[j] for j in range(len(confidences)) if bin_mask[j]])
                
                bin_accuracies.append(bin_accuracy)
                bin_confidences.append(bin_avg_conf)
                bin_counts.append(int(bin_count))
        
        # Expected Calibration Error (ECE)
        ece = 0
        total_samples = len(predictions)
        for i in range(len(bin_counts)):
            ece += (bin_counts[i] / total_samples) * abs(bin_accuracies[i] - bin_confidences[i])
        
        return {
            "expected_calibration_error": float(ece),
            "bin_accuracies": bin_accuracies,
            "bin_confidences": bin_confidences,
            "bin_counts": bin_counts
        }


def generate_sentiment_report(evaluation_results: Dict[str, Any]) -> str:
    """
    Generate markdown report for sentiment evaluation.
    
    Args:
        evaluation_results: Results from evaluate_model
        
    Returns:
        Markdown report
    """
    report = "# Sentiment Analysis Evaluation Report\n\n"
    
    report += "## Overall Metrics\n\n"
    report += f"- **Total Samples:** {evaluation_results['total_samples']}\n"
    report += f"- **Accuracy:** {evaluation_results['accuracy']:.4f}\n"
    report += f"- **Macro-F1:** {evaluation_results['macro_f1']:.4f}\n"
    report += f"- **Weighted-F1:** {evaluation_results['weighted_f1']:.4f}\n\n"
    
    report += "## Per-Class Metrics\n\n"
    report += "| Class | Precision | Recall | F1-Score | Support |\n"
    report += "|-------|-----------|--------|----------|----------|\n"
    
    for label in ["positive", "neutral", "negative"]:
        metrics = evaluation_results["per_class_metrics"][label]
        report += f"| {label.capitalize()} | {metrics['precision']:.4f} | "
        report += f"{metrics['recall']:.4f} | {metrics['f1']:.4f} | "
        report += f"{metrics['support']} |\n"
    
    report += "\n## Confusion Matrix\n\n"
    report += "```\n"
    report += "           Predicted\n"
    report += "           Pos  Neu  Neg\n"
    cm = evaluation_results["confusion_matrix"]
    labels = ["Positive", "Neutral ", "Negative"]
    for i, label in enumerate(labels):
        report += f"True {label}  {cm[i][0]:3d}  {cm[i][1]:3d}  {cm[i][2]:3d}\n"
    report += "```\n"
    
    return report
