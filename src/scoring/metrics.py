"""
Metrics Module
Calculate accuracy, precision, recall, F1 score for evaluation results.
"""

from typing import Dict, List, Any
from collections import Counter


def calculate_metrics(
    results: List[Dict],
    predicted_field: str = "phase2_flag",
    ground_truth_field: str = "updated_flag"
) -> Dict[str, Any]:
    """
    Calculate classification metrics.
    
    Args:
        results: List of result dictionaries
        predicted_field: Field name containing model predictions
        ground_truth_field: Field name containing ground truth labels
        
    Returns:
        Dictionary with all metrics
    """
    if not results:
        return {
            "total": 0,
            "correct": 0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0
        }
    
    # Count predictions vs ground truth
    total = len(results)
    correct = 0
    tp = fp = fn = tn = 0
    errors = 0
    
    for r in results:
        pred = normalize_flag(r.get(predicted_field, 'Unknown'))
        truth = normalize_flag(r.get(ground_truth_field, 'Unknown'))
        
        if pred == 'Error' or pred == 'Unknown':
            errors += 1
            continue
        
        if pred == truth:
            correct += 1
        
        # Calculate confusion matrix for "Affected" class
        if truth == 'Affected' and pred == 'Affected':
            tp += 1
        elif truth == 'Not Affected' and pred == 'Affected':
            fp += 1
        elif truth == 'Affected' and pred != 'Affected':
            fn += 1
        elif truth == 'Not Affected' and pred == 'Not Affected':
            tn += 1
    
    # Calculate metrics
    accuracy = correct / total * 100 if total > 0 else 0
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "total_records": total,
        "correct": correct,
        "incorrect": total - correct - errors,
        "errors": errors,
        "accuracy": round(accuracy, 2),
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "f1_score": round(f1 / 100 * 100, 2),  # Keep as percentage
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn
    }


def normalize_flag(flag: str) -> str:
    """Normalize flag values for consistent comparison."""
    if not isinstance(flag, str):
        return 'Unknown'
    
    flag_lower = flag.lower().strip()
    
    if flag_lower in ['affected', 'true', '1', 'yes']:
        return 'Affected'
    elif flag_lower in ['not affected', 'false', '0', 'no', 'unaffected']:
        return 'Not Affected'
    elif flag_lower in ['error', 'failed']:
        return 'Error'
    else:
        return 'Unknown'


def calculate_justification_stats(
    results: List[Dict],
    score_field: str = "similarity_score"
) -> Dict[str, Any]:
    """
    Calculate statistics for justification similarity scores.
    
    Args:
        results: List of result dicts with similarity scores
        score_field: Field containing the score (0-100)
    """
    scores = [r.get(score_field) for r in results if r.get(score_field) is not None]
    
    if not scores:
        return {
            "count": 0,
            "avg_similarity": None,
            "min_similarity": None,
            "max_similarity": None
        }
    
    return {
        "count": len(scores),
        "avg_similarity": round(sum(scores) / len(scores), 2),
        "min_similarity": min(scores),
        "max_similarity": max(scores),
        "high_similarity_count": len([s for s in scores if s >= 70]),
        "low_similarity_count": len([s for s in scores if s < 50])
    }


def generate_summary(
    metrics: Dict[str, Any],
    justification_stats: Dict[str, Any] = None,
    model: str = "",
    prompt_version: str = ""
) -> str:
    """Generate a human-readable summary of evaluation results."""
    summary = [
        "=" * 60,
        "📊 EVALUATION SUMMARY",
        "=" * 60,
        f"Model: {model}" if model else "",
        f"Prompt Version: {prompt_version}" if prompt_version else "",
        "",
        f"Total Records:     {metrics.get('total_records', 0)}",
        f"Correct:           {metrics.get('correct', 0)}",
        f"Incorrect:         {metrics.get('incorrect', 0)}",
        f"Errors:            {metrics.get('errors', 0)}",
        "",
        f"Accuracy:          {metrics.get('accuracy', 0):.2f}%",
        f"Precision:         {metrics.get('precision', 0):.2f}%",
        f"Recall:            {metrics.get('recall', 0):.2f}%",
        f"F1 Score:          {metrics.get('f1_score', 0):.2f}%",
    ]
    
    if justification_stats and justification_stats.get('count', 0) > 0:
        summary.extend([
            "",
            "--- Justification Quality ---",
            f"Avg Similarity:    {justification_stats.get('avg_similarity', 0):.2f}",
            f"High Quality (≥70): {justification_stats.get('high_similarity_count', 0)}",
            f"Low Quality (<50):  {justification_stats.get('low_similarity_count', 0)}"
        ])
    
    summary.append("=" * 60)
    
    return "\n".join([s for s in summary if s != "" or s == ""])
