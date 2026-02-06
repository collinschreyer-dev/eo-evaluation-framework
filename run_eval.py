#!/usr/bin/env python3
"""
EO Policy Tracker - Evaluation Framework
Main CLI Entry Point

A modular, configurable benchmarking tool for policy-EO compliance analysis.
Supports multiple models, externalized prompts, and historical tracking.

Usage:
    python run_eval.py --help
    python run_eval.py --model claude_3_5_sonnet --phases 1,2
    python run_eval.py --dataset datasets/eo/golden_dataset.csv --prompt prompts/phase1_v2.txt
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ingestion.loader import ConfigLoader, PromptLoader, DatasetLoader
from orchestration.pipeline import EvaluationPipeline
from orchestration.llm_client import create_client
from scoring.metrics import calculate_metrics, calculate_justification_stats, generate_summary
from storage.database import ResultStorage


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="EO Policy Tracker - Evaluation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_eval.py --model gemini-2.5-pro
  python run_eval.py --model claude-sonnet-4-20250514 --phases 1,2
  python run_eval.py --phases 1,2,3 --dataset datasets/eo/golden_dataset.csv
  python run_eval.py --prompt prompts/phase1_v2.txt
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Model to use (overrides config default). E.g., gemini-2.5-pro, claude-sonnet-4-20250514'
    )
    
    parser.add_argument(
        '--models-config',
        type=str,
        default='config/models.json',
        help='Path to models configuration file'
    )
    
    parser.add_argument(
        '--prompt',
        type=str,
        default=None,
        help='Path to prompt file for Phase 1 (overrides default)'
    )
    
    parser.add_argument(
        '--prompt2',
        type=str,
        default=None,
        help='Path to prompt file for Phase 2 (overrides default)'
    )
    
    parser.add_argument(
        '--prompt3',
        type=str,
        default=None,
        help='Path to prompt file for Phase 3 (overrides default)'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default='datasets/eo/golden_dataset.csv',
        help='Path to golden dataset CSV'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='results',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--phases',
        type=str,
        default='1,2',
        help='Comma-separated phases to run (e.g., "1,2" or "1,2,3")'
    )
    
    parser.add_argument(
        '--sample',
        type=int,
        default=None,
        help='Run on N random samples instead of full dataset'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=20,
        help='Batch size for parallel processing'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=5,
        help='Number of parallel workers'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate config without running evaluation'
    )
    
    parser.add_argument(
        '--history',
        action='store_true',
        help='Show recent run history and exit'
    )
    
    parser.add_argument(
        '--compare',
        type=str,
        nargs=2,
        metavar=('RUN1', 'RUN2'),
        help='Compare two runs by ID'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    print("=" * 70)
    print("🚀 EO Policy Tracker - Evaluation Framework")
    print("=" * 70)
    
    # Initialize storage
    storage = ResultStorage(output_dir=args.output)
    
    # Handle history and compare commands
    if args.history:
        history = storage.get_run_history(limit=10)
        if not history:
            print("No run history found.")
        else:
            print(f"\n📊 Recent Runs ({len(history)}):\n")
            for run in history:
                print(f"  {run['run_id']}")
                print(f"    Model: {run['model']}")
                print(f"    Accuracy: {run['accuracy']:.2f}% | F1: {run['f1_score']:.2f}%")
                print()
        return
    
    if args.compare:
        comparison = storage.compare_runs(args.compare[0], args.compare[1])
        if "error" in comparison:
            print(f"❌ {comparison['error']}")
        else:
            print(f"\n📊 Run Comparison:")
            print(f"  Run 1: {comparison['run_1']}")
            print(f"  Run 2: {comparison['run_2']}")
            print(f"\n  Accuracy Δ:  {comparison['accuracy_diff']:+.2f}%")
            print(f"  Precision Δ: {comparison['precision_diff']:+.2f}%")
            print(f"  Recall Δ:    {comparison['recall_diff']:+.2f}%")
            print(f"  F1 Δ:        {comparison['f1_diff']:+.2f}%")
        return
    
    # Load configurations
    config_dir = Path(args.models_config).parent
    config_loader = ConfigLoader(str(config_dir))
    prompt_loader = PromptLoader("prompts")
    dataset_loader = DatasetLoader("datasets")
    
    # Get model
    model = args.model or config_loader.get_default_model()
    
    # Parse phases to run
    phases = [int(p.strip()) for p in args.phases.split(',')]
    
    print(f"\n📋 Configuration:")
    print(f"   Model:      {model}")
    print(f"   Phases:     {phases}")
    print(f"   Dataset:    {args.dataset}")
    print(f"   Output:     {args.output}")
    print(f"   Batch Size: {args.batch_size}")
    print(f"   Workers:    {args.workers}")
    
    # Load prompts
    prompt1 = prompt_loader.load_prompt(args.prompt or "phase1_classification.txt")
    prompt2 = prompt_loader.load_prompt(args.prompt2 or "phase2_reasoning.txt") if 2 in phases else None
    prompt3 = prompt_loader.load_prompt(args.prompt3 or "phase3_justification.txt") if 3 in phases else None
    
    print(f"   Prompt P1:  {args.prompt or 'phase1_classification.txt'}")
    if prompt2:
        print(f"   Prompt P2:  {args.prompt2 or 'phase2_reasoning.txt'}")
    if prompt3:
        print(f"   Prompt P3:  {args.prompt3 or 'phase3_justification.txt'}")
    
    if args.dry_run:
        print("\n✅ Dry run complete - config is valid")
        return
    
    print("=" * 70)
    
    # Load dataset
    if args.sample:
        df = dataset_loader.load_sample(args.dataset, n=args.sample)
        print(f"📊 Running on {args.sample}-record sample")
    else:
        df = dataset_loader.load_dataset(args.dataset)
    
    records = df.to_dict('records')
    
    # Initialize pipeline with appropriate client for the model
    llm_client = create_client(model)
    pipeline = EvaluationPipeline(
        llm_client=llm_client,
        batch_size=args.batch_size,
        max_workers=args.workers,
        default_model=model
    )
    
    # Run phases
    results = records
    
    if 1 in phases:
        results = pipeline.run_phase1(results, prompt1, model)
    
    if 2 in phases:
        results = pipeline.run_phase2(results, prompt2, model)
    
    if 3 in phases:
        results = pipeline.run_phase3(results, prompt3, model)
    
    # Calculate metrics
    predicted_field = "phase2_flag" if 2 in phases else "phase1_flag"
    metrics = calculate_metrics(results, predicted_field=predicted_field)
    
    justification_stats = {}
    if 3 in phases:
        justification_stats = calculate_justification_stats(results)
        metrics["avg_justification_similarity"] = justification_stats.get("avg_similarity")
    
    # Print summary
    prompt_version = Path(args.prompt or "phase1_classification.txt").stem
    summary = generate_summary(metrics, justification_stats, model, prompt_version)
    print(summary)
    
    # Save results
    run_id = storage.save_run(
        results=results,
        metrics=metrics,
        model=model,
        prompt_version=prompt_version,
        dataset=args.dataset,
        phases_run=args.phases
    )
    
    print(f"\n🎉 Evaluation complete!")
    print(f"   Run ID: {run_id}")
    print(f"   View history: python run_eval.py --history")
    print()


if __name__ == "__main__":
    main()
