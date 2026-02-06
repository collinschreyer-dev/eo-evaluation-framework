"""
Ingestion Module - Loader
Handles loading of datasets, configurations, and prompt templates.
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional


class ConfigLoader:
    """Load and manage configuration files."""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
    
    def load_models(self) -> Dict[str, Any]:
        """Load model configuration from models.json"""
        models_path = self.config_dir / "models.json"
        if not models_path.exists():
            raise FileNotFoundError(f"Models config not found: {models_path}")
        
        with open(models_path, 'r') as f:
            return json.load(f)
    
    def load_settings(self) -> Dict[str, Any]:
        """Load general settings from settings.json"""
        settings_path = self.config_dir / "settings.json"
        if not settings_path.exists():
            # Return defaults if no settings file
            return {
                "batch_size": 20,
                "max_workers": 5,
                "timeout_seconds": 120,
                "database_path": "results/benchmark.db",
                "output_formats": ["csv", "json", "sqlite"]
            }
        
        with open(settings_path, 'r') as f:
            return json.load(f)
    
    def get_enabled_models(self) -> List[Dict[str, Any]]:
        """Get list of enabled models only."""
        config = self.load_models()
        return [m for m in config.get("models", []) if m.get("enabled", True)]
    
    def get_default_model(self) -> str:
        """Get the default model ID."""
        config = self.load_models()
        return config.get("default_model", "claude_3_5_sonnet")


class PromptLoader:
    """Load and manage prompt templates."""
    
    def __init__(self, prompts_dir: str = "prompts"):
        self.prompts_dir = Path(prompts_dir)
        self._cache: Dict[str, str] = {}
    
    def load_prompt(self, prompt_name: str) -> str:
        """
        Load a prompt template by name.
        
        Args:
            prompt_name: Either a filename (e.g., 'phase1_classification.txt')
                        or just the name (e.g., 'phase1_classification')
        """
        # Add .txt extension if not present
        if not prompt_name.endswith('.txt'):
            prompt_name = f"{prompt_name}.txt"
        
        # Check cache
        if prompt_name in self._cache:
            return self._cache[prompt_name]
        
        prompt_path = self.prompts_dir / prompt_name
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt not found: {prompt_path}")
        
        with open(prompt_path, 'r') as f:
            content = f.read()
        
        self._cache[prompt_name] = content
        return content
    
    def render_prompt(self, prompt_name: str, **kwargs) -> str:
        """
        Load and render a prompt template with variable substitution.
        
        Args:
            prompt_name: Name of the prompt template
            **kwargs: Variables to substitute in the template
        """
        template = self.load_prompt(prompt_name)
        return template.format(**kwargs)
    
    def list_prompts(self) -> List[str]:
        """List all available prompt files."""
        if not self.prompts_dir.exists():
            return []
        return [f.name for f in self.prompts_dir.glob("*.txt")]


class DatasetLoader:
    """Load and validate golden datasets."""
    
    REQUIRED_COLUMNS = [
        'compliance_id', 'Policy Text', 'EO_Text', 'updated_flag'
    ]
    
    def __init__(self, datasets_dir: str = "datasets"):
        self.datasets_dir = Path(datasets_dir)
    
    def load_dataset(self, dataset_path: str) -> pd.DataFrame:
        """
        Load a dataset from CSV.
        
        Args:
            dataset_path: Path to the CSV file (absolute or relative to datasets_dir)
        """
        path = Path(dataset_path)
        if not path.is_absolute():
            path = self.datasets_dir / path
        
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        
        df = pd.read_csv(path)
        self._validate_dataset(df)
        
        print(f"✅ Loaded {len(df)} records from {path.name}")
        return df
    
    def _validate_dataset(self, df: pd.DataFrame) -> None:
        """Validate that dataset has required columns."""
        missing = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"Dataset missing required columns: {missing}")
    
    def load_sample(self, dataset_path: str, n: int = 5) -> pd.DataFrame:
        """Load a random sample from the dataset."""
        df = self.load_dataset(dataset_path)
        return df.sample(min(n, len(df)))


# Convenience function for quick loading
def load_all(
    config_dir: str = "config",
    prompts_dir: str = "prompts",
    datasets_dir: str = "datasets"
) -> tuple:
    """
    Load all configurations at once.
    
    Returns:
        Tuple of (ConfigLoader, PromptLoader, DatasetLoader)
    """
    return (
        ConfigLoader(config_dir),
        PromptLoader(prompts_dir),
        DatasetLoader(datasets_dir)
    )
