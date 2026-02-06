"""
Database Module
Multi-format persistence: CSV, JSON, and SQLite for historical tracking.
"""

import os
import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd


class ResultStorage:
    """
    Multi-format result storage: CSV, JSON, and SQLite.
    
    Supports historical run tracking and comparison.
    """
    
    def __init__(
        self,
        output_dir: str = "results",
        db_path: str = None,
        formats: List[str] = None
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.db_path = db_path or self.output_dir / "benchmark.db"
        self.formats = formats or ["csv", "json", "sqlite"]
        
        # Initialize SQLite if needed
        if "sqlite" in self.formats:
            self._init_database()
    
    def _init_database(self) -> None:
        """Initialize SQLite database with schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Runs table - one row per evaluation run
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                timestamp TEXT,
                model TEXT,
                prompt_version TEXT,
                dataset TEXT,
                phases_run TEXT,
                total_records INTEGER,
                accuracy REAL,
                precision_score REAL,
                recall REAL,
                f1_score REAL,
                avg_justification_similarity REAL,
                notes TEXT
            )
        """)
        
        # Results table - one row per record per run
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                compliance_id TEXT,
                ground_truth TEXT,
                phase1_flag TEXT,
                phase2_flag TEXT,
                is_correct INTEGER,
                similarity_score REAL,
                phase2_justification TEXT,
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            )
        """)
        
        conn.commit()
        conn.close()
        print(f"✅ SQLite database initialized: {self.db_path}")
    
    def save_run(
        self,
        results: List[Dict],
        metrics: Dict[str, Any],
        model: str,
        prompt_version: str = "",
        dataset: str = "",
        phases_run: str = "1,2",
        notes: str = ""
    ) -> str:
        """
        Save a complete evaluation run to all configured formats.
        
        Returns:
            run_id: Unique identifier for this run
        """
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        timestamp = datetime.now().isoformat()
        
        print(f"\n💾 Saving results (run_id: {run_id})")
        
        # Save to each format
        if "csv" in self.formats:
            self._save_csv(results, metrics, run_id, model, prompt_version)
        
        if "json" in self.formats:
            self._save_json(results, metrics, run_id, model, prompt_version, timestamp)
        
        if "sqlite" in self.formats:
            self._save_sqlite(
                results, metrics, run_id, timestamp,
                model, prompt_version, dataset, phases_run, notes
            )
        
        return run_id
    
    def _save_csv(
        self,
        results: List[Dict],
        metrics: Dict[str, Any],
        run_id: str,
        model: str,
        prompt_version: str
    ) -> None:
        """Save results to CSV file."""
        # Results CSV
        df = pd.DataFrame(results)
        results_path = self.output_dir / f"{run_id}_results.csv"
        df.to_csv(results_path, index=False)
        print(f"   📄 CSV (results): {results_path}")
        
        # Metrics CSV (append to history)
        metrics_path = self.output_dir / "run_history.csv"
        metrics_row = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "prompt_version": prompt_version,
            **metrics
        }
        
        if metrics_path.exists():
            history_df = pd.read_csv(metrics_path)
            history_df = pd.concat([history_df, pd.DataFrame([metrics_row])], ignore_index=True)
        else:
            history_df = pd.DataFrame([metrics_row])
        
        history_df.to_csv(metrics_path, index=False)
        print(f"   📄 CSV (history): {metrics_path}")
    
    def _save_json(
        self,
        results: List[Dict],
        metrics: Dict[str, Any],
        run_id: str,
        model: str,
        prompt_version: str,
        timestamp: str
    ) -> None:
        """Save results to JSON file."""
        output = {
            "run_id": run_id,
            "timestamp": timestamp,
            "model": model,
            "prompt_version": prompt_version,
            "metrics": metrics,
            "results": results
        }
        
        json_path = self.output_dir / f"{run_id}_full.json"
        with open(json_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        print(f"   📄 JSON: {json_path}")
    
    def _save_sqlite(
        self,
        results: List[Dict],
        metrics: Dict[str, Any],
        run_id: str,
        timestamp: str,
        model: str,
        prompt_version: str,
        dataset: str,
        phases_run: str,
        notes: str
    ) -> None:
        """Save results to SQLite database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Insert run summary
        cursor.execute("""
            INSERT INTO runs (
                run_id, timestamp, model, prompt_version, dataset, phases_run,
                total_records, accuracy, precision_score, recall, f1_score,
                avg_justification_similarity, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id, timestamp, model, prompt_version, dataset, phases_run,
            metrics.get("total_records", 0),
            metrics.get("accuracy", 0),
            metrics.get("precision", 0),
            metrics.get("recall", 0),
            metrics.get("f1_score", 0),
            metrics.get("avg_justification_similarity"),
            notes
        ))
        
        # Insert individual results
        for r in results:
            ground_truth = r.get('updated_flag', r.get('ground_truth', ''))
            phase2_flag = r.get('phase2_flag', r.get('phase1_flag', ''))
            is_correct = 1 if self._normalize_flag(ground_truth) == self._normalize_flag(phase2_flag) else 0
            
            cursor.execute("""
                INSERT INTO results (
                    run_id, compliance_id, ground_truth, phase1_flag, phase2_flag,
                    is_correct, similarity_score, phase2_justification
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id,
                r.get('compliance_id', ''),
                ground_truth,
                r.get('phase1_flag', ''),
                phase2_flag,
                is_correct,
                r.get('similarity_score'),
                r.get('phase2_justification', '')[:5000]  # Limit size
            ))
        
        conn.commit()
        conn.close()
        print(f"   🗄️  SQLite: {self.db_path}")
    
    def _normalize_flag(self, flag: str) -> str:
        """Normalize flag for comparison."""
        if not isinstance(flag, str):
            return 'Unknown'
        flag_lower = flag.lower().strip()
        if flag_lower in ['affected', 'true', '1']:
            return 'Affected'
        elif flag_lower in ['not affected', 'false', '0']:
            return 'Not Affected'
        return 'Unknown'
    
    def get_run_history(self, limit: int = 10) -> List[Dict]:
        """Get recent run history from SQLite."""
        if not Path(self.db_path).exists():
            return []
        
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM runs ORDER BY timestamp DESC LIMIT ?
        """, (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def compare_runs(self, run_id_1: str, run_id_2: str) -> Dict[str, Any]:
        """Compare metrics between two runs."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        runs = {}
        for run_id in [run_id_1, run_id_2]:
            cursor.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,))
            row = cursor.fetchone()
            if row:
                runs[run_id] = dict(row)
        
        conn.close()
        
        if len(runs) != 2:
            return {"error": "One or both runs not found"}
        
        r1, r2 = runs[run_id_1], runs[run_id_2]
        
        return {
            "run_1": run_id_1,
            "run_2": run_id_2,
            "accuracy_diff": r2.get("accuracy", 0) - r1.get("accuracy", 0),
            "precision_diff": r2.get("precision_score", 0) - r1.get("precision_score", 0),
            "recall_diff": r2.get("recall", 0) - r1.get("recall", 0),
            "f1_diff": r2.get("f1_score", 0) - r1.get("f1_score", 0),
            "run_1_details": r1,
            "run_2_details": r2
        }
