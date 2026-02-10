"""
Database Module
Multi-format persistence: CSV, JSON, and SQLite for historical tracking.
"""

import os
import json
import hashlib
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
        
        self.db_path = db_path or os.environ.get('DB_PATH') or str(self.output_dir / "benchmark.db")
        self.formats = formats or ["csv", "json", "sqlite"]
        
        # Initialize SQLite if needed
        if "sqlite" in self.formats:
            self._init_database()
    
    def _init_database(self) -> None:
        """Initialize SQLite database with schema and run migrations."""
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
        
        # Prompt versions table - content-addressed prompt snapshots
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prompt_versions (
                version_id TEXT PRIMARY KEY,
                phase TEXT,
                filename TEXT,
                content TEXT,
                content_hash TEXT UNIQUE,
                created_at TEXT
            )
        """)
        
        # --- Non-destructive migrations for existing databases ---
        existing_run_cols = {row[1] for row in cursor.execute("PRAGMA table_info(runs)").fetchall()}
        existing_result_cols = {row[1] for row in cursor.execute("PRAGMA table_info(results)").fetchall()}
        
        # Runs table migrations
        run_migrations = {
            'prompt_version_id': 'TEXT',
            'dataset_hash': 'TEXT',
            'is_test_run': 'INTEGER DEFAULT 0',
        }
        for col, col_type in run_migrations.items():
            if col not in existing_run_cols:
                cursor.execute(f"ALTER TABLE runs ADD COLUMN {col} {col_type}")
        
        # Results table migrations
        result_migrations = {
            'office': 'TEXT',
            'phase1_response': 'TEXT',
            'phase2_response': 'TEXT',
            'phase3_response': 'TEXT',
        }
        for col, col_type in result_migrations.items():
            if col not in existing_result_cols:
                cursor.execute(f"ALTER TABLE results ADD COLUMN {col} {col_type}")
        
        conn.commit()
        conn.close()
        print(f"✅ SQLite database initialized: {self.db_path}")
    
    # ------------------------------------------------------------------
    # Prompt versioning
    # ------------------------------------------------------------------
    
    def save_prompt_version(self, phase: str, filename: str, content: str) -> str:
        """
        Save a prompt version. Uses content hashing for deduplication.
        Returns the version_id (existing if content already stored).
        """
        content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
        
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Check for existing version with same content
        cursor.execute("SELECT version_id FROM prompt_versions WHERE content_hash = ?", (content_hash,))
        row = cursor.fetchone()
        
        if row:
            conn.close()
            return row['version_id']
        
        # Insert new version
        version_id = f"pv_{uuid.uuid4().hex[:12]}"
        cursor.execute("""
            INSERT INTO prompt_versions (version_id, phase, filename, content, content_hash, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (version_id, phase, filename, content, content_hash, datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
        return version_id
    
    def get_prompt_version(self, version_id: str) -> Optional[Dict]:
        """Get a specific prompt version by ID."""
        if not Path(self.db_path).exists():
            return None
        
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM prompt_versions WHERE version_id = ?", (version_id,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None
    
    def get_prompt_versions(self, phase: Optional[str] = None) -> List[Dict]:
        """Get all prompt versions, optionally filtered by phase."""
        if not Path(self.db_path).exists():
            return []
        
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        if phase:
            cursor.execute("SELECT * FROM prompt_versions WHERE phase = ? ORDER BY created_at DESC", (phase,))
        else:
            cursor.execute("SELECT * FROM prompt_versions ORDER BY created_at DESC")
        
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    
    def save_run(
        self,
        results: List[Dict],
        metrics: Dict[str, Any],
        model: str,
        prompt_version: str = "",
        dataset: str = "",
        phases_run: str = "1,2",
        notes: str = "",
        prompt_version_id: str = "",
        dataset_hash: str = "",
        is_test_run: bool = False
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
                model, prompt_version, dataset, phases_run, notes,
                prompt_version_id=prompt_version_id,
                dataset_hash=dataset_hash,
                is_test_run=is_test_run
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
        notes: str,
        prompt_version_id: str = "",
        dataset_hash: str = "",
        is_test_run: bool = False
    ) -> None:
        """Save results to SQLite database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Insert run summary
        cursor.execute("""
            INSERT INTO runs (
                run_id, timestamp, model, prompt_version, dataset, phases_run,
                total_records, accuracy, precision_score, recall, f1_score,
                avg_justification_similarity, notes,
                prompt_version_id, dataset_hash, is_test_run
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id, timestamp, model, prompt_version, dataset, phases_run,
            metrics.get("total_records", 0),
            metrics.get("accuracy", 0),
            metrics.get("precision", 0),
            metrics.get("recall", 0),
            metrics.get("f1_score", 0),
            metrics.get("avg_justification_similarity"),
            notes,
            prompt_version_id or None,
            dataset_hash or None,
            1 if is_test_run else 0
        ))
        
        # Insert individual results
        for r in results:
            ground_truth = r.get('updated_flag', r.get('ground_truth', ''))
            phase2_flag = r.get('phase2_flag', r.get('phase1_flag', ''))
            is_correct = 1 if self._normalize_flag(ground_truth) == self._normalize_flag(phase2_flag) else 0
            
            cursor.execute("""
                INSERT INTO results (
                    run_id, compliance_id, ground_truth, phase1_flag, phase2_flag,
                    is_correct, similarity_score, phase2_justification,
                    office, phase1_response, phase2_response, phase3_response
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id,
                r.get('compliance_id', ''),
                ground_truth,
                r.get('phase1_flag', ''),
                phase2_flag,
                is_correct,
                r.get('similarity_score'),
                r.get('phase2_justification', '')[:5000],
                r.get('Office', ''),
                r.get('phase1_response', '')[:10000],
                r.get('phase2_response', '')[:10000],
                r.get('phase3_response', '')[:10000],
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
    
    def get_run_history(self, limit: int = 20, include_test_runs: bool = False) -> List[Dict]:
        """Get recent run history from SQLite."""
        if not Path(self.db_path).exists():
            return []
        
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        if include_test_runs:
            cursor.execute("SELECT * FROM runs ORDER BY timestamp DESC LIMIT ?", (limit,))
        else:
            cursor.execute(
                "SELECT * FROM runs WHERE COALESCE(is_test_run, 0) = 0 ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            )
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    # ------------------------------------------------------------------
    # Run results & drill-down
    # ------------------------------------------------------------------
    
    def get_run_results(self, run_id: str) -> List[Dict]:
        """Get all item-level results for a specific run."""
        if not Path(self.db_path).exists():
            return []
        
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM results WHERE run_id = ? ORDER BY compliance_id", (run_id,))
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    
    def get_run_results_by_office(self, run_id: str, office: str) -> List[Dict]:
        """Get results for a run filtered by Office/division (substring match)."""
        if not Path(self.db_path).exists():
            return []
        
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM results WHERE run_id = ? AND office LIKE ? ORDER BY compliance_id",
            (run_id, f"%{office}%")
        )
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    
    # ------------------------------------------------------------------
    # Run comparison & diffing
    # ------------------------------------------------------------------
    
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
    
    def get_flipped_items(self, run_id_a: str, run_id_b: str) -> List[Dict]:
        """
        Find items whose correctness flipped between two runs.
        Returns items with both runs' flags and correctness.
        """
        if not Path(self.db_path).exists():
            return []
        
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT
                a.compliance_id,
                a.office,
                a.ground_truth,
                a.phase2_flag AS run_a_flag,
                a.is_correct   AS run_a_correct,
                b.phase2_flag  AS run_b_flag,
                b.is_correct   AS run_b_correct
            FROM results a
            INNER JOIN results b
                ON a.compliance_id = b.compliance_id
            WHERE a.run_id = ? AND b.run_id = ?
              AND a.is_correct != b.is_correct
            ORDER BY a.compliance_id
        """, (run_id_a, run_id_b))
        
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
