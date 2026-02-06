"""
Evaluation Pipeline
Three-phase evaluation orchestration: Classification, Reasoning, Justification Comparison
"""

import json
import re
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional
from tqdm import tqdm
import pandas as pd

from .llm_client import create_client


class EvaluationPipeline:
    """
    Three-phase evaluation pipeline for policy-EO compliance analysis.
    
    Phase 1: Classification (Affected/Not Affected)
    Phase 2: Final reasoning with full justification
    Phase 3: Justification comparison (model vs SME)
    """
    
    def __init__(
        self,
        llm_client = None,
        batch_size: int = 20,
        max_workers: int = 5,
        default_model: str = "gemini-2.5-pro"
    ):
        self.default_model = default_model
        self.llm = llm_client or create_client(default_model)
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.results: List[Dict] = []
    
    def run_phase1(
        self,
        records: List[Dict],
        prompt_template: str,
        model: str = "gemini-2.5-pro"
    ) -> List[Dict]:
        """
        Phase 1: Initial Classification
        
        Args:
            records: List of record dicts with 'Policy Text' and 'EO_Text'
            prompt_template: Prompt template with {policy_text} and {eo_text} placeholders
            model: Model to use
            
        Returns:
            List of records with 'phase1_flag' and 'phase1_response' added
        """
        print(f"\n🔬 Phase 1: Classification ({len(records)} records)")
        print(f"   Model: {model}")
        start_time = datetime.now()
        
        results = []
        for i in tqdm(range(0, len(records), self.batch_size), desc="Phase 1"):
            batch = records[i:i + self.batch_size]
            batch_results = self._process_batch(
                batch, prompt_template, model, self._extract_phase1_flag
            )
            results.extend(batch_results)
        
        duration = datetime.now() - start_time
        print(f"✅ Phase 1 complete in {duration}")
        return results
    
    def run_phase2(
        self,
        records: List[Dict],
        prompt_template: str,
        model: str = "gemini-2.5-pro"
    ) -> List[Dict]:
        """
        Phase 2: Final Reasoning with Full Justification
        
        Args:
            records: Records from Phase 1
            prompt_template: Prompt template with {record_json} placeholder
            model: Model to use
            
        Returns:
            List of records with 'phase2_flag', 'phase2_justification', 'phase2_response' added
        """
        print(f"\n🔬 Phase 2: Reasoning ({len(records)} records)")
        print(f"   Model: {model}")
        start_time = datetime.now()
        
        results = []
        for i in tqdm(range(0, len(records), self.batch_size), desc="Phase 2"):
            batch = records[i:i + self.batch_size]
            batch_results = self._process_batch_phase2(batch, prompt_template, model)
            results.extend(batch_results)
        
        duration = datetime.now() - start_time
        print(f"✅ Phase 2 complete in {duration}")
        return results
    
    def run_phase3(
        self,
        records: List[Dict],
        prompt_template: str,
        model: str = "gemini-2.5-pro"
    ) -> List[Dict]:
        """
        Phase 3: Justification Comparison (NEW)
        
        Compares model justification to SME justification.
        
        Args:
            records: Records from Phase 2 (must have 'phase2_justification')
            prompt_template: Prompt with {sme_justification} and {model_justification}
            model: Model to use for comparison
            
        Returns:
            List of records with 'similarity_score', 'key_differences', 'commentary' added
        """
        print(f"\n🔬 Phase 3: Justification Comparison ({len(records)} records)")
        print(f"   Model: {model}")
        start_time = datetime.now()
        
        results = []
        for i in tqdm(range(0, len(records), self.batch_size), desc="Phase 3"):
            batch = records[i:i + self.batch_size]
            batch_results = self._process_batch_phase3(batch, prompt_template, model)
            results.extend(batch_results)
        
        duration = datetime.now() - start_time
        print(f"✅ Phase 3 complete in {duration}")
        return results
    
    def _process_batch(
        self,
        batch: List[Dict],
        prompt_template: str,
        model: str,
        extractor: callable
    ) -> List[Dict]:
        """Process a batch of records in parallel."""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            for record in batch:
                prompt = prompt_template.format(
                    policy_text=record.get('Policy Text', ''),
                    eo_text=record.get('EO_Text', '')
                )
                future = executor.submit(self.llm.call, prompt, model)
                futures[future] = record
            
            for future in as_completed(futures):
                record = futures[future]
                response = future.result()
                
                if response:
                    flag = extractor(response)
                    result = {
                        **record,
                        'phase1_flag': flag,
                        'phase1_response': response,
                        'processed_at': datetime.utcnow().isoformat()
                    }
                else:
                    result = {
                        **record,
                        'phase1_flag': 'Error',
                        'phase1_response': 'LLM call failed',
                        'processed_at': datetime.utcnow().isoformat()
                    }
                results.append(result)
        
        return results
    
    def _process_batch_phase2(
        self,
        batch: List[Dict],
        prompt_template: str,
        model: str
    ) -> List[Dict]:
        """Process Phase 2 batch."""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            for record in batch:
                # Create JSON representation of record for prompt
                record_json = json.dumps({
                    'compliance_id': record.get('compliance_id', ''),
                    'Policy_ID': record.get('Policy_ID', ''),
                    'Policy_Title': record.get('Policy_Title', ''),
                    'Policy Text': record.get('Policy Text', ''),
                    'EO_Number': record.get('EO_Number', ''),
                    'EO_Name': record.get('EO_Name', ''),
                    'EO_Text': record.get('EO_Text', ''),
                    'phase1_flag': record.get('phase1_flag', ''),
                    'phase1_response': record.get('phase1_response', '')[:500]  # Truncate
                }, indent=2)
                
                prompt = prompt_template.format(record_json=record_json)
                future = executor.submit(self.llm.call, prompt, model)
                futures[future] = record
            
            for future in as_completed(futures):
                record = futures[future]
                response = future.result()
                
                if response:
                    flag, justification = self._extract_phase2_decision(response)
                    result = {
                        **record,
                        'phase2_flag': flag,
                        'phase2_justification': justification,
                        'phase2_response': response,
                        'processed_at': datetime.utcnow().isoformat()
                    }
                else:
                    result = {
                        **record,
                        'phase2_flag': 'Error',
                        'phase2_justification': '',
                        'phase2_response': 'LLM call failed',
                        'processed_at': datetime.utcnow().isoformat()
                    }
                results.append(result)
        
        return results
    
    def _process_batch_phase3(
        self,
        batch: List[Dict],
        prompt_template: str,
        model: str
    ) -> List[Dict]:
        """Process Phase 3 justification comparison batch."""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            for record in batch:
                # SME justification from golden dataset - uses 'updated_summary' column
                sme_justification = record.get('updated_summary', 
                                              record.get('SME_Justification', 
                                              'No SME justification provided'))
                model_justification = record.get('phase2_justification', '')
                
                if not model_justification:
                    # Skip records without Phase 2 output
                    results.append({
                        **record,
                        'similarity_score': None,
                        'key_differences': 'Phase 2 not completed',
                        'commentary': '',
                        'processed_at': datetime.utcnow().isoformat()
                    })
                    continue
                
                prompt = prompt_template.format(
                    sme_justification=sme_justification,
                    model_justification=model_justification
                )
                future = executor.submit(self.llm.call, prompt, model)
                futures[future] = record
            
            for future in as_completed(futures):
                record = futures[future]
                response = future.result()
                
                if response:
                    score, differences, commentary = self._extract_phase3_results(response)
                    result = {
                        **record,
                        'similarity_score': score,
                        'key_differences': differences,
                        'commentary': commentary,
                        'phase3_response': response,
                        'processed_at': datetime.utcnow().isoformat()
                    }
                else:
                    result = {
                        **record,
                        'similarity_score': None,
                        'key_differences': 'LLM call failed',
                        'commentary': '',
                        'phase3_response': '',
                        'processed_at': datetime.utcnow().isoformat()
                    }
                results.append(result)
        
        return results
    
    def _extract_phase1_flag(self, response: str) -> str:
        """Extract Affected/Not Affected from Phase 1 response."""
        text_lower = response.lower()
        
        # Check for explicit final determination
        if "final determination: [not affected]" in text_lower or \
           "final determination: not affected" in text_lower:
            return "Not Affected"
        elif "final determination: [affected]" in text_lower or \
             "final determination: affected" in text_lower:
            return "Affected"
        # Fallback
        elif "not affected" in text_lower:
            return "Not Affected"
        elif "affected" in text_lower:
            return "Affected"
        return "Unknown"
    
    def _extract_phase2_decision(self, response: str) -> tuple:
        """Extract decision and justification from Phase 2 response."""
        flag = "Unknown"
        justification = ""
        
        # Extract flag
        match_flag = re.search(
            r"Final Decision\s*:\s*\[?(Affected|Not Affected)\]?",
            response, re.IGNORECASE
        )
        if match_flag:
            flag = match_flag.group(1)
        
        # Extract justification
        match_just = re.search(
            r"Full Justification\s*:\s*(.+)",
            response, re.IGNORECASE | re.DOTALL
        )
        if match_just:
            justification = match_just.group(1).strip()
        else:
            # Fallback: look for "Explanation:"
            match_expl = re.search(
                r"Explanation\s*:\s*(.+)",
                response, re.IGNORECASE | re.DOTALL
            )
            if match_expl:
                justification = match_expl.group(1).strip()
        
        return flag, justification
    
    def _extract_phase3_results(self, response: str) -> tuple:
        """Extract similarity score and commentary from Phase 3 response."""
        score = None
        differences = ""
        commentary = ""
        
        # Extract score
        match_score = re.search(
            r"Similarity Score\s*:\s*(\d+)",
            response, re.IGNORECASE
        )
        if match_score:
            score = int(match_score.group(1))
        
        # Extract key differences
        match_diff = re.search(
            r"Key Differences\s*:\s*(.+?)(?=Commentary|$)",
            response, re.IGNORECASE | re.DOTALL
        )
        if match_diff:
            differences = match_diff.group(1).strip()
        
        # Extract commentary
        match_comm = re.search(
            r"Commentary\s*:\s*(.+)",
            response, re.IGNORECASE | re.DOTALL
        )
        if match_comm:
            commentary = match_comm.group(1).strip()
        
        return score, differences, commentary
