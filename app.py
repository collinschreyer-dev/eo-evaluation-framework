"""
EO Evaluation Framework - Streamlit Frontend
A minimal, focused UI for running evaluations and editing prompts.

Run with: streamlit run app.py
"""

import streamlit as st
import json
import sys
import os
import base64
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ingestion.loader import ConfigLoader, PromptLoader, DatasetLoader
from orchestration.pipeline import EvaluationPipeline
from orchestration.llm_client import create_client
from scoring.metrics import calculate_metrics, calculate_justification_stats, generate_summary
from storage.database import ResultStorage

# ============================================================
# Page Config
# ============================================================
st.set_page_config(
    page_title="EO Evaluation Framework",
    page_icon="EO_ICON.jpeg",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# Authentication Configuration
# ============================================================
AUTH_USERNAME = "admin"
AUTH_PASSWORD = "EOevaluator2026!"

def check_auth():
    """Returns True if the user is authenticated."""
    return st.session_state.get("authenticated", False)

def login_form():
    """Display login form and handle authentication."""
    st.markdown("""
    <style>
        .login-container {
            max-width: 400px;
            margin: 100px auto;
            padding: 40px;
            background: linear-gradient(135deg, #1e3a5f 0%, #0d1b2a 100%);
            border-radius: 16px;
            border: 1px solid #2a4a6b;
        }
        .login-title {
            text-align: center;
            color: #fff;
            font-size: 1.8rem;
            margin-bottom: 30px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("## 🔐 EO Evaluation Framework")
        st.markdown("Please log in to continue.")
        
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login", use_container_width=True, type="primary")
            
            if submitted:
                if username == AUTH_USERNAME and password == AUTH_PASSWORD:
                    st.session_state["authenticated"] = True
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials. Please try again.")

# Check authentication before proceeding
if not check_auth():
    login_form()
    st.stop()

# ============================================================
# Dark Theme CSS
# ============================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #000;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .main-header img {
        height: 80px;
        width: auto;
    }
    .sub-header {
        font-size: 1rem;
        color: #333;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #0d1b2a 100%);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid #2a4a6b;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #4ade80;
    }
    .metric-label {
        color: #94a3b8;
        font-size: 0.85rem;
        text-transform: uppercase;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #2563eb, #7c3aed);
    }
    .run-button>button {
        background: linear-gradient(90deg, #10b981, #059669) !important;
        font-size: 1.1rem;
    }
    .prompt-editor textarea {
        font-family: 'Monaco', 'Menlo', monospace;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Initialize Session State
# ============================================================
if 'results' not in st.session_state:
    st.session_state.results = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'running' not in st.session_state:
    st.session_state.running = False

# ============================================================
# Loaders
# ============================================================
@st.cache_resource
def get_loaders():
    return {
        'config': ConfigLoader('config'),
        'prompts': PromptLoader('prompts'),
        'dataset': DatasetLoader('datasets')
    }

@st.cache_resource
def get_storage():
    # Check for Render persistent disk, fall back to local results/
    if os.path.exists('/data') and os.access('/data', os.W_OK):
        output_dir = '/data'
    else:
        output_dir = 'results'
    return ResultStorage(output_dir=output_dir)

loaders = get_loaders()
storage = get_storage()

# ============================================================
# Sidebar - Configuration
# ============================================================
with st.sidebar:
    # Logout button at top
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state["authenticated"] = False
        st.rerun()
    
    st.divider()
    st.markdown("### ⚙️ Configuration")
    
    # Model Selection
    models_config = loaders['config'].load_models()
    enabled_models = [m for m in models_config.get('models', []) if m.get('enabled')]
    model_options = {m['name']: m['id'] for m in enabled_models}
    
    selected_model_name = st.selectbox(
        "Model",
        options=list(model_options.keys()),
        help="Select which LLM to use for evaluation"
    )
    selected_model = model_options[selected_model_name]
    
    st.divider()
    
    # Pipeline Configuration
    st.markdown("### 🔧 Pipeline Configuration")
    
    # Get available prompts for each phase
    prompts_dir = Path('prompts')
    
    # Phase 1 prompts
    phase1_prompts = ['phase1_classification.txt'] + sorted([f.name for f in prompts_dir.glob('phase1_v*.txt')])
    phase1_prompts_display = ['Default'] + [f.stem.replace('phase1_', '').upper() for f in prompts_dir.glob('phase1_v*.txt')]
    
    # Phase 2 prompts
    phase2_prompts = ['phase2_reasoning.txt'] + sorted([f.name for f in prompts_dir.glob('phase2_v*.txt')])
    phase2_prompts_display = ['Default'] + [f.stem.replace('phase2_', '').upper() for f in prompts_dir.glob('phase2_v*.txt')]
    
    # Phase 3 prompts
    phase3_prompts = ['phase3_justification.txt'] + sorted([f.name for f in prompts_dir.glob('phase3_v*.txt')])
    
    # Phase 1 Configuration
    run_phase1 = st.checkbox("Phase 1: Classification", value=True)
    if run_phase1:
        selected_phase1_prompt = st.selectbox(
            "Phase 1 Prompt",
            options=phase1_prompts,
            format_func=lambda x: "Original" if x == 'phase1_classification.txt' else x.replace('.txt', '').replace('phase1_', 'Phase1 ').upper(),
            key="phase1_prompt_select"
        )
    else:
        selected_phase1_prompt = phase1_prompts[0]
    
    # Phase 2 Configuration
    run_phase2 = st.checkbox("Phase 2: Reasoning", value=True)
    if run_phase2:
        selected_phase2_prompt = st.selectbox(
            "Phase 2 Prompt",
            options=phase2_prompts,
            format_func=lambda x: "Original" if x == 'phase2_reasoning.txt' else x.replace('.txt', '').replace('phase2_', 'Phase2 ').upper(),
            key="phase2_prompt_select"
        )
    else:
        selected_phase2_prompt = phase2_prompts[0]
    
    # Phase 3 Configuration
    run_phase3 = st.checkbox("Phase 3: Justification Comparison", value=True)
    if run_phase3:
        selected_phase3_prompt = st.selectbox(
            "Phase 3 Prompt",
            options=phase3_prompts,
            format_func=lambda x: "Original" if x == 'phase3_justification.txt' else x.replace('.txt', '').replace('phase3_', 'Phase3 ').upper(),
            key="phase3_prompt_select"
        )
    else:
        selected_phase3_prompt = phase3_prompts[0]
    
    # Store selected prompts in session state for pipeline use
    st.session_state['pipeline_prompts'] = {
        'phase1': selected_phase1_prompt,
        'phase2': selected_phase2_prompt,
        'phase3': selected_phase3_prompt
    }
    
    st.divider()
    
    # Sample Size
    st.markdown("### 📁 Dataset")
    sample_size = st.slider(
        "Sample Size",
        min_value=1,
        max_value=152,
        value=10,
        help="Number of records to evaluate (full dataset: 152)"
    )
    
    if sample_size == 152:
        st.info("Running full dataset")
    else:
        st.caption(f"Testing on {sample_size} records")
    
    st.divider()
    
    # Run History
    st.markdown("### 📜 Recent Runs")
    history = storage.get_run_history(limit=5)
    if history:
        for run in history:
            acc = run.get('accuracy', 0) or 0
            st.caption(f"**{run['model']}** - {acc:.1f}%")
    else:
        st.caption("No runs yet")

# ============================================================
# Main Content
# ============================================================
st.markdown('<p class="main-header">EO Policy Tracker Evaluation Framework</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Edit prompts • Run evaluations • Track results</p>', unsafe_allow_html=True)

# Tabs
tab_run, tab_prompts, tab_analyze, tab_results = st.tabs(["▶️ Run Evaluation", "📝 Edit Prompts", "🔬 Analyze & Debug", "📊 Results History"])

# ============================================================
# TAB 1: Run Evaluation
# ============================================================
with tab_run:
    # Quick Start + Run Button in same row
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### Quick Start")
        st.markdown("""
        1. **Select model** in sidebar → 2. **Choose phases** → 3. **Set sample size** → 4. **Click Run**
        """)
        
        # Phases summary
        phases = []
        if run_phase1: phases.append("1")
        if run_phase2: phases.append("2")
        if run_phase3: phases.append("3")
        
        st.info(f"**Ready:** {selected_model_name} • Phases {', '.join(phases)} • {sample_size} records")
    
    with col2:
        st.markdown("### ")  # Spacer
        run_clicked = st.button("🚀 Run Evaluation", use_container_width=True, type="primary")
        is_test_run = st.checkbox("🧪 Mark as test run", value=False, key="is_test_run_cb")
    
    # ==========================================
    # Scrollable Console Container
    # ==========================================
    with st.expander("📟 **Live Console** (click to expand)", expanded=True):
        # Custom CSS for scrollable console
        st.markdown("""
        <style>
            .console-box {
                background: #0d1117;
                border: 1px solid #30363d;
                border-radius: 8px;
                padding: 12px;
                max-height: 400px;
                overflow-y: auto;
                font-family: 'Monaco', 'Menlo', 'Consolas', monospace;
                font-size: 0.85rem;
                color: #c9d1d9;
            }
            .console-line {
                margin: 4px 0;
                line-height: 1.4;
            }
            .console-time {
                color: #8b949e;
            }
            .console-success { color: #3fb950; }
            .console-error { color: #f85149; }
            .console-phase { color: #58a6ff; font-weight: bold; }
        </style>
        """, unsafe_allow_html=True)
        
        console_placeholder = st.empty()
        
        if run_clicked and not st.session_state.running:
            st.session_state.running = True
            log_lines = []
            
            def log(msg, emoji=""):
                """Add a line to the console log"""
                timestamp = datetime.now().strftime("%H:%M:%S")
                log_lines.append(f'<div class="console-line"><span class="console-time">{timestamp}</span> {emoji} {msg}</div>')
                # Render all lines in scrollable div
                html = f'<div class="console-box">{"".join(log_lines)}</div>'
                console_placeholder.markdown(html, unsafe_allow_html=True)
            
            try:
                log("Starting evaluation...", "🚀")
                
                # Load data
                log("Loading dataset...", "📁")
                df = loaders['dataset'].load_sample('eo/golden_dataset.csv', n=sample_size)
                records = df.to_dict('records')
                log(f"Loaded <b>{len(records)}</b> records", "✅")
                
                # Load prompts - use selected versions from sidebar
                log("Loading prompts...", "📝")
                pipeline_prompts = st.session_state.get('pipeline_prompts', {})
                prompt1_file = pipeline_prompts.get('phase1', 'phase1_classification.txt')
                prompt2_file = pipeline_prompts.get('phase2', 'phase2_reasoning.txt')
                prompt3_file = pipeline_prompts.get('phase3', 'phase3_justification.txt')
                
                prompt1 = loaders['prompts'].load_prompt(prompt1_file)
                prompt2 = loaders['prompts'].load_prompt(prompt2_file)
                prompt3 = loaders['prompts'].load_prompt(prompt3_file)
                
                log(f"Phase 1: <b>{prompt1_file}</b>", "📝")
                log(f"Phase 2: <b>{prompt2_file}</b>", "📝")
                if run_phase3:
                    log(f"Phase 3: <b>{prompt3_file}</b>", "📝")
                
                # Initialize client
                log(f"Initializing <b>{selected_model_name}</b>...", "🤖")
                llm_client = create_client(selected_model)
                log(f"{selected_model_name} ready", "✅")
                
                results = records
                
                # ==========================================
                # PHASE 1: Classification
                # ==========================================
                if run_phase1:
                    log('<span class="console-phase">━━━ PHASE 1: Classification ━━━</span>', "🔬")
                    
                    phase1_results = []
                    for i, record in enumerate(results):
                        cid = record.get('compliance_id', 'unknown')[:25]
                        log(f"Record {i+1}/{len(results)}: <code>{cid}</code>", "⏳")
                        
                        # Run phase 1 for single record
                        prompt = prompt1.format(
                            policy_text=record.get('Policy Text', '')[:2000],
                            eo_text=record.get('EO_Text', '')[:2000]
                        )
                        response = llm_client.call(prompt, selected_model)
                        
                        # Extract flag
                        flag = "Unknown"
                        if response:
                            resp_lower = response.lower()
                            if "not affected" in resp_lower:
                                flag = "Not Affected"
                            elif "affected" in resp_lower:
                                flag = "Affected"
                        
                        result = {
                            **record,
                            'phase1_flag': flag,
                            'phase1_response': response or 'Error'
                        }
                        phase1_results.append(result)
                        
                        # Log result with color indicator
                        color = "console-success" if flag == "Affected" else "console-error"
                        log(f'  └→ <span class="{color}"><b>{flag}</b></span>', "")
                    
                    results = phase1_results
                    log(f"Phase 1 complete: {len(results)} processed", "✅")
                
                # ==========================================
                # PHASE 2: Reasoning
                # ==========================================
                if run_phase2:
                    log('<span class="console-phase">━━━ PHASE 2: Reasoning ━━━</span>', "🔬")
                    
                    phase2_results = []
                    for i, record in enumerate(results):
                        p1_flag = record.get('phase1_flag', 'Unknown')
                        log(f"Record {i+1}/{len(results)}: P1={p1_flag} → P2", "⏳")
                        
                        # Build phase 2 prompt
                        import json as json_lib
                        record_json = json_lib.dumps({
                            'compliance_id': record.get('compliance_id', ''),
                            'Policy_Title': record.get('Policy_Title', ''),
                            'Policy Text': record.get('Policy Text', '')[:1000],
                            'EO_Name': record.get('EO_Name', ''),
                            'EO_Text': record.get('EO_Text', '')[:1000],
                            'phase1_flag': p1_flag
                        }, indent=2)
                        
                        prompt = prompt2.format(record_json=record_json)
                        response = llm_client.call(prompt, selected_model)
                        
                        # Extract decision and justification
                        flag = "Unknown"
                        justification = ""
                        if response:
                            resp_lower = response.lower()
                            if "not affected" in resp_lower:
                                flag = "Not Affected"
                            elif "affected" in resp_lower:
                                flag = "Affected"
                            
                            # Better justification extraction - get full reasoning
                            # Try to extract after common headers, or use full response
                            for header in ["justification:", "reasoning:", "analysis:", "explanation:"]:
                                if header in resp_lower:
                                    idx = resp_lower.find(header)
                                    justification = response[idx + len(header):].strip()[:800]
                                    break
                            
                            # If no header found, use full response minus the flag line
                            if not justification:
                                justification = response.strip()[:800]
                        
                        result = {
                            **record,
                            'phase2_flag': flag,
                            'phase2_justification': justification,
                            'phase2_response': response or 'Error'
                        }
                        phase2_results.append(result)
                        
                        # Compare with ground truth
                        ground_truth = record.get('updated_flag', 'Unknown')
                        is_match = flag.lower().strip() == ground_truth.lower().strip()
                        match_icon = "✅" if is_match else "❌"
                        color = "console-success" if is_match else "console-error"
                        log(f'  └→ P2: <span class="{color}"><b>{flag}</b></span> | Truth: {ground_truth} {match_icon}', "")
                    
                    results = phase2_results
                    log(f"Phase 2 complete", "✅")
                
                # ==========================================
                # PHASE 3: Justification Comparison
                # ==========================================
                if run_phase3:
                    log('<span class="console-phase">━━━ PHASE 3: Justification Comparison ━━━</span>', "🔬")
                    
                    phase3_results = []
                    for i, record in enumerate(results):
                        sme_just = record.get('updated_summary', '')
                        model_just = record.get('phase2_justification', '')
                        
                        log(f"Record {i+1}: Comparing justifications...", "⏳")
                        
                        # Show what we're comparing (truncated for display)
                        sme_preview = sme_just[:80] + "..." if len(sme_just) > 80 else sme_just or "(empty)"
                        model_preview = model_just[:80] + "..." if len(model_just) > 80 else model_just or "(empty)"
                        log(f'  <span style="color:#8b949e">SME: "{sme_preview}"</span>', "")
                        log(f'  <span style="color:#8b949e">Model: "{model_preview}"</span>', "")
                        
                        # Build the comparison prompt
                        prompt = prompt3.format(
                            sme_justification=sme_just[:1000] if sme_just else "No SME justification provided",
                            model_justification=model_just[:1000] if model_just else "No model justification provided"
                        )
                        response = llm_client.call(prompt, selected_model)
                        
                        # Extract similarity score - look for number after "Score:" or just any number
                        score = None
                        key_differences = ""
                        commentary = ""
                        
                        if response:
                            import re
                            # Try to find "Similarity Score: XX"
                            score_match = re.search(r'similarity\s*score[:\s]*(\d+)', response, re.IGNORECASE)
                            if score_match:
                                score = int(score_match.group(1))
                            else:
                                # Fallback: find first number in response
                                num_match = re.search(r'\b(\d{1,3})\b', response)
                                if num_match:
                                    score = int(num_match.group(1))
                            
                            if score and score > 100:
                                score = 100
                            
                            # Extract key differences
                            if "key differences:" in response.lower():
                                diff_start = response.lower().find("key differences:")
                                diff_text = response[diff_start + 16:diff_start + 300]
                                key_differences = diff_text.split("Commentary:")[0].strip() if "Commentary:" in diff_text else diff_text.strip()
                            
                            # Extract commentary
                            if "commentary:" in response.lower():
                                comm_start = response.lower().find("commentary:")
                                commentary = response[comm_start + 11:comm_start + 300].strip()
                        
                        result = {
                            **record,
                            'similarity_score': score,
                            'key_differences': key_differences,
                            'phase3_commentary': commentary,
                            'phase3_response': response or 'Error'
                        }
                        phase3_results.append(result)
                        
                        # Show score with color
                        if score is not None:
                            color = "console-success" if score >= 70 else "#f0ad4e" if score >= 50 else "console-error"
                            log(f'  └→ Similarity: <span style="color:{color}"><b>{score}%</b></span>', "")
                            if commentary:
                                log(f'  <span style="color:#8b949e;font-size:0.8em">💬 {commentary[:100]}...</span>', "")
                        else:
                            log(f'  └→ <span class="console-error">Could not extract score</span>', "")
                    
                    results = phase3_results
                    log(f"Phase 3 complete", "✅")
                
                # ==========================================
                # Calculate & Save
                # ==========================================
                log("Calculating metrics...", "📊")
                
                predicted_field = "phase2_flag" if run_phase2 else "phase1_flag"
                metrics = calculate_metrics(results, predicted_field=predicted_field)
                
                if run_phase3:
                    just_stats = calculate_justification_stats(results)
                    metrics['avg_justification_similarity'] = just_stats.get('avg_similarity')
                
                log(f'<span class="console-success"><b>Accuracy: {metrics.get("accuracy", 0):.1f}%</b></span>', "🎯")
                log(f'<span class="console-success"><b>F1 Score: {metrics.get("f1_score", 0):.1f}%</b></span>', "📈")
                
                if run_phase3 and metrics.get('avg_justification_similarity'):
                    log(f'<b>Avg Justification Similarity: {metrics["avg_justification_similarity"]:.1f}%</b>', "📝")
                
                # Save — snapshot prompts and compute dataset hash
                import hashlib as _hl
                prompt_version_id = ""
                dataset_hash = ""
                try:
                    # Snapshot the primary prompt used
                    primary_phase = 'phase2' if run_phase2 else 'phase1'
                    primary_file = prompt2_file if run_phase2 else prompt1_file
                    primary_content = prompt2 if run_phase2 else prompt1
                    prompt_version_id = storage.save_prompt_version(primary_phase, primary_file, primary_content)
                    # Dataset hash
                    dataset_hash = _hl.sha256(df.to_csv(index=False).encode()).hexdigest()[:16]
                except Exception:
                    pass  # Non-critical
                
                run_id = storage.save_run(
                    results=results,
                    metrics=metrics,
                    model=selected_model,
                    prompt_version=prompt2_file if run_phase2 else prompt1_file,
                    phases_run=",".join(phases),
                    prompt_version_id=prompt_version_id,
                    dataset_hash=dataset_hash,
                    is_test_run=is_test_run
                )
                
                st.session_state.results = results
                st.session_state.metrics = metrics
                st.session_state.run_id = run_id
                
                log(f"<b>Run saved: {run_id}</b>", "💾")
                log('<span class="console-success"><b>🎉 Evaluation complete!</b></span>', "")
                
            except Exception as e:
                log(f'<span class="console-error"><b>Error: {str(e)}</b></span>', "❌")
                import traceback
                log(f'<span style="color:#8b949e;font-size:0.8em">{traceback.format_exc()[:300]}</span>', "")
            finally:
                st.session_state.running = False
    
    # ==========================================
    # Results Display
    # ==========================================
    if st.session_state.metrics:
        st.divider()
        st.markdown("### 📊 Results")
        
        m = st.session_state.metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{m.get('accuracy', 0):.1f}%")
        with col2:
            st.metric("Precision", f"{m.get('precision', 0):.1f}%")
        with col3:
            st.metric("Recall", f"{m.get('recall', 0):.1f}%")
        with col4:
            st.metric("F1 Score", f"{m.get('f1_score', 0):.1f}%")
        
        # Show results table
        if st.session_state.results:
            st.markdown("#### Detailed Results")
            df_results = pd.DataFrame(st.session_state.results)
            display_cols = ['compliance_id', 'updated_flag', 'phase1_flag']
            if run_phase2:
                display_cols.append('phase2_flag')
            if run_phase3 and 'similarity_score' in df_results.columns:
                display_cols.extend(['similarity_score', 'key_differences'])
            
            available_cols = [c for c in display_cols if c in df_results.columns]
            if available_cols:
                st.dataframe(df_results[available_cols], use_container_width=True)

# ============================================================
# TAB 2: Edit Prompts
# ============================================================
with tab_prompts:
    st.markdown("### 📝 Prompt Editor")
    st.caption("Edit prompts directly here. Changes are saved immediately.")
    
    # Prompt selector
    prompt_files = list(Path('prompts').glob('*.txt'))
    prompt_names = [p.name for p in prompt_files]
    
    selected_prompt = st.selectbox("Select Prompt", prompt_names)
    
    if selected_prompt:
        prompt_path = Path('prompts') / selected_prompt
        current_content = prompt_path.read_text()
        
        # Editor
        new_content = st.text_area(
            "Prompt Content",
            value=current_content,
            height=400,
            key=f"prompt_{selected_prompt}"
        )
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("💾 Save Changes", type="primary"):
                prompt_path.write_text(new_content)
                st.success("Saved!")
                st.cache_resource.clear()
        
        with col2:
            if st.button("↩️ Revert"):
                st.rerun()
        
        # Create new prompt
        st.divider()
        st.markdown("### Create New Prompt")
        
        new_name = st.text_input("New prompt filename (e.g., phase1_v2.txt)")
        if st.button("➕ Create") and new_name:
            if not new_name.endswith('.txt'):
                new_name += '.txt'
            new_path = Path('prompts') / new_name
            new_path.write_text("# New Prompt\n\nEdit this prompt...")
            st.success(f"Created {new_name}")
            st.rerun()

# ============================================================
# TAB 3: Analyze & Debug
# ============================================================
with tab_analyze:
    st.markdown("### 🔬 Analyze Results & Debug Prompts")
    st.caption("Deep dive into results, find disagreements, and test new prompts on specific records.")
    
    # ==========================================
    # Mode Selection: Previous Run vs Fresh Records
    # ==========================================
    analyze_mode = st.radio(
        "Data Source",
        ["📂 From Previous Run", "📊 Fresh from Golden Dataset"],
        horizontal=True,
        key="analyze_mode"
    )
    
    st.divider()
    
    # ==========================================
    # MODE 1: Fresh from Golden Dataset
    # ==========================================
    if analyze_mode == "📊 Fresh from Golden Dataset":
        st.markdown("### 📊 Test on Golden Dataset Records")
        st.caption("Select specific records from the golden dataset to test prompts on.")
        
        # Load the golden dataset
        try:
            golden_df = pd.read_csv('datasets/eo/golden_dataset.csv')
            
            # Filter controls
            col1, col2, col3 = st.columns(3)
            
            with col1:
                flag_filter = st.selectbox(
                    "Filter by Ground Truth",
                    ["All", "Affected Only", "Not Affected Only"],
                    key="golden_flag_filter"
                )
            
            with col2:
                st.metric("Total Records", len(golden_df))
            
            with col3:
                affected_count = len(golden_df[golden_df['updated_flag'].str.lower().str.strip() == 'affected'])
                st.metric("Affected / Not Affected", f"{affected_count} / {len(golden_df) - affected_count}")
            
            # Apply filter
            if flag_filter == "Affected Only":
                filtered_df = golden_df[golden_df['updated_flag'].str.lower().str.strip() == 'affected']
            elif flag_filter == "Not Affected Only":
                filtered_df = golden_df[golden_df['updated_flag'].str.lower().str.strip() == 'not affected']
            else:
                filtered_df = golden_df
            
            st.info(f"Showing {len(filtered_df)} records")
            
            # Record selector with preview
            if len(filtered_df) > 0:
                record_options = filtered_df['compliance_id'].tolist()
                
                selected_golden_id = st.selectbox(
                    "Select Record",
                    options=record_options,
                    format_func=lambda x: f"{x} {'🟢' if filtered_df[filtered_df['compliance_id']==x]['updated_flag'].values[0].lower().strip() == 'affected' else '🔴'}",
                    key="golden_record_select"
                )
                
                # Get selected record
                record_data = filtered_df[filtered_df['compliance_id'] == selected_golden_id].iloc[0].to_dict()
                
                # Show record details
                st.divider()
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 📑 Ground Truth")
                    st.markdown(f"**Flag:** `{record_data.get('updated_flag', 'N/A')}`")
                    st.markdown(f"**Policy:** {record_data.get('Policy_Title', 'N/A')}")
                    st.text_area(
                        "SME Justification",
                        record_data.get('updated_summary', 'No SME summary'),
                        height=120,
                        disabled=True,
                        key="golden_sme"
                    )
                
                with col2:
                    st.markdown("#### 📜 Record Content")
                    st.markdown(f"**EO:** {record_data.get('EO_Name', 'N/A')}")
                    with st.expander("View Policy Text"):
                        st.text(str(record_data.get('Policy Text', ''))[:1000])
                    with st.expander("View EO Text"):
                        st.text(str(record_data.get('EO_Text', ''))[:1000])
                
                # ==========================================
                # Quick Prompt Tester (for fresh records)
                # ==========================================
                st.divider()
                st.markdown("### 🧪 Test Prompt on This Record")
                
                test_phase_golden = st.selectbox(
                    "Phase to Test",
                    ["Phase 2 (Reasoning)", "Phase 1 (Classification)", "Custom"],
                    key="golden_phase_select"
                )
                
                # Load prompt - check for suggested prompt first
                if 'golden_suggested_prompt' in st.session_state and st.session_state.get('golden_suggested_prompt'):
                    golden_prompt = st.session_state['golden_suggested_prompt']
                    st.info("📋 Using AI-suggested prompt from iteration. Modify as needed.")
                    # Clear after loading
                    del st.session_state['golden_suggested_prompt']
                elif test_phase_golden == "Phase 2 (Reasoning)":
                    golden_prompt = loaders['prompts'].load_prompt('phase2_reasoning.txt')
                elif test_phase_golden == "Phase 1 (Classification)":
                    golden_prompt = loaders['prompts'].load_prompt('phase1_classification.txt')
                else:
                    golden_prompt = "You are analyzing a policy-EO pair.\n\nPolicy: {policy_text}\n\nExecutive Order: {eo_text}\n\nDetermine if this policy is AFFECTED or NOT AFFECTED by the EO."
                
                golden_test_prompt = st.text_area(
                    "Prompt Template",
                    value=golden_prompt,
                    height=200,
                    key="golden_prompt_editor"
                )
                
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    golden_test_model = st.selectbox(
                        "Model",
                        list(model_options.keys()),
                        key="golden_model_select"
                    )
                
                with col2:
                    if st.button("🧪 Test Prompt", type="primary", key="golden_test_btn"):
                        with st.spinner("Testing..."):
                            try:
                                # Build prompt
                                if test_phase_golden == "Phase 2 (Reasoning)":
                                    import json as json_lib
                                    record_json = json_lib.dumps({
                                        'compliance_id': record_data.get('compliance_id', ''),
                                        'Policy_Title': record_data.get('Policy_Title', ''),
                                        'Policy Text': str(record_data.get('Policy Text', ''))[:1000],
                                        'EO_Name': record_data.get('EO_Name', ''),
                                        'EO_Text': str(record_data.get('EO_Text', ''))[:1000],
                                        'phase1_flag': 'Unknown'
                                    }, indent=2)
                                    filled_prompt = golden_test_prompt.format(record_json=record_json)
                                else:
                                    filled_prompt = golden_test_prompt.format(
                                        policy_text=str(record_data.get('Policy Text', ''))[:2000],
                                        eo_text=str(record_data.get('EO_Text', ''))[:2000]
                                    )
                                
                                # Call LLM
                                test_client = create_client(model_options[golden_test_model])
                                response = test_client.call(filled_prompt, model_options[golden_test_model])
                                
                                st.session_state['golden_test_response'] = response
                                st.session_state['golden_ground_truth'] = record_data.get('updated_flag', '')
                                
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
                
                # Show response
                if 'golden_test_response' in st.session_state and st.session_state.get('golden_test_response'):
                    st.markdown("#### 📤 Response")
                    response = st.session_state['golden_test_response']
                    ground_truth = st.session_state.get('golden_ground_truth', '')
                    
                    # Extract flag
                    resp_lower = response.lower()
                    if "not affected" in resp_lower:
                        extracted_flag = "Not Affected"
                    elif "affected" in resp_lower:
                        extracted_flag = "Affected"
                    else:
                        extracted_flag = "Unknown"
                    
                    is_correct = extracted_flag.lower().strip() == ground_truth.lower().strip()
                    
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        if is_correct:
                            st.success(f"✅ {extracted_flag}")
                            st.caption("Correct!")
                        else:
                            st.error(f"❌ {extracted_flag}")
                            st.caption(f"Expected: {ground_truth}")
                    
                    with col2:
                        with st.expander("📖 View Full LLM Response (Markdown)", expanded=True):
                            st.markdown(response)
                    
                    # Save option
                    if st.button("💾 Save as New Prompt Version", key="golden_save_btn"):
                        phase_num = "2" if "Phase 2" in test_phase_golden else "1"
                        existing = list(Path('prompts').glob(f'phase{phase_num}_v*.txt'))
                        ver = len(existing) + 2
                        Path('prompts', f'phase{phase_num}_v{ver}.txt').write_text(golden_test_prompt)
                        st.success(f"Saved as phase{phase_num}_v{ver}.txt!")
                    
                    # ==========================================
                    # Auto-Iterate Prompt Refinement for Golden Dataset
                    # ==========================================
                    if not is_correct:
                        st.divider()
                        st.markdown("### 🔄 Auto-Iterate Prompt Refinement")
                        st.caption(f"The model returned **{extracted_flag}** but ground truth is **{ground_truth}**. Let AI suggest an improved prompt.")
                        
                        # Track iteration count
                        if 'golden_iteration_count' not in st.session_state:
                            st.session_state['golden_iteration_count'] = 0
                        
                        iter_col1, iter_col2 = st.columns([1, 3])
                        with iter_col1:
                            st.metric("Iteration", st.session_state.get('golden_iteration_count', 0))
                        
                        with iter_col2:
                            if st.button("🔄 Suggest Improved Prompt", type="primary", key="golden_auto_iterate_btn"):
                                with st.spinner("Analyzing failure and generating improved prompt..."):
                                    try:
                                        current_prompt = st.session_state.get('golden_last_prompt', golden_test_prompt)
                                        
                                        iterate_prompt = f"""You are an expert prompt engineer. A prompt was tested on a policy-EO compliance analysis task but produced the WRONG answer.

**Task Details:**
- Ground Truth Flag: {ground_truth}
- Model's Wrong Answer: {extracted_flag}
- Iteration: #{st.session_state.get('golden_iteration_count', 0) + 1}

**The Model's Full Response:**
{response[:2000]}

**Current Prompt Being Used:**
```
{current_prompt[:2000]}
```

**Your Task:**
Create an IMPROVED prompt that will make the model output "{ground_truth}" for this specific case.

Consider:
1. Is the model misinterpreting what constitutes "Affected" vs "Not Affected"?
2. Does the prompt need stronger guidance on policy impact criteria?
3. Should the prompt include examples or emphasize key criteria?

Provide the complete IMPROVED prompt (ready to use) with the same placeholders.

IMPORTANT: Output ONLY the new prompt, no explanations. Start directly with the prompt text."""

                                        iterate_client = create_client(model_options[test_model_golden])
                                        suggested = iterate_client.call(iterate_prompt, model_options[test_model_golden])
                                        
                                        if suggested:
                                            import re
                                            suggested = re.sub(r'^```\w*\n?', '', suggested.strip())
                                            suggested = re.sub(r'\n?```$', '', suggested)
                                            
                                            st.session_state['golden_suggested_prompt'] = suggested
                                            st.session_state['golden_iteration_count'] = st.session_state.get('golden_iteration_count', 0) + 1
                                            st.session_state['golden_last_prompt'] = suggested
                                            
                                            st.success("✅ Improved prompt generated! It will be loaded in the editor.")
                                            st.rerun()
                                            
                                    except Exception as e:
                                        st.error(f"Error generating improved prompt: {str(e)}")
                        
                        if st.session_state.get('golden_iteration_count', 0) > 0:
                            st.info(f"💡 Iteration #{st.session_state.get('golden_iteration_count', 0)} - New prompt loaded. Click 'Test Prompt' to test it.")
                            if st.button("🔄 Reset", key="golden_reset_iter"):
                                st.session_state['golden_iteration_count'] = 0
                                st.session_state.pop('golden_last_prompt', None)
                                st.session_state.pop('golden_suggested_prompt', None)
                                st.rerun()
            else:
                st.warning("No records match the filter.")
                
        except Exception as e:
            st.error(f"Error loading golden dataset: {str(e)}")
    
    # ==========================================
    # MODE 2: From Previous Run
    # ==========================================
    else:
        # Load Run Data - use same output_dir as storage for consistency
        if os.path.exists('/data') and os.access('/data', os.W_OK):
            results_dir = Path('/data')
        else:
            results_dir = Path('results')
        json_files = list(results_dir.glob('*_full.json')) if results_dir.exists() else []
        
        if not json_files:
            st.info("No runs available. Run an evaluation first to analyze results here.")
        else:
            # Sort by modification time (newest first)
            json_files = sorted(json_files, key=lambda x: x.stat().st_mtime, reverse=True)
            run_options = {f.stem.replace('_full', ''): f for f in json_files}
            
            selected_run = st.selectbox(
                "Select Run to Analyze",
                options=list(run_options.keys()),
                format_func=lambda x: f"{x[:30]}..." if len(x) > 30 else x
            )
            
            if selected_run:
                # Load the run data
                run_file = run_options[selected_run]
                import json as json_lib
                with open(run_file) as f:
                    run_data = json_lib.load(f)
                
                results_list = run_data.get('results', [])
                metrics_info = run_data.get('metrics', {})
                
                if results_list:
                    df_run = pd.DataFrame(results_list)
                    
                    # ==========================================
                    # Filter Controls
                    # ==========================================
                    st.divider()
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        filter_mode = st.radio(
                            "Show",
                            ["All Records", "Disagreements Only", "Agreements Only"],
                            horizontal=True
                        )
                    
                    with col2:
                        st.metric("Total Records", len(df_run))
                    
                    with col3:
                        st.metric("Accuracy", f"{metrics_info.get('accuracy', 0):.1f}%")
                    
                    # Apply filter
                    if 'phase2_flag' in df_run.columns:
                        pred_col = 'phase2_flag'
                    elif 'phase1_flag' in df_run.columns:
                        pred_col = 'phase1_flag'
                    else:
                        pred_col = None
                    
                    if pred_col and 'updated_flag' in df_run.columns:
                        df_run['is_match'] = df_run.apply(
                            lambda r: str(r.get(pred_col, '')).lower().strip() == str(r.get('updated_flag', '')).lower().strip(),
                            axis=1
                        )
                        
                        if filter_mode == "Disagreements Only":
                            df_filtered = df_run[~df_run['is_match']]
                            st.warning(f"Showing {len(df_filtered)} disagreements")
                        elif filter_mode == "Agreements Only":
                            df_filtered = df_run[df_run['is_match']]
                            st.success(f"Showing {len(df_filtered)} agreements")
                        else:
                            df_filtered = df_run
                    else:
                        df_filtered = df_run
                    
                    # ==========================================
                    # Record Selector
                    # ==========================================
                    st.divider()
                    st.markdown("### 📋 Select Record to Analyze")
                    
                    if len(df_filtered) > 0:
                        record_options = df_filtered['compliance_id'].tolist()
                        selected_record_id = st.selectbox(
                            "Record",
                            options=record_options,
                            format_func=lambda x: f"{x} {'❌' if not df_filtered[df_filtered['compliance_id']==x]['is_match'].values[0] else '✅'}" if 'is_match' in df_filtered.columns else x
                        )
                        
                        # Get selected record data
                        record_data = df_filtered[df_filtered['compliance_id'] == selected_record_id].iloc[0].to_dict()
                        
                        # ==========================================
                        # Record Details View
                        # ==========================================
                        st.divider()
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### 📑 Ground Truth")
                            st.markdown(f"**Flag:** `{record_data.get('updated_flag', 'N/A')}`")
                            sme_summary = record_data.get('updated_summary', 'No SME summary available')
                            with st.expander("📖 View Full SME Justification", expanded=True):
                                st.markdown(sme_summary)
                        
                        with col2:
                            st.markdown("#### 🤖 AI Response")
                            p2_flag = record_data.get('phase2_flag', record_data.get('phase1_flag', 'N/A'))
                            is_correct = str(p2_flag).lower().strip() == str(record_data.get('updated_flag', '')).lower().strip()
                            st.markdown(f"**Flag:** `{p2_flag}` {'✅' if is_correct else '❌'}")
                            # Use phase2_response for FULL response, fallback to phase2_justification
                            model_just = record_data.get('phase2_response', record_data.get('phase2_justification', record_data.get('phase1_response', 'No model response')))
                            with st.expander("📖 View Full AI Response", expanded=True):
                                st.markdown(model_just)
                        
                        # Phase 3 Comparison (if available)
                        if 'similarity_score' in record_data and record_data.get('similarity_score'):
                            st.markdown("#### 📊 Phase 3 Comparison")
                            cols = st.columns([1, 3])
                            with cols[0]:
                                score = record_data.get('similarity_score', 0)
                                st.metric("Similarity", f"{score}%")
                            with cols[1]:
                                # Use phase3_response for FULL response
                                phase3_text = record_data.get('phase3_response', record_data.get('phase3_commentary', 'N/A'))
                                with st.expander("📖 View Full LLM Commentary", expanded=True):
                                    st.markdown(phase3_text)
                        
                        # ==========================================
                        # AI Prompt Suggestion
                        # ==========================================
                        if not is_correct:
                            st.divider()
                            st.markdown("### 🪄 AI Prompt Suggestion")
                            st.caption("Use AI to analyze this disagreement and suggest an improved prompt.")
                            
                            if st.button("🪄 Generate Suggested Prompt", type="secondary"):
                                with st.spinner("Analyzing disagreement and generating improved prompt..."):
                                    try:
                                        # Build a meta-prompt to suggest improvements
                                        current_p2_prompt = loaders['prompts'].load_prompt('phase2_reasoning.txt')
                                        
                                        meta_prompt = f"""You are an expert at prompt engineering for LLM-based policy compliance analysis.

A policy-EO pair was analyzed but the model gave the WRONG answer.

**Ground Truth (Correct):**
- Flag: {record_data.get('updated_flag', 'Unknown')}
- SME Justification: {record_data.get('updated_summary', 'N/A')[:500]}

**Model's Wrong Answer:**
- Flag: {record_data.get('phase2_flag', 'Unknown')}
- Model Justification: {record_data.get('phase2_justification', 'N/A')[:500]}

**Current Prompt Being Used:**
```
{current_p2_prompt[:1500]}
```

**Your Task:**
Analyze why the model got this wrong and suggest an IMPROVED version of the prompt that would help the model get the correct answer. Consider:
1. Was the model missing key context?
2. Was the instruction unclear about how to determine "Affected" vs "Not Affected"?
3. Should the prompt emphasize certain aspects more?

Provide:
1. A brief analysis of what went wrong (2-3 sentences)
2. The complete IMPROVED prompt (ready to use, with same placeholders like {{record_json}})

Format your response as:
## Analysis
[Your analysis here]

## Improved Prompt
```
[The complete improved prompt here]
```"""

                                        # Call LLM
                                        suggest_client = create_client(selected_model)
                                        suggestion_response = suggest_client.call(meta_prompt, selected_model)
                                        
                                        if suggestion_response:
                                            st.session_state['prompt_suggestion'] = suggestion_response
                                            
                                    except Exception as e:
                                        st.error(f"Error generating suggestion: {str(e)}")
                            
                            # Display suggestion if available
                            if 'prompt_suggestion' in st.session_state and st.session_state.get('prompt_suggestion'):
                                suggestion = st.session_state['prompt_suggestion']
                                
                                # Parse out the analysis and prompt
                                if "## Analysis" in suggestion and "## Improved Prompt" in suggestion:
                                    parts = suggestion.split("## Improved Prompt")
                                    analysis_part = parts[0].replace("## Analysis", "").strip()
                                    prompt_part = parts[1].strip()
                                    
                                    # Clean up prompt part (remove markdown code blocks)
                                    import re
                                    prompt_clean = re.sub(r'^```\w*\n?', '', prompt_part)
                                    prompt_clean = re.sub(r'\n?```$', '', prompt_clean)
                                    
                                    st.success("**Analysis:**")
                                    st.markdown(analysis_part)
                                    
                                    st.markdown("**Suggested Prompt:**")
                                    st.code(prompt_clean[:1500], language="text")
                                    
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        if st.button("📋 Copy to Prompt Tester"):
                                            st.session_state['suggested_prompt_for_tester'] = prompt_clean
                                            st.success("Copied! Scroll down to the Quick Prompt Tester.")
                                    with col2:
                                        if st.button("💾 Save as phase2_v2.txt"):
                                            existing_versions = list(Path('prompts').glob('phase2_v*.txt'))
                                            next_ver = len(existing_versions) + 2
                                            new_name = f"phase2_v{next_ver}.txt"
                                            Path('prompts', new_name).write_text(prompt_clean)
                                            st.success(f"Saved as `{new_name}`!")
                                else:
                                    # Just show raw response
                                    st.text_area("Suggestion", suggestion[:2000], height=300, disabled=True)
                        
                        # ==========================================
                        # Quick Prompt Tester
                        # ==========================================
                        st.divider()
                        st.markdown("### 🧪 Quick Prompt Tester")
                        st.caption("Test a modified prompt on this specific record to see if you can get the correct result.")
                        
                        # Phase selector
                        test_phase = st.selectbox(
                            "Phase to Test",
                            ["Phase 2 (Reasoning)", "Phase 1 (Classification)", "Custom Phase"],
                            key="test_phase_selector"
                        )
                        
                        # Initialize prompt in session state if not exists or if phase changed
                        phase_key = f"prompt_for_{test_phase.replace(' ', '_')}"
                        
                        # Check if we have a suggested prompt to inject
                        if 'suggested_prompt_for_tester' in st.session_state and st.session_state.get('suggested_prompt_for_tester'):
                            # Store the suggested prompt as the current working prompt
                            st.session_state['working_test_prompt'] = st.session_state['suggested_prompt_for_tester']
                            del st.session_state['suggested_prompt_for_tester']
                            st.info("📋 AI-suggested prompt loaded! Modify as needed and click Test Prompt.")
                        
                        # If no working prompt yet, load from phase
                        if 'working_test_prompt' not in st.session_state:
                            if test_phase == "Phase 2 (Reasoning)":
                                st.session_state['working_test_prompt'] = loaders['prompts'].load_prompt('phase2_reasoning.txt')
                            elif test_phase == "Phase 1 (Classification)":
                                st.session_state['working_test_prompt'] = loaders['prompts'].load_prompt('phase1_classification.txt')
                            else:
                                st.session_state['working_test_prompt'] = "# Custom Phase Prompt\n\nYou are analyzing a policy-EO pair.\n\nPolicy: {policy_text}\n\nExecutive Order: {eo_text}\n\nProvide your analysis:"
                        
                        # Button to reset to original prompt
                        if st.button("🔄 Reset to Original Prompt", key="reset_prompt_btn"):
                            if test_phase == "Phase 2 (Reasoning)":
                                st.session_state['working_test_prompt'] = loaders['prompts'].load_prompt('phase2_reasoning.txt')
                            elif test_phase == "Phase 1 (Classification)":
                                st.session_state['working_test_prompt'] = loaders['prompts'].load_prompt('phase1_classification.txt')
                            else:
                                st.session_state['working_test_prompt'] = "# Custom Phase Prompt\n\nYou are analyzing a policy-EO pair.\n\nPolicy: {policy_text}\n\nExecutive Order: {eo_text}\n\nProvide your analysis:"
                            st.rerun()
                        
                        # Prompt editor - use a callback to save changes
                        def update_working_prompt():
                            st.session_state['working_test_prompt'] = st.session_state['test_prompt_editor_input']
                        
                        test_prompt = st.text_area(
                            "Prompt Template (modify to test)",
                            value=st.session_state.get('working_test_prompt', ''),
                            height=250,
                            key="test_prompt_editor_input",
                            on_change=update_working_prompt
                        )
                        
                        col1, col2, col3 = st.columns([1, 1, 2])
                        
                        with col1:
                            test_model = st.selectbox(
                                "Model",
                                list(model_options.keys()),
                                key="test_model_select"
                            )
                        
                        with col2:
                            if st.button("🧪 Test Prompt", type="primary"):
                                with st.spinner("Testing prompt..."):
                                    try:
                                        # Build the prompt with record data
                                        if test_phase == "Phase 2 (Reasoning)":
                                            import json as json_lib
                                            record_json = json_lib.dumps({
                                                'compliance_id': record_data.get('compliance_id', ''),
                                                'Policy_Title': record_data.get('Policy_Title', ''),
                                                'Policy Text': record_data.get('Policy Text', '')[:1000],
                                                'EO_Name': record_data.get('EO_Name', ''),
                                                'EO_Text': record_data.get('EO_Text', '')[:1000],
                                                'phase1_flag': record_data.get('phase1_flag', 'Unknown')
                                            }, indent=2)
                                            filled_prompt = test_prompt.format(record_json=record_json)
                                        else:
                                            filled_prompt = test_prompt.format(
                                                policy_text=record_data.get('Policy Text', '')[:2000],
                                                eo_text=record_data.get('EO_Text', '')[:2000]
                                            )
                                        
                                        # Call LLM
                                        test_client = create_client(model_options[test_model])
                                        response = test_client.call(filled_prompt, model_options[test_model])
                                        
                                        st.session_state['test_response'] = response
                                        st.session_state['test_ground_truth'] = record_data.get('updated_flag', '')
                                        
                                    except Exception as e:
                                        st.error(f"Error: {str(e)}")
                        
                        with col3:
                            if st.button("💾 Save as New Prompt"):
                                if test_phase == "Custom Phase":
                                    # Count existing custom phases
                                    custom_prompts = list(Path('prompts').glob('custom_phase*.txt'))
                                    next_num = len(custom_prompts) + 1
                                    new_name = f"custom_phase{next_num}.txt"
                                else:
                                    phase_num = "2" if "Phase 2" in test_phase else "1"
                                    existing_versions = list(Path('prompts').glob(f'phase{phase_num}_v*.txt'))
                                    next_ver = len(existing_versions) + 2
                                    new_name = f"phase{phase_num}_v{next_ver}.txt"
                                
                                Path('prompts', new_name).write_text(test_prompt)
                                st.success(f"Saved as `{new_name}`")
                                st.cache_resource.clear()
                        
                        # Show test response
                        if 'test_response' in st.session_state and st.session_state.get('test_response'):
                            st.markdown("#### 📤 LLM Response")
                            response = st.session_state['test_response']
                            ground_truth = st.session_state.get('test_ground_truth', '')
                            
                            # Extract flag from response
                            resp_lower = response.lower()
                            if "not affected" in resp_lower:
                                extracted_flag = "Not Affected"
                            elif "affected" in resp_lower:
                                extracted_flag = "Affected"
                            else:
                                extracted_flag = "Unknown"
                            
                            is_now_correct = extracted_flag.lower().strip() == ground_truth.lower().strip()
                            
                            col1, col2 = st.columns([1, 3])
                            with col1:
                                if is_now_correct:
                                    st.success(f"✅ {extracted_flag}")
                                    st.caption("Matches ground truth!")
                                else:
                                    st.error(f"❌ {extracted_flag}")
                                    st.caption(f"Expected: {ground_truth}")
                            
                            with col2:
                                with st.expander("📖 View Full LLM Response (Markdown)", expanded=True):
                                    st.markdown(response)
                            
                            # ==========================================
                            # Similarity Analysis (when correct)
                            # ==========================================
                            if is_now_correct:
                                st.divider()
                                st.markdown("### 📊 Similarity Analysis with SME Justification")
                                st.caption("Compare the AI's reasoning with the Subject Matter Expert's justification.")
                                
                                # Get SME justification from the current record
                                sme_justification = record_data.get('updated_summary', record_data.get('Compliance_Summary', ''))
                                
                                if sme_justification:
                                    if st.button("🔍 Run Similarity Analysis", type="primary", key="run_similarity_btn"):
                                        with st.spinner("Analyzing similarity between AI response and SME justification..."):
                                            try:
                                                similarity_prompt = f"""You are an expert at comparing policy analysis justifications.

Compare the following two justifications for the SAME policy-EO classification decision.

## AI Model Justification:
{response[:3000]}

## Subject Matter Expert (SME) Justification:
{sme_justification[:3000]}

## Your Task:
1. **Similarity Score**: Rate how similar the reasoning is on a scale of 0-100%.
2. **Key Agreements**: What key points do both agree on?
3. **Key Differences**: Where does the reasoning differ?
4. **Quality Assessment**: Is the AI's reasoning as thorough as the SME's?

Provide your response in this format:
- **Similarity Score:** [0-100]%
- **Key Agreements:** [bullet points]
- **Key Differences:** [bullet points]  
- **Quality Assessment:** [paragraph]"""

                                                # Use the same model as test
                                                sim_client = create_client(model_options[test_model])
                                                similarity_response = sim_client.call(similarity_prompt, model_options[test_model])
                                                
                                                if similarity_response:
                                                    st.session_state['similarity_analysis'] = similarity_response
                                                    
                                            except Exception as e:
                                                st.error(f"Error running similarity analysis: {str(e)}")
                                    
                                    # Show similarity analysis if available
                                    if 'similarity_analysis' in st.session_state and st.session_state.get('similarity_analysis'):
                                        st.markdown("---")
                                        st.markdown(st.session_state['similarity_analysis'])
                                        
                                        if st.button("🔄 Clear Analysis", key="clear_sim_btn"):
                                            del st.session_state['similarity_analysis']
                                            st.rerun()
                                else:
                                    st.info("No SME justification available for this record to compare against.")
                            
                            # ==========================================
                            # Auto-Iterate Prompt Refinement
                            # ==========================================
                            if not is_now_correct:
                                st.divider()
                                st.markdown("### 🔄 Auto-Iterate Prompt Refinement")
                                st.caption(f"The model returned **{extracted_flag}** but ground truth is **{ground_truth}**. Let AI suggest an improved prompt to get the correct answer.")
                                
                                # Track iteration count
                                if 'iteration_count' not in st.session_state:
                                    st.session_state['iteration_count'] = 0
                                
                                iteration_col1, iteration_col2 = st.columns([1, 3])
                                with iteration_col1:
                                    st.metric("Iteration", st.session_state.get('iteration_count', 0))
                                
                                with iteration_col2:
                                    if st.button("🔄 Suggest Improved Prompt", type="primary", key="auto_iterate_btn"):
                                        with st.spinner("Analyzing failure and generating improved prompt..."):
                                            try:
                                                # Get record data for context
                                                current_test_prompt = st.session_state.get('last_test_prompt', test_prompt)
                                                
                                                iterate_meta_prompt = f"""You are an expert prompt engineer. A prompt was tested on a policy-EO compliance analysis task but produced the WRONG answer.

**Task Details:**
- Ground Truth Flag: {ground_truth}
- Model's Wrong Answer: {extracted_flag}
- Iteration: #{st.session_state.get('iteration_count', 0) + 1}

**The Model's Full Response:**
{response[:2000]}

**Current Prompt Being Used:**
```
{current_test_prompt[:2000]}
```

**Your Task:**
Analyze why the model got this wrong and create an IMPROVED prompt that will make the model output "{ground_truth}" for this specific case.

Consider:
1. Is the model misinterpreting key criteria for "Affected" vs "Not Affected"?
2. Does the prompt need stronger guidance on what constitutes a policy impact?
3. Should the prompt include specific examples or counter-examples?
4. Are there keywords or phrases that should be emphasized?

Provide the complete IMPROVED prompt (ready to use) with the same placeholders ({{record_json}} or {{policy_text}}/{{eo_text}}).

IMPORTANT: Output ONLY the new prompt, no explanations. Start directly with the prompt text."""

                                                # Call LLM
                                                iterate_client = create_client(model_options[test_model])
                                                suggested_prompt = iterate_client.call(iterate_meta_prompt, model_options[test_model])
                                                
                                                if suggested_prompt:
                                                    # Clean up any markdown code blocks
                                                    import re
                                                    suggested_prompt = re.sub(r'^```\w*\n?', '', suggested_prompt.strip())
                                                    suggested_prompt = re.sub(r'\n?```$', '', suggested_prompt)
                                                    
                                                    st.session_state['suggested_prompt_for_tester'] = suggested_prompt
                                                    st.session_state['iteration_count'] = st.session_state.get('iteration_count', 0) + 1
                                                    st.session_state['last_test_prompt'] = suggested_prompt
                                                    
                                                    st.success("✅ Improved prompt generated! Scroll up to test it.")
                                                    st.rerun()
                                                    
                                            except Exception as e:
                                                st.error(f"Error generating improved prompt: {str(e)}")
                                
                                # Show iteration history if we've iterated
                                if st.session_state.get('iteration_count', 0) > 0:
                                    st.info(f"💡 Tip: Click 'Test Prompt' above to test the new suggested prompt (iteration #{st.session_state.get('iteration_count', 0)})")
                                    
                                    if st.button("🔄 Reset Iteration Counter", key="reset_iter"):
                                        st.session_state['iteration_count'] = 0
                                        st.session_state.pop('last_test_prompt', None)
                                        st.rerun()
                    else:
                        st.warning("No records match the current filter.")

# ============================================================
# TAB 4: Results History
# ============================================================
with tab_results:
    st.markdown("### 📊 Run History")
    
    # --- Controls ---
    ctrl_col1, ctrl_col2 = st.columns([3, 1])
    with ctrl_col2:
        show_test_runs = st.checkbox("Show test runs", value=False, key="show_test_runs_cb")
    
    history = storage.get_run_history(limit=30, include_test_runs=show_test_runs)
    
    if not history:
        st.info("No runs yet. Start an evaluation to see results here.")
    else:
        df_history = pd.DataFrame(history)
        
        # Display columns
        display_cols = ['run_id', 'model', 'accuracy', 'precision_score', 'recall', 'f1_score',
                        'total_records', 'prompt_version', 'is_test_run', 'timestamp']
        available_cols = [c for c in display_cols if c in df_history.columns]
        
        # Format test-run flag
        if 'is_test_run' in df_history.columns:
            df_history['is_test_run'] = df_history['is_test_run'].apply(lambda x: '🧪' if x == 1 else '')
        
        st.dataframe(df_history[available_cols], use_container_width=True)
        
        run_ids = df_history['run_id'].tolist()
        
        # ==========================================
        # Run Drill-Down
        # ==========================================
        st.divider()
        st.markdown("### 🔍 Run Drill-Down")
        
        selected_drill_run = st.selectbox(
            "Select a run to inspect",
            run_ids,
            key="drill_run_select"
        )
        
        if selected_drill_run:
            # Get run metadata
            run_meta = df_history[df_history['run_id'] == selected_drill_run].iloc[0].to_dict()
            
            # --- Provenance Card ---
            st.markdown("#### 📋 Run Provenance")
            prov_c1, prov_c2, prov_c3, prov_c4 = st.columns(4)
            with prov_c1:
                st.metric("Model", run_meta.get('model', 'N/A'))
            with prov_c2:
                st.metric("Prompt", run_meta.get('prompt_version', 'N/A'))
            with prov_c3:
                st.metric("Phases", run_meta.get('phases_run', 'N/A'))
            with prov_c4:
                st.metric("Records", run_meta.get('total_records', 0))
            
            # Show prompt text if version_id exists
            pv_id = run_meta.get('prompt_version_id')
            if pv_id:
                pv_data = storage.get_prompt_version(pv_id)
                if pv_data:
                    with st.expander(f"📝 Prompt Text ({pv_data.get('filename', 'unknown')})"):
                        st.code(pv_data.get('content', ''), language='text')
            
            # --- Metrics Summary ---
            st.markdown("#### 📊 Metrics")
            m_c1, m_c2, m_c3, m_c4, m_c5 = st.columns(5)
            with m_c1:
                st.metric("Accuracy", f"{run_meta.get('accuracy', 0):.1f}%")
            with m_c2:
                st.metric("Precision", f"{run_meta.get('precision_score', 0):.1f}%")
            with m_c3:
                st.metric("Recall", f"{run_meta.get('recall', 0):.1f}%")
            with m_c4:
                st.metric("F1", f"{run_meta.get('f1_score', 0):.1f}%")
            with m_c5:
                sim = run_meta.get('avg_justification_similarity')
                st.metric("Avg Similarity", f"{sim:.1f}%" if sim else "N/A")
            
            # --- Item Determinations ---
            st.markdown("#### 📋 Item Determinations")
            
            run_results = storage.get_run_results(selected_drill_run)
            
            if run_results:
                df_items = pd.DataFrame(run_results)
                
                # Division filter
                if 'office' in df_items.columns:
                    offices = sorted(df_items['office'].dropna().unique().tolist())
                    offices = [o for o in offices if o]  # Remove empties
                    if offices:
                        selected_offices = st.multiselect(
                            "🏢 Filter by Division (Office)",
                            options=offices,
                            default=[],
                            key="drill_office_filter"
                        )
                        if selected_offices:
                            df_items = df_items[df_items['office'].apply(
                                lambda x: any(off in str(x) for off in selected_offices)
                            )]
                
                # Display table
                item_display_cols = ['compliance_id', 'office', 'ground_truth', 'phase1_flag',
                                     'phase2_flag', 'is_correct', 'similarity_score']
                item_available = [c for c in item_display_cols if c in df_items.columns]
                
                # Color-code correctness
                if 'is_correct' in df_items.columns:
                    df_items['Result'] = df_items['is_correct'].apply(lambda x: '✅' if x == 1 else '❌')
                    item_available = [c for c in item_available if c != 'is_correct'] + ['Result']
                
                st.dataframe(df_items[item_available], use_container_width=True, height=400)
                
                # Expandable detail for selected item
                if len(df_items) > 0:
                    detail_item = st.selectbox(
                        "Select item for full response",
                        df_items['compliance_id'].tolist(),
                        key="detail_item_select"
                    )
                    if detail_item:
                        item_row = df_items[df_items['compliance_id'] == detail_item].iloc[0].to_dict()
                        det_c1, det_c2 = st.columns(2)
                        with det_c1:
                            st.markdown("**Phase 2 Response:**")
                            resp = item_row.get('phase2_response', item_row.get('phase2_justification', 'N/A'))
                            with st.expander("View", expanded=False):
                                st.markdown(resp if resp else 'No response stored')
                        with det_c2:
                            st.markdown("**Phase 3 Response:**")
                            resp3 = item_row.get('phase3_response', 'N/A')
                            with st.expander("View", expanded=False):
                                st.markdown(resp3 if resp3 else 'No response stored')
                
                # --- FP/FN Breakdown by Division ---
                if 'office' in df_items.columns and len(df_items) > 0:
                    st.markdown("#### 📊 FP/FN Breakdown by Division")
                    
                    from src.scoring.metrics import calculate_metrics_by_group
                    
                    # Map DB column names to what calculate_metrics expects
                    items_for_metrics = []
                    for _, row in df_items.iterrows():
                        items_for_metrics.append({
                            'Office': row.get('office', ''),
                            'phase2_flag': row.get('phase2_flag', ''),
                            'updated_flag': row.get('ground_truth', '')
                        })
                    
                    group_metrics = calculate_metrics_by_group(items_for_metrics)
                    
                    if group_metrics:
                        chart_data = []
                        for office, gm in group_metrics.items():
                            chart_data.append({
                                'Division': office[:20],
                                'TP': gm.get('true_positives', 0),
                                'FP': gm.get('false_positives', 0),
                                'FN': gm.get('false_negatives', 0),
                                'TN': gm.get('true_negatives', 0),
                                'Accuracy': gm.get('accuracy', 0)
                            })
                        
                        df_chart = pd.DataFrame(chart_data)
                        st.dataframe(df_chart, use_container_width=True)
                        
                        # Bar chart
                        chart_cols = ['Division', 'FP', 'FN']
                        df_bar = df_chart[chart_cols].set_index('Division')
                        st.bar_chart(df_bar)
            else:
                st.info("No item-level results stored for this run (older runs may lack item data).")
        
        # ==========================================
        # Compare Runs (Enhanced Diffing)
        # ==========================================
        st.divider()
        st.markdown("### 🔀 Compare Runs (A vs B)")
        
        diff_c1, diff_c2 = st.columns(2)
        with diff_c1:
            run_a = st.selectbox("Run A (baseline)", run_ids, key="compare_run1")
        with diff_c2:
            run_b = st.selectbox("Run B (new)", run_ids, index=min(1, len(run_ids)-1), key="compare_run2")
        
        if st.button("🔀 Compare", type="primary"):
            comparison = storage.compare_runs(run_a, run_b)
            if "error" not in comparison:
                # Metric deltas
                st.markdown("#### 📈 Metric Deltas")
                delta_c1, delta_c2, delta_c3, delta_c4 = st.columns(4)
                with delta_c1:
                    st.metric("Accuracy Δ", f"{comparison['accuracy_diff']:+.1f}%")
                with delta_c2:
                    st.metric("Precision Δ", f"{comparison['precision_diff']:+.1f}%")
                with delta_c3:
                    st.metric("Recall Δ", f"{comparison['recall_diff']:+.1f}%")
                with delta_c4:
                    st.metric("F1 Δ", f"{comparison['f1_diff']:+.1f}%")
                
                # Flipped items
                st.markdown("#### 🔄 Flipped Items")
                st.caption("Items whose correctness changed between runs")
                
                flipped = storage.get_flipped_items(run_a, run_b)
                
                if flipped:
                    df_flipped = pd.DataFrame(flipped)
                    
                    # Add direction indicator
                    df_flipped['Direction'] = df_flipped.apply(
                        lambda r: '🟢 Improved' if r['run_b_correct'] == 1 else '🔴 Regressed',
                        axis=1
                    )
                    
                    improved = len(df_flipped[df_flipped['run_b_correct'] == 1])
                    regressed = len(df_flipped[df_flipped['run_b_correct'] == 0])
                    
                    flip_c1, flip_c2, flip_c3 = st.columns(3)
                    with flip_c1:
                        st.metric("Total Flipped", len(flipped))
                    with flip_c2:
                        st.metric("🟢 Improved", improved)
                    with flip_c3:
                        st.metric("🔴 Regressed", regressed)
                    
                    flip_display = ['compliance_id', 'office', 'ground_truth',
                                    'run_a_flag', 'run_b_flag', 'Direction']
                    flip_avail = [c for c in flip_display if c in df_flipped.columns]
                    st.dataframe(df_flipped[flip_avail], use_container_width=True)
                else:
                    st.success("No items flipped between these runs.")
                
                # Side-by-side prompt comparison
                r1_details = comparison.get('run_1_details', {})
                r2_details = comparison.get('run_2_details', {})
                pv1 = r1_details.get('prompt_version_id')
                pv2 = r2_details.get('prompt_version_id')
                
                if pv1 and pv2 and pv1 != pv2:
                    st.markdown("#### 📝 Prompt Diff")
                    pv1_data = storage.get_prompt_version(pv1)
                    pv2_data = storage.get_prompt_version(pv2)
                    
                    if pv1_data and pv2_data:
                        import difflib
                        diff_lines = list(difflib.unified_diff(
                            pv1_data['content'].splitlines(keepends=True),
                            pv2_data['content'].splitlines(keepends=True),
                            fromfile=f"Run A: {pv1_data.get('filename', 'unknown')}",
                            tofile=f"Run B: {pv2_data.get('filename', 'unknown')}",
                            lineterm=''
                        ))
                        if diff_lines:
                            st.code(''.join(diff_lines), language='diff')
                        else:
                            st.info("Prompt content is identical.")
                elif pv1 and pv2 and pv1 == pv2:
                    st.info("Both runs used the same prompt version.")
            else:
                st.error(comparison.get('error', 'Comparison failed'))

# ============================================================
# Footer
# ============================================================
st.divider()
st.caption("EO Evaluation Framework v1.0 | Built for rapid prompt iteration and model comparison")
