"""
Streamlit UI for LoRA Model Evaluator.
Provides interactive interface for generation, evaluation, and reporting.
"""

import streamlit as st
import pandas as pd
import os
from backend import ComfyRunner
import matplotlib.pyplot as plt
from PIL import Image
import time
import json
from google import genai
import shutil
import base64


def decode_key(encoded_key):
    """Simple decode function to prevent web scraping"""
    try:
        return base64.b64decode(encoded_key).decode('utf-8')
    except:
        return ""


# Encoded default API key (Base64 encoded to prevent simple scraping)
# To encode a new key: base64.b64encode(b"your_key_here").decode()
DEFAULT_ENCODED_KEY = "QUl6YVN5QVFlWTg5MTRJQTNzUWpwM2xYSFVKYXFVNDdPSUg3UGRz"

# State save/load directory
SESSIONS_DIR = "./sessions"


def get_session_id():
    """Get or create session ID for current session"""
    if 'session_id' not in st.session_state or not st.session_state.session_id:
        # Create new session ID based on timestamp
        st.session_state.session_id = time.strftime("%Y%m%d_%H%M%S")
    return st.session_state.session_id


def get_session_dir(session_id=None):
    """Get session directory path"""
    if session_id is None:
        session_id = get_session_id()
    return os.path.join(SESSIONS_DIR, session_id)


def get_session_output_dir(session_id=None):
    """Get output directory for session images"""
    return os.path.join(get_session_dir(session_id), "output")


def list_available_sessions():
    """List all available sessions"""
    if not os.path.exists(SESSIONS_DIR):
        return []
    
    sessions = []
    for item in os.listdir(SESSIONS_DIR):
        session_path = os.path.join(SESSIONS_DIR, item)
        if os.path.isdir(session_path):
            meta_file = os.path.join(session_path, "session_meta.json")
            if os.path.exists(meta_file):
                try:
                    with open(meta_file, 'r', encoding='utf-8') as f:
                        meta = json.load(f)
                    sessions.append({
                        'id': item,
                        'path': session_path,
                        'meta': meta
                    })
                except:
                    pass
    
    # Sort by timestamp (newest first)
    sessions.sort(key=lambda x: x['meta'].get('timestamp', 0), reverse=True)
    return sessions


def save_session_state(session_id=None):
    """Save current session state to JSON file"""
    try:
        if session_id is None:
            session_id = get_session_id()
        
        session_dir = get_session_dir(session_id)
        os.makedirs(session_dir, exist_ok=True)
        
        # Prepare state data
        state_data = {
            'session_id': session_id,
            'timestamp': time.time(),
            'step_1_setup': {
                'lora_filename': st.session_state.get('lora_filename'),
                'weight_range': st.session_state.get('weight_range'),
                'seeds': st.session_state.get('seeds', []),
                'metrics': st.session_state.get('metrics', []),
                'prompt_pairs': st.session_state.get('prompt_pairs', [])
            },
            'step_2_generation': {
                'generation_complete': st.session_state.get('generation_complete', False),
                'generating': st.session_state.get('generating', False),
                'results': st.session_state.get('results', []),
                'shuffled_results': st.session_state.get('shuffled_results', []),
                'lora_filename': st.session_state.get('lora_filename'),
                'weight_range': st.session_state.get('weight_range'),
                'prompt_pairs_for_generation': st.session_state.get('prompt_pairs_for_generation', []),
                'metrics_for_evaluation': st.session_state.get('metrics_for_evaluation', []),
                'seeds_for_generation': st.session_state.get('seeds_for_generation', [])
            },
            'step_3_evaluation': {
                'evaluation_complete': st.session_state.get('evaluation_complete', False),
                'current_eval_index': st.session_state.get('current_eval_index', 0),
                'current_metric_index': st.session_state.get('current_metric_index', 0),
                'scores': st.session_state.get('scores', [])
            },
            'step_4_report': {
                'report_generated': st.session_state.get('evaluation_complete', False)
            }
        }
        
        # Save main state file
        state_file = os.path.join(session_dir, "session_meta.json")
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(state_data, f, indent=2, ensure_ascii=False)
        
        return True
    except Exception as e:
        print(f"Error saving session state: {e}")
        return False


def load_session_state(session_id):
    """Load session state from JSON file"""
    try:
        session_dir = get_session_dir(session_id)
        state_file = os.path.join(session_dir, "session_meta.json")
        
        if not os.path.exists(state_file):
            return False
        
        with open(state_file, 'r', encoding='utf-8') as f:
            state_data = json.load(f)
        
        # Set session ID
        st.session_state.session_id = session_id
        
        # Restore step 1 (Setup)
        setup = state_data.get('step_1_setup', {})
        if setup.get('seeds'):
            st.session_state.seeds = setup['seeds']
        if setup.get('metrics'):
            st.session_state.metrics = setup['metrics']
        if setup.get('prompt_pairs'):
            st.session_state.prompt_pairs = setup['prompt_pairs']
        
        # Restore step 2 (Generation)
        generation = state_data.get('step_2_generation', {})
        st.session_state.generation_complete = generation.get('generation_complete', False)
        st.session_state.generating = generation.get('generating', False)
        st.session_state.results = generation.get('results', [])
        st.session_state.shuffled_results = generation.get('shuffled_results', [])
        if generation.get('lora_filename'):
            st.session_state.lora_filename = generation['lora_filename']
        if generation.get('weight_range'):
            st.session_state.weight_range = tuple(generation['weight_range'])
        if generation.get('prompt_pairs_for_generation'):
            st.session_state.prompt_pairs_for_generation = generation['prompt_pairs_for_generation']
        if generation.get('metrics_for_evaluation'):
            st.session_state.metrics_for_evaluation = generation['metrics_for_evaluation']
        if generation.get('seeds_for_generation'):
            st.session_state.seeds_for_generation = generation['seeds_for_generation']
        
        # Restore step 3 (Evaluation)
        evaluation = state_data.get('step_3_evaluation', {})
        st.session_state.evaluation_complete = evaluation.get('evaluation_complete', False)
        st.session_state.current_eval_index = evaluation.get('current_eval_index', 0)
        st.session_state.current_metric_index = evaluation.get('current_metric_index', 0)
        st.session_state.scores = evaluation.get('scores', [])
        
        return True
    except Exception as e:
        print(f"Error loading session state: {e}")
        return False


def create_new_session():
    """Create a new session and clear current state"""
    # Clear current session state
    for key in ['session_id', 'generation_complete', 'results', 'shuffled_results', 
                'current_eval_index', 'current_metric_index', 'scores', 
                'evaluation_complete', 'generating']:
        if key in st.session_state:
            del st.session_state[key]
    
    # Create new session ID
    new_id = time.strftime("%Y%m%d_%H%M%S")
    st.session_state.session_id = new_id
    
    return new_id


def get_current_step():
    """Determine current step based on session state"""
    if st.session_state.get('evaluation_complete', False):
        return 4  # Report
    elif st.session_state.get('generation_complete', False):
        return 3  # Evaluation
    elif st.session_state.get('generating', False) or len(st.session_state.get('results', [])) > 0:
        return 2  # Generation
    else:
        return 1  # Setup


# Initialize session state
if 'generation_complete' not in st.session_state:
    st.session_state.generation_complete = False
if 'results' not in st.session_state:
    st.session_state.results = []
if 'shuffled_results' not in st.session_state:
    st.session_state.shuffled_results = []
if 'current_eval_index' not in st.session_state:
    st.session_state.current_eval_index = 0
if 'current_metric_index' not in st.session_state:
    st.session_state.current_metric_index = 0
if 'scores' not in st.session_state:
    st.session_state.scores = []
if 'evaluation_complete' not in st.session_state:
    st.session_state.evaluation_complete = False
if 'generating' not in st.session_state:
    st.session_state.generating = False
if 'metrics' not in st.session_state:
    st.session_state.metrics = []
if 'prompt_pairs' not in st.session_state:
    st.session_state.prompt_pairs = []
if 'seeds' not in st.session_state:
    import random
    st.session_state.seeds = [random.randint(1, 1000000)]
if 'session_loaded' not in st.session_state:
    st.session_state.session_loaded = False


def call_llm(prompt, api_key, model="gemini-2.5-flash"):
    """Call Gemini API to generate content"""
    try:
        # Initialize client with API key
        client = genai.Client(api_key=api_key)
        
        full_prompt = f"""You are a helpful assistant that generates prompts and evaluation metrics for image generation tasks. Return only valid JSON.

{prompt}"""
        
        response = client.models.generate_content(
            model=model,
            contents=full_prompt
        )
        return response.text
    except Exception as e:
        print(f"LLM Error: {str(e)}")
        raise e


def load_prompt_template(filename, default_content):
    """Load prompt template from file or create with default content"""
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        # Create default file
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(default_content)
        return default_content


def generate_prompts_with_llm(description, api_key, model="gemini-2.5-flash"):
    """Generate prompt pairs using LLM"""
    default_prompt = """Generate 2-3 image generation prompt pairs based on this description: "{description}"
    
Return a JSON array of objects, each with 'positive' and 'negative' keys.
Positive prompts should be detailed and descriptive.
Negative prompts should list common artifacts to avoid.

Example format:
{{
  "prompts": [
    {{
      "positive": "detailed prompt here",
      "negative": "bad quality, artifacts"
    }}
  ]
}}"""
    
    prompt_template = load_prompt_template('prompts/generate_prompts_template.txt', default_prompt)
    prompt = prompt_template.replace('{description}', description)
    
    try:
        result = call_llm(prompt, api_key, model)
        # Try to parse JSON from response
        if '```json' in result:
            result = result.split('```json')[1].split('```')[0]
        elif '```' in result:
            result = result.split('```')[1].split('```')[0]
        
        data = json.loads(result.strip())
        return data.get('prompts', [])
    except Exception as e:
        print(f"Prompts generation error: {str(e)}")
        return None


def generate_metrics_with_llm(description, api_key, model="gemini-2.5-flash"):
    """Generate evaluation metrics using LLM"""
    default_prompt = """Generate 2-3 evaluation metrics for assessing LoRA model performance based on: "{description}"
    
Return a JSON array of metric names (short phrases, 2-4 words each).
Metrics should be specific, measurable aspects of image quality.

Example format:
{{
  "metrics": ["Visual Quality", "Style Consistency", "Detail Level"]
}}"""
    
    prompt_template = load_prompt_template('prompts/generate_metrics_template.txt', default_prompt)
    prompt = prompt_template.replace('{description}', description)
    
    try:
        result = call_llm(prompt, api_key, model)
        if '```json' in result:
            result = result.split('```json')[1].split('```')[0]
        elif '```' in result:
            result = result.split('```')[1].split('```')[0]
        
        data = json.loads(result.strip())
        return data.get('metrics', [])
    except Exception as e:
        print(f"Metrics generation error: {str(e)}")
        return None


def inject_custom_css():
    """Inject custom CSS for better button styling"""
    st.markdown("""
    <style>
    /* Improve button styling with hover effects */
    .stButton > button {
        transition: all 0.3s ease;
        border: 2px solid transparent;
        font-weight: 600;
        font-size: 16px;
        padding: 12px 24px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        border-color: currentColor;
    }
    
    .stButton > button:active {
        transform: translateY(0px);
    }
    
    /* Choice indicator */
    .choice-indicator {
        padding: 12px;
        border-radius: 8px;
        margin: 8px 0;
        font-weight: 600;
        text-align: center;
        transition: all 0.3s ease;
        animation: fadeIn 0.3s ease;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .choice-yes {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: 2px solid #667eea;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    .choice-no {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        border: 2px solid #f5576c;
        box-shadow: 0 4px 12px rgba(245, 87, 108, 0.3);
    }
    
    .choice-same {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        border: 2px solid #4facfe;
        box-shadow: 0 4px 12px rgba(79, 172, 254, 0.3);
    }
    </style>
    """, unsafe_allow_html=True)


def render_star_rating(current_idx):
    """Render interactive star rating"""
    if st.session_state.temp_stars:
        stars_display = "⭐" * st.session_state.temp_stars + "☆" * (5 - st.session_state.temp_stars)
    else:
        stars_display = "☆" * 5
    
    st.markdown(f"""
    <div style="text-align: center; padding: 16px 0;">
        <div style="font-size: 32px; letter-spacing: 4px; margin-bottom: 8px;">
            {stars_display}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Star buttons
    star_cols = st.columns(5)
    for i, col in enumerate(star_cols):
        star_num = i + 1
        with col:
            is_selected = st.session_state.temp_stars and star_num <= st.session_state.temp_stars
            if st.button(
                f"{'⭐' if is_selected else '☆'}", 
                key=f"star_{star_num}_{current_idx}", 
                use_container_width=True,
                type="primary" if is_selected else "secondary"
            ):
                st.session_state.temp_stars = star_num
                st.rerun()
    
    if st.session_state.temp_stars:
        quality_map = {
            1: "Slightly Better",
            2: "Somewhat Better", 
            3: "Moderately Better",
            4: "Significantly Better",
            5: "Dramatically Better"
        }
        st.markdown(f"<p style='text-align: center; color: #666; font-size: 14px; margin-top: 8px;'>{quality_map[st.session_state.temp_stars]}</p>", unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="LoRA Model Evaluator", layout="wide")
    inject_custom_css()
    
    st.title("🎨 Semi-automatic LoRA Model Evaluator")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    comfyui_url = st.sidebar.text_input("ComfyUI Server", "127.0.0.1:8188")
    
    # Session management in sidebar
    st.sidebar.divider()
    st.sidebar.subheader("💾 Session Management")
    
    # Current session info
    current_session_id = get_session_id()
    st.sidebar.info(f"**Current Session:**\n`{current_session_id}`")
    
    current_step = get_current_step()
    step_names = {1: "Setup", 2: "Generation", 3: "Evaluation", 4: "Report"}
    st.sidebar.caption(f"Step: {current_step}. {step_names[current_step]}")
    
    # Session actions
    col_new, col_save = st.sidebar.columns(2)
    with col_new:
        if st.button("🆕 New", help="Start a new session"):
            new_id = create_new_session()
            st.sidebar.success(f"✅ New session: {new_id}")
            time.sleep(1)
            st.rerun()
    with col_save:
        if st.button("💾 Save", help="Save current session"):
            if save_session_state():
                st.sidebar.success("✅ Saved!")
                time.sleep(1)
                st.rerun()
    
    # Load existing sessions
    st.sidebar.divider()
    st.sidebar.subheader("📂 Load Session")
    
    available_sessions = list_available_sessions()
    if available_sessions:
        session_options = {}
        for sess in available_sessions:
            sess_id = sess['id']
            meta = sess['meta']
            timestamp = meta.get('timestamp', 0)
            lora = meta.get('step_1_setup', {}).get('lora_filename', 'N/A')
            import datetime
            dt = datetime.datetime.fromtimestamp(timestamp)
            label = f"{sess_id} - {lora} ({dt.strftime('%m/%d %H:%M')})"
            session_options[label] = sess_id
        
        selected_label = st.sidebar.selectbox(
            "Select session to load",
            options=[""] + list(session_options.keys()),
            format_func=lambda x: "Choose a session..." if x == "" else x
        )
        
        if selected_label and selected_label != "":
            if st.sidebar.button("📥 Load Selected"):
                selected_id = session_options[selected_label]
                if load_session_state(selected_id):
                    st.sidebar.success(f"✅ Loaded {selected_id}")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.sidebar.error("❌ Failed to load")
    else:
        st.sidebar.caption("No saved sessions found")
    
    # Initialize ComfyRunner with session-specific output directory
    session_id = get_session_id()
    session_output_dir = get_session_output_dir(session_id)
    runner = ComfyRunner(server_address=comfyui_url, output_dir=session_output_dir)
    
    # Main workflow tabs
    tab1, tab2, tab3, tab4 = st.tabs(["⚙️ Setup", "🎬 Generation", "📊 Evaluation", "📈 Report"])
    
    # ===== TAB 1: SETUP =====
    with tab1:
        st.header("Step 1: Setup Parameters")
        
        # Show session restore info if applicable
        if st.session_state.get('generation_complete') or len(st.session_state.get('results', [])) > 0:
            st.info("📂 Session restored from previous run. You can continue from where you left off or reset to start over.")
        
        # API Key configuration - use default encoded key
        api_key = os.environ.get("GEMINI_API_KEY", decode_key(DEFAULT_ENCODED_KEY))
        llm_model = "gemini-2.5-flash"  # Default model
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("LoRA Configuration")
            
            # ComfyUI LoRA path configuration
            with st.expander("⚙️ ComfyUI Path Settings", expanded=False):
                comfyui_lora_path = st.text_input(
                    "ComfyUI LoRA Folder Path",
                    value=os.environ.get("COMFYUI_LORA_PATH", r"D:\ComfyUI\models\loras"),
                    help="Path to your ComfyUI's models/loras folder."
                )
                if comfyui_lora_path and not os.path.exists(comfyui_lora_path):
                    st.warning(f"⚠️ Path does not exist: {comfyui_lora_path}")
            
            # Get existing LoRA files from directory
            lora_dir = comfyui_lora_path if comfyui_lora_path and os.path.exists(comfyui_lora_path) else "./loras"
            os.makedirs(lora_dir, exist_ok=True)
            
            existing_loras = []
            if os.path.exists(lora_dir):
                existing_loras = [f for f in os.listdir(lora_dir) 
                                if f.endswith(('.safetensors', '.pt', '.ckpt'))]
            
            # LoRA file upload
            uploaded_lora = st.file_uploader(
                "Upload LoRA File",
                type=["safetensors", "pt", "ckpt"],
                help="Upload your LoRA model file (.safetensors, .pt, or .ckpt)"
            )
            
            # Save uploaded file and get filename
            lora_filename = None
            if uploaded_lora is not None:
                lora_path = os.path.join(lora_dir, uploaded_lora.name)
                lora_filename = uploaded_lora.name
                
                # Check if file already exists
                if os.path.exists(lora_path):
                    st.info(f"ℹ️ File already exists: {lora_filename}")
                    st.caption(f"📁 Using existing file at: {lora_path}")
                else:
                    # Save the file
                    with open(lora_path, "wb") as f:
                        f.write(uploaded_lora.getbuffer())
                    st.success(f"✅ Uploaded: {lora_filename}")
                    st.caption(f"📁 Saved to: {lora_path}")
                    # Refresh the list
                    existing_loras.append(uploaded_lora.name)
            else:
                # Select from existing files
                if existing_loras:
                    lora_filename = st.selectbox(
                        "Or select existing LoRA file",
                        options=[""] + existing_loras,
                        format_func=lambda x: "Select a file..." if x == "" else x,
                        help="Choose from LoRA files already in the folder"
                    )
                    if lora_filename == "":
                        lora_filename = None
                else:
                    st.info(f"📁 No LoRA files found in: {lora_dir}")
                    lora_filename = st.text_input(
                        "Or enter LoRA filename manually",
                        placeholder="e.g., my_lora.safetensors",
                        help="Enter the filename if it exists elsewhere"
                    )
            
            st.subheader("Weight Range")
            weight_min = st.number_input("Minimum Weight", value=0.0, step=0.1)
            weight_max = st.number_input("Maximum Weight", value=1.0, step=0.1)
            weight_step = st.number_input("Weight Step", value=0.5, step=0.1, min_value=0.1)
            
            st.divider()
            st.subheader("Seeds")
            st.write("**Seeds** (for reproducible generation)")
            for i, seed in enumerate(st.session_state.seeds):
                col_seed, col_del = st.columns([4, 1])
                with col_seed:
                    new_seed = st.number_input(
                        f"Seed {i+1}",
                        min_value=1,
                        max_value=1000000,
                        value=seed,
                        key=f"seed_{i}"
                    )
                    if new_seed != seed:
                        st.session_state.seeds[i] = new_seed
                with col_del:
                    st.write("")
                    st.write("")
                    if len(st.session_state.seeds) > 1:
                        if st.button("❌", key=f"del_seed_{i}"):
                            st.session_state.seeds.pop(i)
                            st.rerun()
            
            if st.button("➕ Add Seed"):
                import random
                st.session_state.seeds.append(random.randint(1, 1000000))
                st.rerun()
        
        with col2:
            st.subheader("Evaluation Metrics")
            st.write("Define 1-3 metrics for evaluating image quality (max 3).")
            
            # Display metrics
            for idx, metric in enumerate(st.session_state.metrics):
                col_metric, col_remove = st.columns([4, 1])
                with col_metric:
                    st.session_state.metrics[idx] = st.text_input(
                        f"Metric {idx + 1}",
                        value=metric,
                        key=f"metric_{idx}",
                        label_visibility="collapsed"
                    )
                with col_remove:
                    if len(st.session_state.metrics) > 1:
                        if st.button("🗑️", key=f"remove_metric_{idx}"):
                            st.session_state.metrics.pop(idx)
                            st.rerun()
            
            # Add/Reset metrics buttons
            col_add_m, col_clear_m = st.columns(2)
            with col_add_m:
                if st.button("➕ Add Metric", disabled=len(st.session_state.metrics) >= 3):
                    if len(st.session_state.metrics) < 3:
                        st.session_state.metrics.append('')
                        st.rerun()
            with col_clear_m:
                if st.button("🔄 Clear All Metrics"):
                    st.session_state.metrics = []
                    st.rerun()
            
            # LLM generation for metrics
            metric_description = st.text_input(
                "Or describe what aspects to evaluate",
                placeholder="e.g., focus on facial details and color accuracy",
                key="metric_desc"
            )
            if st.button("✨ Generate Metrics with AI"):
                if metric_description:
                    with st.spinner("Generating metrics..."):
                        generated = generate_metrics_with_llm(metric_description, api_key, llm_model)
                        if generated:
                            # 追加模式：将新生成的metrics添加到现有列表
                            st.session_state.metrics.extend(generated)
                            st.success(f"已追加 {len(generated)} 个评价指标！当前共 {len(st.session_state.metrics)} 个指标")
                            st.rerun()
                        else:
                            st.error("生成失败，请检查网络连接或稍后重试。")
                else:
                    st.warning("Please enter a description first.")
            
            st.divider()
            st.subheader("Prompts")
            st.write("Enter prompt pairs (positive and negative). Each pair will be tested with all weights.")
            
            # Display existing prompt pairs
            for idx, pair in enumerate(st.session_state.prompt_pairs):
                with st.expander(f"Prompt Pair {idx + 1}", expanded=(idx == 0)):
                    pair['positive'] = st.text_area(
                        "Positive Prompt",
                        value=pair['positive'],
                        height=80,
                        key=f"positive_{idx}"
                    )
                    pair['negative'] = st.text_area(
                        "Negative Prompt",
                        value=pair['negative'],
                        height=60,
                        key=f"negative_{idx}"
                    )
                    if len(st.session_state.prompt_pairs) > 1:
                        if st.button(f"🗑️ Remove Pair {idx + 1}", key=f"remove_{idx}"):
                            st.session_state.prompt_pairs.pop(idx)
                            st.rerun()
            
            # Add new prompt pair button
            col_add, col_clear = st.columns(2)
            with col_add:
                if st.button("➕ Add Prompt Pair"):
                    st.session_state.prompt_pairs.append({
                        'positive': '',
                        'negative': '(worst quality, low quality:1.4)'
                    })
                    st.rerun()
            with col_clear:
                if st.button("🗑️ Clear All"):
                    st.session_state.prompt_pairs = []
                    st.rerun()
            
            # LLM generation for prompts
            st.divider()
            st.write("**🤖 AI Generation**")
            prompt_description = st.text_input(
                "Describe what you want to generate",
                placeholder="e.g., anime characters in various poses and styles",
                key="prompt_desc"
            )
            if st.button("✨ Generate Prompts with AI"):
                if prompt_description:
                    with st.spinner("Generating prompts..."):
                        generated = generate_prompts_with_llm(prompt_description, api_key, llm_model)
                        if generated:
                            # 追加模式：将新生成的prompt pairs添加到现有列表
                            st.session_state.prompt_pairs.extend(generated)
                            st.success(f"已追加 {len(generated)} 组提示词对！当前共 {len(st.session_state.prompt_pairs)} 组")
                            st.rerun()
                        else:
                            st.error("生成失败，请检查网络连接或稍后重试。")
                else:
                    st.warning("Please enter a description first.")
        
        # Parse prompts (filter out empty ones)
        prompt_pairs = [pair for pair in st.session_state.prompt_pairs if pair['positive'].strip()]
        metrics = [m.strip() for m in st.session_state.metrics if m.strip()]
        
        num_seeds = len(st.session_state.seeds)
        # Calculate actual number of non-zero weights
        weights = []
        current = weight_min
        while current <= weight_max:
            w = round(current, 2)
            if w != 0:  # Backend skips 0 as baseline is separate
                weights.append(w)
            current += weight_step
        num_weights = len(weights)
        total_pairs = len(prompt_pairs) * num_weights * num_seeds
        st.info(f"📊 Total pairs to evaluate: {len(prompt_pairs)} prompts × {num_weights} non-zero weights × {num_seeds} seeds = {total_pairs} pairs (each with baseline + lora images)")
        
        # Validation messages
        can_generate = True
        validation_messages = []
        
        if not lora_filename:
            validation_messages.append("❌ Please enter a LoRA filename")
            can_generate = False
        if not prompt_pairs:
            validation_messages.append("❌ Please enter at least one prompt pair")
            can_generate = False
        if not metrics:
            validation_messages.append("❌ Please define at least one metric")
            can_generate = False
        if len(metrics) > 3:
            validation_messages.append(f"❌ Maximum 3 metrics allowed (current: {len(metrics)}). Please remove {len(metrics) - 3} metric(s)")
            can_generate = False
        
        if validation_messages:
            for msg in validation_messages:
                st.warning(msg)
        
        # Start Generation Button
        if st.session_state.generating:
            st.info("🎬 Generation in progress... Please wait or check the Generation tab.")
        
        button_disabled = not can_generate or st.session_state.generating
        button_label = "⏳ Generating..." if st.session_state.generating else "🚀 Start Generation"
        
        if st.button(button_label, type="primary", use_container_width=True, disabled=button_disabled):
            st.session_state.generation_complete = False
            st.session_state.results = []
            st.session_state.current_eval_index = 0
            st.session_state.scores = []
            st.session_state.evaluation_complete = False
            st.session_state.generating = True
            
            # Store parameters
            st.session_state.lora_filename = lora_filename
            st.session_state.weight_range = (weight_min, weight_max, weight_step)
            st.session_state.prompt_pairs_for_generation = prompt_pairs
            st.session_state.metrics_for_evaluation = metrics
            st.session_state.seeds_for_generation = st.session_state.seeds
            
            # Save before starting generation
            save_session_state()
            
            st.rerun()
    
    # ===== TAB 2: GENERATION =====
    with tab2:
        st.header("Step 2: Image Generation")
        
        if not st.session_state.generation_complete and hasattr(st.session_state, 'lora_filename'):
            st.info("🎬 Generation in progress...")
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def progress_callback(current, total):
                progress = current / total
                progress_bar.progress(progress)
                status_text.text(f"Generating image {current} of {total}...")
            
            # Run generation for all seeds
            try:
                all_results = []
                for seed in st.session_state.seeds_for_generation:
                    results = runner.generate_batch(
                        lora_name=st.session_state.lora_filename,
                        weight_range=st.session_state.weight_range,
                        prompt_pairs=st.session_state.prompt_pairs_for_generation,
                        base_seed=seed,
                        progress_callback=progress_callback
                    )
                    all_results.extend(results)
                
                st.session_state.results = all_results
                
                # Shuffle results for blind evaluation and prepare with random left/right order
                import random
                shuffled = all_results.copy()
                random.shuffle(shuffled)
                
                # For each result, randomly decide left/right order
                st.session_state.shuffled_results = []
                for result in shuffled:
                    swap = random.choice([True, False])
                    st.session_state.shuffled_results.append({
                        'result': result,
                        'swap': swap  # If True, swap baseline and lora positions
                    })
                
                st.session_state.generation_complete = True
                st.session_state.generating = False
                st.session_state.current_eval_index = 0
                st.session_state.current_metric_index = 0
                st.session_state.scores = []
                
                # Auto-save after generation
                save_session_state()
                
                st.success(f"✅ Generation complete! {len(all_results)} pairs created.")
                st.balloons()
                
            except Exception as e:
                st.error(f"❌ Error during generation: {str(e)}")
                st.exception(e)
                st.session_state.generating = False
        
        elif st.session_state.generation_complete:
            st.success(f"✅ Generation complete! {len(st.session_state.results)} pairs created.")
            
            # Show generated image pairs in a grid
            st.subheader("Generated Image Pairs")
            
            if st.session_state.results:
                for idx, result in enumerate(st.session_state.results):
                    with st.expander(f"Pair {idx+1}: Weight {result['weight']} - {result['prompt'][:50]}..."):
                        cols = st.columns(2)
                        with cols[0]:
                            st.write("**Without LoRA (Weight=0)**")
                            if os.path.exists(result['baseline']['image_path']):
                                img = Image.open(result['baseline']['image_path'])
                                st.image(img, use_container_width=True)
                        with cols[1]:
                            st.write(f"**With LoRA (Weight={result['weight']})**")
                            if os.path.exists(result['lora']['image_path']):
                                img = Image.open(result['lora']['image_path'])
                                st.image(img, use_container_width=True)
        else:
            st.info("👈 Please configure parameters in the Setup tab and click 'Start Generation'")
    
    # ===== TAB 3: EVALUATION =====
    with tab3:
        st.header("Step 3: Blind Image Evaluation")
        
        if not st.session_state.generation_complete:
            st.info("⏳ Please complete generation first")
        elif st.session_state.evaluation_complete:
            st.success("✅ Evaluation complete! Check the Report tab for results.")
        else:
            shuffled_results = st.session_state.shuffled_results
            metrics = st.session_state.metrics_for_evaluation
            current_idx = st.session_state.current_eval_index
            current_metric_idx = st.session_state.current_metric_index
            
            # Calculate total evaluations needed (pairs × metrics)
            total_evals = len(shuffled_results) * len(metrics)
            current_eval_num = current_idx * len(metrics) + current_metric_idx + 1
            
            if current_idx >= len(shuffled_results):
                st.session_state.evaluation_complete = True
                
                # Save scores to CSV
                scores_df = pd.DataFrame(st.session_state.scores)
                scores_path = os.path.join(runner.output_dir, "scores.csv")
                scores_df.to_csv(scores_path, index=False)
                
                # Auto-save on completion
                save_session_state()
                
                st.success("✅ All evaluations complete!")
                st.rerun()
            else:
                shuffled_item = shuffled_results[current_idx]
                result = shuffled_item['result']
                swap = shuffled_item['swap']
                current_metric = metrics[current_metric_idx]
                
                st.progress(current_eval_num / total_evals)
                st.write(f"**Evaluation {current_eval_num} of {total_evals}** | Pair {current_idx + 1}/{len(shuffled_results)} | Metric: **{current_metric}**")
                
                # Main layout: images on left (70%), controls on right (30%)
                col_images, col_controls = st.columns([7, 3])
                
                # Load images once at the beginning (outside columns for efficiency)
                left_path = result['lora']['image_path'] if swap else result['baseline']['image_path']
                right_path = result['baseline']['image_path'] if swap else result['lora']['image_path']
                
                # Cache images to avoid reloading on every interaction
                @st.cache_data
                def load_image(path):
                    if os.path.exists(path):
                        return Image.open(path)
                    return None
                
                left_img = load_image(left_path)
                right_img = load_image(right_path)
                
                with col_images:
                    # Blind evaluation - randomly swap left/right positions
                    # Don't show which is baseline or lora
                    img_col1, img_col2 = st.columns(2)
                    
                    with img_col1:
                        st.markdown("### 🔵 Image A")
                        if left_img:
                            st.image(left_img, use_container_width=True)
                        else:
                            st.error("Image not found")
                    
                    with img_col2:
                        st.markdown("### 🟢 Image B")
                        if right_img:
                            st.image(right_img, use_container_width=True)
                        else:
                            st.error("Image not found")
                
                with col_controls:
                    st.markdown(f"### 📊 {current_metric}")
                    st.write("Which image is better for this metric?")
                    
                    # Initialize temp rating state for this comparison
                    if 'temp_choice' not in st.session_state:
                        st.session_state.temp_choice = None
                    if 'temp_stars' not in st.session_state:
                        st.session_state.temp_stars = None
                    
                    # Show current choice with visual indicator
                    if st.session_state.temp_choice:
                        choice_class = f"choice-{st.session_state.temp_choice}"
                        choice_text = {
                            'A': '✅ Image A is Better',
                            'B': '✅ Image B is Better',
                            'same': '➖ About the Same'
                        }
                        st.markdown(f'<div class="choice-indicator {choice_class}">{choice_text[st.session_state.temp_choice]}</div>', unsafe_allow_html=True)
                    
                    st.write("")
                    
                    # A/B/Same buttons (horizontal layout)
                    btn_col1, btn_col2, btn_col3 = st.columns(3)
                    
                    with btn_col1:
                        if st.button("🔵\nImage A", use_container_width=True, 
                                   type="primary" if st.session_state.temp_choice == 'A' else "secondary",
                                   key=f"btn_A_{current_idx}_{current_metric_idx}"):
                            if st.session_state.temp_choice != 'A':
                                st.session_state.temp_choice = 'A'
                                st.rerun()
                    
                    with btn_col2:
                        if st.button("🟢\nImage B", use_container_width=True,
                                   type="primary" if st.session_state.temp_choice == 'B' else "secondary",
                                   key=f"btn_B_{current_idx}_{current_metric_idx}"):
                            if st.session_state.temp_choice != 'B':
                                st.session_state.temp_choice = 'B'
                                st.session_state.temp_stars = None
                                st.rerun()
                    
                    with btn_col3:
                        if st.button("➖\nSame", use_container_width=True,
                                   type="primary" if st.session_state.temp_choice == 'same' else "secondary",
                                   key=f"btn_same_{current_idx}_{current_metric_idx}"):
                            if st.session_state.temp_choice != 'same':
                                st.session_state.temp_choice = 'same'
                                st.session_state.temp_stars = None
                                st.rerun()
                    
                    st.divider()
                    
                    # Show star rating if A or B is selected (how much better)
                    if st.session_state.temp_choice in ['A', 'B']:
                        st.markdown("**How much better?** *(Optional)*")
                        render_star_rating(f"{current_idx}_{current_metric_idx}")
                    
                    st.divider()
                    
                    # Submit button
                    if st.session_state.temp_choice:
                        if st.button("✅ Submit & Next", use_container_width=True, type="primary", key=f"submit_{current_idx}_{current_metric_idx}"):
                            # Determine which was lora based on choice and swap
                            lora_better = None
                            if st.session_state.temp_choice != 'same':
                                # If chose A and swap is False (A is baseline), then lora is not better
                                # If chose A and swap is True (A is lora), then lora is better
                                # If chose B and swap is False (B is lora), then lora is better
                                # If chose B and swap is True (B is baseline), then lora is not better
                                if st.session_state.temp_choice == 'A':
                                    lora_better = swap  # True if A is lora
                                else:  # chose B
                                    lora_better = not swap  # True if B is lora
                            
                            # Save the score
                            score_entry = {
                                'pair_index': current_idx,
                                'metric': current_metric,
                                'baseline_path': result['baseline']['image_path'],
                                'lora_path': result['lora']['image_path'],
                                'weight': result['weight'],
                                'prompt': result['prompt'],
                                'seed': result['seed'],
                                'choice': st.session_state.temp_choice,
                                'stars': st.session_state.temp_stars if st.session_state.temp_choice in ['A', 'B'] else None,
                                'lora_better': lora_better
                            }
                            st.session_state.scores.append(score_entry)
                            
                            # Reset temp state
                            st.session_state.temp_choice = None
                            st.session_state.temp_stars = None
                            
                            # Move to next metric or next pair
                            if current_metric_idx + 1 < len(metrics):
                                st.session_state.current_metric_index += 1
                            else:
                                st.session_state.current_metric_index = 0
                                st.session_state.current_eval_index += 1
                            
                            # Auto-save progress every 5 evaluations
                            if len(st.session_state.scores) % 5 == 0:
                                save_session_state()
                            
                            st.rerun()
                    else:
                        st.info("👆 Please select an option above")
                    
                    st.write("")
                    
                    # Skip buttons at bottom
                    skip_col1, skip_col2 = st.columns(2)
                    with skip_col1:
                        if st.button("⏭️ Skip", use_container_width=True, key=f"skip_{current_idx}_{current_metric_idx}"):
                            st.session_state.scores.append({
                                'pair_index': current_idx,
                                'metric': current_metric,
                                'baseline_path': result['baseline']['image_path'],
                                'lora_path': result['lora']['image_path'],
                                'weight': result['weight'],
                                'prompt': result['prompt'],
                                'seed': result['seed'],
                                'choice': 'skip',
                                'stars': None,
                                'lora_better': None
                            })
                            st.session_state.temp_choice = None
                            st.session_state.temp_stars = None
                            # Move to next metric or next pair
                            if current_metric_idx + 1 < len(metrics):
                                st.session_state.current_metric_index += 1
                            else:
                                st.session_state.current_metric_index = 0
                                st.session_state.current_eval_index += 1
                            
                            # Auto-save on skip
                            save_session_state()
                            st.rerun()
                    
                    with skip_col2:
                        if st.button("⏩ Skip to End", use_container_width=True, key=f"skip_all_{current_idx}_{current_metric_idx}"):
                            # Skip all remaining evaluations
                            st.session_state.evaluation_complete = True
                            save_session_state()
                            st.success("Skipped to end!")
                            st.rerun()
    
    # ===== TAB 4: REPORT =====
    with tab4:
        st.header("Step 4: Evaluation Report")
        
        if not st.session_state.evaluation_complete:
            st.info("⏳ Please complete evaluation first")
        else:
            scores_df = pd.DataFrame(st.session_state.scores)
            
            if len(scores_df) > 0:
                # Filter out skipped evaluations
                scored_df = scores_df[scores_df['choice'] != 'skip'].copy()
                
                if len(scored_df) > 0:
                    # Calculate statistics by metric and weight
                    st.subheader("📊 Results by Metric")
                    
                    for metric in st.session_state.metrics_for_evaluation:
                        metric_df = scored_df[scored_df['metric'] == metric].copy()
                        if len(metric_df) > 0:
                            st.write(f"**{metric}**")
                            
                            # Count lora_better results by weight
                            weight_stats = metric_df.groupby('weight').agg({
                                'lora_better': lambda x: x.sum(),  # Count True values
                                'pair_index': 'count'  # Total count
                            }).reset_index()
                            weight_stats.columns = ['weight', 'lora_better_count', 'total_count']
                            weight_stats['lora_better_rate'] = weight_stats['lora_better_count'] / weight_stats['total_count']
                            
                            # Calculate average star rating for "better" choices
                            better_df = metric_df[metric_df['lora_better'] == True].copy()
                            if len(better_df) > 0 and 'stars' in better_df.columns:
                                star_stats = better_df.groupby('weight')['stars'].mean().reset_index()
                                star_stats.columns = ['weight', 'avg_stars']
                                weight_stats = weight_stats.merge(star_stats, on='weight', how='left')
                            
                            st.dataframe(weight_stats, use_container_width=True)
                            
                            # Chart for this metric
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                            
                            # Success rate chart
                            ax1.plot(weight_stats['weight'], weight_stats['lora_better_rate'], marker='o', linewidth=2, markersize=8, color='#1f77b4')
                            ax1.set_xlabel('LoRA Weight', fontsize=11)
                            ax1.set_ylabel('LoRA Better Rate', fontsize=11)
                            ax1.set_title(f'{metric} - LoRA Better Rate by Weight', fontsize=12, fontweight='bold')
                            ax1.grid(True, alpha=0.3)
                            ax1.set_ylim(0, 1)
                            
                            # Star rating chart (if available)
                            if 'avg_stars' in weight_stats.columns:
                                star_data = weight_stats.dropna(subset=['avg_stars'])
                                if len(star_data) > 0:
                                    ax2.plot(star_data['weight'], star_data['avg_stars'], marker='*', linewidth=2, markersize=12, color='#ff7f0e')
                                    ax2.set_xlabel('LoRA Weight', fontsize=11)
                                    ax2.set_ylabel('Average Star Rating', fontsize=11)
                                    ax2.set_title(f'{metric} - Avg Quality Rating', fontsize=12, fontweight='bold')
                                    ax2.grid(True, alpha=0.3)
                                    ax2.set_ylim(0, 5.5)
                                else:
                                    ax2.text(0.5, 0.5, 'No star ratings', ha='center', va='center', transform=ax2.transAxes)
                            else:
                                ax2.text(0.5, 0.5, 'No star ratings', ha='center', va='center', transform=ax2.transAxes)
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()
                            st.write("")
                    
                    # Overall summary across all metrics
                    st.divider()
                    st.subheader("📊 Overall Summary")
                    overall_stats = scored_df.groupby('weight').agg({
                        'lora_better': lambda x: x.sum(),
                        'pair_index': 'count'
                    }).reset_index()
                    overall_stats.columns = ['weight', 'lora_better_count', 'total_count']
                    overall_stats['lora_better_rate'] = overall_stats['lora_better_count'] / overall_stats['total_count']
                    
                    st.dataframe(overall_stats, use_container_width=True)
                    
                    if len(overall_stats) > 0:
                        best_weight = overall_stats.loc[overall_stats['lora_better_rate'].idxmax(), 'weight']
                        best_rate = overall_stats.loc[overall_stats['lora_better_rate'].idxmax(), 'lora_better_rate']
                        st.success(f"🏆 Best Weight (Overall): **{best_weight}** (LoRA Better Rate: {best_rate:.1%})")
                    
                    # Show best images (pairs where lora was better)
                    st.subheader("🌟 Top Rated Pairs (LoRA Better)")
                    best_images = scored_df[scored_df['lora_better'] == True].copy()
                    if 'stars' in best_images.columns:
                        best_images = best_images.dropna(subset=['stars']).nlargest(6, 'stars')
                    else:
                        best_images = best_images.head(6)
                    
                    for idx, (_, row) in enumerate(best_images.iterrows()):
                        if idx % 2 == 0:
                            cols = st.columns(2)
                        
                        with cols[idx % 2]:
                            st.write(f"**Weight: {row['weight']}** - Rating: {'⭐' * int(row['stars']) if pd.notna(row.get('stars')) else 'N/A'}")
                            sub_cols = st.columns(2)
                            with sub_cols[0]:
                                if os.path.exists(row['baseline_path']):
                                    img = Image.open(row['baseline_path'])
                                    st.image(img, caption="Without LoRA", use_container_width=True)
                            with sub_cols[1]:
                                if os.path.exists(row['lora_path']):
                                    img = Image.open(row['lora_path'])
                                    st.image(img, caption="With LoRA", use_container_width=True)
                            st.caption(f"Prompt: {row['prompt'][:50]}...")
                            st.divider()
                    
                    # Download results
                    st.subheader("💾 Download Results")
                    csv = scores_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Scores CSV",
                        data=csv,
                        file_name="lora_evaluation_scores.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No evaluations were completed (all skipped)")
            else:
                st.warning("No evaluation data available")


if __name__ == "__main__":
    main()
