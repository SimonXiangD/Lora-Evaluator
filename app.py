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


# Initialize session state
if 'generation_complete' not in st.session_state:
    st.session_state.generation_complete = False
if 'results' not in st.session_state:
    st.session_state.results = []
if 'current_eval_index' not in st.session_state:
    st.session_state.current_eval_index = 0
if 'scores' not in st.session_state:
    st.session_state.scores = []
if 'evaluation_complete' not in st.session_state:
    st.session_state.evaluation_complete = False


def main():
    st.set_page_config(page_title="LoRA Model Evaluator", layout="wide")
    st.title("🎨 Semi-automatic LoRA Model Evaluator")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    comfyui_url = st.sidebar.text_input("ComfyUI Server", "127.0.0.1:8188")
    
    # Initialize ComfyRunner
    runner = ComfyRunner(server_address=comfyui_url)
    
    # Main workflow tabs
    tab1, tab2, tab3, tab4 = st.tabs(["⚙️ Setup", "🎬 Generation", "📊 Evaluation", "📈 Report"])
    
    # ===== TAB 1: SETUP =====
    with tab1:
        st.header("Step 1: Setup Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("LoRA Configuration")
            lora_filename = st.text_input(
                "LoRA Filename",
                value="blindbox_v1_mix.safetensors",
                help="Enter the LoRA filename (e.g., my_lora.safetensors)"
            )
            
            st.subheader("Weight Range")
            weight_min = st.number_input("Minimum Weight", value=-1.0, step=0.1)
            weight_max = st.number_input("Maximum Weight", value=1.0, step=0.1)
            weight_step = st.number_input("Weight Step", value=0.2, step=0.1, min_value=0.1)
            
            base_seed = st.number_input("Base Seed", value=int(time.time()), step=1)
        
        with col2:
            st.subheader("Prompts")
            prompts_text = st.text_area(
                "Positive Prompts (one per line)",
                value="upperbody shot, 1girl, solo, chibi, long hair, happy, cute\n1boy, portrait, professional, detailed",
                height=150,
                help="Enter each prompt on a new line"
            )
            
            negative_prompt = st.text_area(
                "Negative Prompt (optional)",
                value="(worst quality, low quality:1.4), (bad anatomy), text, error, missing fingers",
                height=100
            )
        
        # Parse prompts
        prompts = [p.strip() for p in prompts_text.split('\n') if p.strip()]
        
        st.info(f"📊 Total generations: {len(prompts)} prompts × {int((weight_max - weight_min) / weight_step) + 1} weights = {len(prompts) * (int((weight_max - weight_min) / weight_step) + 1)} images")
        
        # Start Generation Button
        if st.button("🚀 Start Generation", type="primary", use_container_width=True):
            if not lora_filename:
                st.error("Please enter a LoRA filename")
            elif not prompts:
                st.error("Please enter at least one prompt")
            else:
                st.session_state.generation_complete = False
                st.session_state.results = []
                st.session_state.current_eval_index = 0
                st.session_state.scores = []
                st.session_state.evaluation_complete = False
                
                # Store parameters
                st.session_state.lora_filename = lora_filename
                st.session_state.weight_range = (weight_min, weight_max, weight_step)
                st.session_state.prompts = prompts
                st.session_state.base_seed = base_seed
                st.session_state.negative_prompt = negative_prompt if negative_prompt else None
                
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
            
            # Run generation
            try:
                results = runner.generate_batch(
                    lora_name=st.session_state.lora_filename,
                    weight_range=st.session_state.weight_range,
                    prompts=st.session_state.prompts,
                    base_seed=st.session_state.base_seed,
                    negative_prompt=st.session_state.negative_prompt,
                    progress_callback=progress_callback
                )
                
                st.session_state.results = results
                st.session_state.generation_complete = True
                st.success(f"✅ Generation complete! {len(results)} images created.")
                st.balloons()
                
            except Exception as e:
                st.error(f"❌ Error during generation: {str(e)}")
                st.exception(e)
        
        elif st.session_state.generation_complete:
            st.success(f"✅ Generation complete! {len(st.session_state.results)} images created.")
            
            # Show generated images in a grid
            st.subheader("Generated Images")
            
            if st.session_state.results:
                cols_per_row = 3
                for i in range(0, len(st.session_state.results), cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j, col in enumerate(cols):
                        idx = i + j
                        if idx < len(st.session_state.results):
                            result = st.session_state.results[idx]
                            with col:
                                if os.path.exists(result['image_path']):
                                    img = Image.open(result['image_path'])
                                    st.image(img, use_container_width=True)
                                    st.caption(f"Weight: {result['weight']}")
                                    st.caption(f"Prompt: {result['prompt'][:40]}...")
        else:
            st.info("👈 Please configure parameters in the Setup tab and click 'Start Generation'")
    
    # ===== TAB 3: EVALUATION =====
    with tab3:
        st.header("Step 3: Image Evaluation")
        
        if not st.session_state.generation_complete:
            st.info("⏳ Please complete generation first")
        elif st.session_state.evaluation_complete:
            st.success("✅ Evaluation complete! Check the Report tab for results.")
        else:
            results = st.session_state.results
            current_idx = st.session_state.current_eval_index
            
            if current_idx >= len(results):
                st.session_state.evaluation_complete = True
                
                # Save scores to CSV
                scores_df = pd.DataFrame(st.session_state.scores)
                scores_path = os.path.join(runner.output_dir, "scores.csv")
                scores_df.to_csv(scores_path, index=False)
                
                st.success("✅ All evaluations complete!")
                st.rerun()
            else:
                result = results[current_idx]
                
                st.progress((current_idx + 1) / len(results))
                st.write(f"Image {current_idx + 1} of {len(results)}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Baseline (Weight = 0)")
                    # For baseline, we show the first image with lowest weight or a placeholder
                    baseline_results = [r for r in results if r['weight'] == 0]
                    if baseline_results and os.path.exists(baseline_results[0]['image_path']):
                        baseline_img = Image.open(baseline_results[0]['image_path'])
                        st.image(baseline_img, use_container_width=True)
                    else:
                        st.info("No baseline image with weight=0. Comparing with LoRA effects.")
                
                with col2:
                    st.subheader(f"LoRA (Weight = {result['weight']})")
                    if os.path.exists(result['image_path']):
                        img = Image.open(result['image_path'])
                        st.image(img, use_container_width=True)
                    else:
                        st.error("Image not found")
                
                st.write(f"**Prompt:** {result['prompt']}")
                st.write(f"**Seed:** {result['seed']}")
                
                st.subheader("Is the LoRA image (right) better?")
                
                col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
                
                with col_btn1:
                    if st.button("👍 Yes", use_container_width=True, type="primary"):
                        st.session_state.scores.append({
                            'image_path': result['image_path'],
                            'weight': result['weight'],
                            'prompt': result['prompt'],
                            'seed': result['seed'],
                            'is_better': True
                        })
                        st.session_state.current_eval_index += 1
                        st.rerun()
                
                with col_btn2:
                    if st.button("👎 No", use_container_width=True):
                        st.session_state.scores.append({
                            'image_path': result['image_path'],
                            'weight': result['weight'],
                            'prompt': result['prompt'],
                            'seed': result['seed'],
                            'is_better': False
                        })
                        st.session_state.current_eval_index += 1
                        st.rerun()
                
                with col_btn3:
                    if st.button("⏭️ Skip", use_container_width=True):
                        st.session_state.scores.append({
                            'image_path': result['image_path'],
                            'weight': result['weight'],
                            'prompt': result['prompt'],
                            'seed': result['seed'],
                            'is_better': None
                        })
                        st.session_state.current_eval_index += 1
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
                scored_df = scores_df[scores_df['is_better'].notna()].copy()
                
                if len(scored_df) > 0:
                    # Calculate statistics by weight
                    weight_stats = scored_df.groupby('weight').agg({
                        'is_better': ['sum', 'count', 'mean']
                    }).reset_index()
                    weight_stats.columns = ['weight', 'yes_count', 'total_count', 'yes_rate']
                    
                    # Display statistics
                    st.subheader("📊 Results by Weight")
                    st.dataframe(weight_stats, use_container_width=True)
                    
                    # Line chart
                    st.subheader("📈 Success Rate by Weight")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(weight_stats['weight'], weight_stats['yes_rate'], marker='o', linewidth=2, markersize=8)
                    ax.set_xlabel('LoRA Weight', fontsize=12)
                    ax.set_ylabel('Success Rate (Yes %)', fontsize=12)
                    ax.set_title('LoRA Performance by Weight', fontsize=14, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    ax.set_ylim(0, 1)
                    st.pyplot(fig)
                    
                    # Best weight
                    best_weight = weight_stats.loc[weight_stats['yes_rate'].idxmax(), 'weight']
                    st.success(f"🏆 Best Weight: **{best_weight}** (Success Rate: {weight_stats.loc[weight_stats['yes_rate'].idxmax(), 'yes_rate']:.1%})")
                    
                    # Show best images
                    st.subheader("🌟 Best Rated Images")
                    best_images = scored_df[scored_df['is_better'] == True].nlargest(6, 'weight')
                    
                    cols = st.columns(3)
                    for idx, (_, row) in enumerate(best_images.iterrows()):
                        with cols[idx % 3]:
                            if os.path.exists(row['image_path']):
                                img = Image.open(row['image_path'])
                                st.image(img, use_container_width=True)
                                st.caption(f"Weight: {row['weight']}")
                                st.caption(f"Prompt: {row['prompt'][:40]}...")
                    
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
