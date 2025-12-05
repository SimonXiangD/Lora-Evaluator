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
                st.write(f"Pair {current_idx + 1} of {len(results)}")
                
                # Main layout: images on left (70%), controls on right (30%)
                col_images, col_controls = st.columns([7, 3])
                
                with col_images:
                    # Side-by-side image comparison
                    img_col1, img_col2 = st.columns(2)
                    
                    with img_col1:
                        st.markdown("### 🔵 Without LoRA (Weight = 0)")
                        baseline_path = result['baseline']['image_path']
                        if os.path.exists(baseline_path):
                            baseline_img = Image.open(baseline_path)
                            st.image(baseline_img, use_container_width=True)
                        else:
                            st.error("Baseline image not found")
                    
                    with img_col2:
                        st.markdown(f"### 🟢 With LoRA (Weight = {result['weight']})")
                        lora_path = result['lora']['image_path']
                        if os.path.exists(lora_path):
                            lora_img = Image.open(lora_path)
                            st.image(lora_img, use_container_width=True)
                        else:
                            st.error("LoRA image not found")
                    
                    # Display metadata below images
                    st.write(f"**Prompt:** {result['prompt']}")
                    st.write(f"**Seed:** {result['seed']}")
                
                with col_controls:
                    st.markdown("### 📊 Rate This Pair")
                    st.write("")
                    
                    # Initialize temp rating state for this comparison
                    if 'temp_choice' not in st.session_state:
                        st.session_state.temp_choice = None
                    if 'temp_stars' not in st.session_state:
                        st.session_state.temp_stars = None
                    
                    # Yes/No buttons (vertical)
                    if st.button("👍 LoRA is Better", use_container_width=True, type="primary"):
                        st.session_state.temp_choice = 'yes'
                    
                    if st.button("👎 LoRA is Worse", use_container_width=True):
                        st.session_state.temp_choice = 'no'
                        st.session_state.temp_stars = None  # Clear stars when choosing No
                    
                    if st.button("➖ About the Same", use_container_width=True):
                        st.session_state.temp_choice = 'same'
                        st.session_state.temp_stars = None  # Clear stars when choosing Same
                    
                    st.write("")
                    st.divider()
                    
                    # Show star rating if Yes is selected
                    if st.session_state.temp_choice == 'yes':
                        st.write("**How much better?** (Optional)")
                        
                        # Create 5 star buttons in a row
                        star_cols = st.columns(5)
                        for i, col in enumerate(star_cols):
                            star_num = i + 1
                            with col:
                                # Show filled or empty star based on selection
                                if st.session_state.temp_stars and star_num <= st.session_state.temp_stars:
                                    button_label = "⭐"
                                    button_type = "primary"
                                else:
                                    button_label = "☆"
                                    button_type = "secondary"
                                
                                if st.button(button_label, key=f"star_{star_num}_{current_idx}", use_container_width=True, type=button_type):
                                    st.session_state.temp_stars = star_num
                                    st.rerun()
                        
                        if st.session_state.temp_stars:
                            quality_map = {1: "Slightly", 2: "Slightly", 3: "Moderately", 4: "Significantly", 5: "Dramatically"}
                            st.caption(f"{'⭐' * st.session_state.temp_stars} - {quality_map[st.session_state.temp_stars]} better")
                        else:
                            st.caption("Click a star to rate (optional)")
                    
                    st.write("")
                    st.divider()
                    
                    # Submit button
                    if st.session_state.temp_choice:
                        if st.button("✅ Submit & Next", use_container_width=True, type="secondary"):
                            # Save the score
                            score_entry = {
                                'baseline_path': result['baseline']['image_path'],
                                'lora_path': result['lora']['image_path'],
                                'weight': result['weight'],
                                'prompt': result['prompt'],
                                'seed': result['seed'],
                                'choice': st.session_state.temp_choice,
                                'stars': st.session_state.temp_stars if st.session_state.temp_choice == 'yes' else None,
                                'is_better': True if st.session_state.temp_choice == 'yes' else False if st.session_state.temp_choice == 'no' else None
                            }
                            st.session_state.scores.append(score_entry)
                            
                            # Reset temp state
                            st.session_state.temp_choice = None
                            st.session_state.temp_stars = None
                            
                            # Move to next
                            st.session_state.current_eval_index += 1
                            st.rerun()
                    else:
                        st.info("👆 Please select an option above")
                    
                    st.write("")
                    st.divider()
                    
                    # Skip button at bottom
                    if st.button("⏭️ Skip This Pair", use_container_width=True):
                        st.session_state.scores.append({
                            'baseline_path': result['baseline']['image_path'],
                            'lora_path': result['lora']['image_path'],
                            'weight': result['weight'],
                            'prompt': result['prompt'],
                            'seed': result['seed'],
                            'choice': 'skip',
                            'stars': None,
                            'is_better': None
                        })
                        st.session_state.temp_choice = None
                        st.session_state.temp_stars = None
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
                scored_df = scores_df[scores_df['choice'] != 'skip'].copy()
                
                if len(scored_df) > 0:
                    # Calculate statistics by weight
                    weight_stats = scored_df.groupby('weight').agg({
                        'is_better': ['sum', 'count', 'mean']
                    }).reset_index()
                    weight_stats.columns = ['weight', 'yes_count', 'total_count', 'yes_rate']
                    
                    # Calculate average star rating for "better" choices
                    better_df = scored_df[scored_df['is_better'] == True].copy()
                    if len(better_df) > 0 and 'stars' in better_df.columns:
                        star_stats = better_df.groupby('weight')['stars'].mean().reset_index()
                        star_stats.columns = ['weight', 'avg_stars']
                        weight_stats = weight_stats.merge(star_stats, on='weight', how='left')
                    
                    # Display statistics
                    st.subheader("📊 Results by Weight")
                    st.dataframe(weight_stats, use_container_width=True)
                    
                    # Dual chart: Success Rate and Average Stars
                    st.subheader("📈 Performance by Weight")
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                    
                    # Success rate chart
                    ax1.plot(weight_stats['weight'], weight_stats['yes_rate'], marker='o', linewidth=2, markersize=8, color='#1f77b4')
                    ax1.set_xlabel('LoRA Weight', fontsize=12)
                    ax1.set_ylabel('Success Rate', fontsize=12)
                    ax1.set_title('Success Rate by Weight', fontsize=14, fontweight='bold')
                    ax1.grid(True, alpha=0.3)
                    ax1.set_ylim(0, 1)
                    
                    # Star rating chart (if available)
                    if 'avg_stars' in weight_stats.columns:
                        star_data = weight_stats.dropna(subset=['avg_stars'])
                        if len(star_data) > 0:
                            ax2.plot(star_data['weight'], star_data['avg_stars'], marker='⭐', linewidth=2, markersize=12, color='#ff7f0e')
                            ax2.set_xlabel('LoRA Weight', fontsize=12)
                            ax2.set_ylabel('Average Star Rating', fontsize=12)
                            ax2.set_title('Average Quality Rating (for "Better" choices)', fontsize=14, fontweight='bold')
                            ax2.grid(True, alpha=0.3)
                            ax2.set_ylim(0, 5.5)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Best weight
                    best_weight = weight_stats.loc[weight_stats['yes_rate'].idxmax(), 'weight']
                    best_rate = weight_stats.loc[weight_stats['yes_rate'].idxmax(), 'yes_rate']
                    st.success(f"🏆 Best Weight: **{best_weight}** (Success Rate: {best_rate:.1%})")
                    
                    # Show best images (pairs)
                    st.subheader("🌟 Top Rated Pairs")
                    best_images = scored_df[scored_df['is_better'] == True].copy()
                    if 'stars' in best_images.columns:
                        best_images = best_images.nlargest(6, 'stars')
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
