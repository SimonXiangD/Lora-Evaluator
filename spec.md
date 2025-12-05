# Technical Specification: Semi-automatic LoRA Model Evaluator

## 1. Project Overview
**Goal:** A Streamlit-based tool to evaluate LoRA models using local ComfyUI as the backend.
**Workflow:**
1.  **Setup:** User inputs LoRA filename, weight range (e.g., -1.0 to 1.0), and Prompts.
2.  **Generation:** System iterates through `Weight x Prompt` combinations, sending requests to ComfyUI API.
3.  **Scoring:** User compares "Base Model Image" vs "LoRA Image" side-by-side.
4.  **Report:** System generates a summary of best weights and prompts.

## 2. Infrastructure
* **ComfyUI URL:** `http://127.0.0.1:8188`
* **Input File:** `workflow_api.json` (located in root).
* **Output Folder:** `./output/` (stores images and CSV logs).

## 3. Core Modules

### A. `backend.py` (The Controller)
* **Class `ComfyRunner`**:
    * Connects to ComfyUI via WebSocket.
    * Methods needed:
        * `load_workflow()`: Reads `workflow_api.json`.
        * `generate_batch(params)`: Loops through weights/prompts.
        * **Crucial Logic:** It must identify Node IDs in the JSON.
            * Find "LoraLoader" to set `strength_model` and `lora_name`.
            * Find "KSampler" to set `seed`.
            * Find "CLIPTextEncode" to set `text` (prompt).
    * Saves images to `./output/` with filenames like `promptA_w0.8_seed123.png`.
    * Saves metadata to `./output/generation_log.csv`.

### B. `app.py` (The Frontend)
* **Framework:** Streamlit.
* **Step 1 (Setup):**
    * Input: LoRA Filename (string, e.g., "my_lora.safetensors").
    * Input: Base Model Name (dropdown or string).
    * Input: Weight Range (min, max, step).
    * Input: Prompt List (text area).
    * Action: "Start Generation" button.
* **Step 2 (Progress):**
    * Show progress bar while `backend.generate_batch` runs.
* **Step 3 (Evaluation):**
    * Layout: Two columns.
        * Left: Baseline Image (Weight=0 or bypassed).
        * Right: LoRA Image (Weight=X).
    * Input: "Is Right Better?" (Yes/No buttons).
    * Logic: Save user vote to `./output/scores.csv`.
* **Step 4 (Report):**
    * Show a line chart: X-axis = Weight, Y-axis = "Yes" Rate.
    * Show best images.

## 4. Dependencies
* `streamlit`
* `pandas`
* `requests`
* `websocket-client`
* `matplotlib`