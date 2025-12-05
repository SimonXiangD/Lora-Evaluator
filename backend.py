"""
Backend module for ComfyUI interaction.
Handles workflow loading, parameter modification, and batch generation.
"""

import json
import os
import time
import urllib.request
import urllib.parse
import uuid
import csv
from datetime import datetime
import websocket
import io
from PIL import Image


class ComfyRunner:
    """Manages interaction with ComfyUI API for LoRA evaluation."""
    
    def __init__(self, server_address="127.0.0.1:8188", workflow_path="workfow_api.json"):
        """
        Initialize ComfyUI runner.
        
        Args:
            server_address: ComfyUI server address (default: 127.0.0.1:8188)
            workflow_path: Path to workflow JSON file
        """
        self.server_address = server_address
        self.workflow_path = workflow_path
        self.workflow = None
        self.client_id = str(uuid.uuid4())
        self.output_dir = "./output"
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_workflow(self):
        """Load workflow from JSON file."""
        with open(self.workflow_path, 'r', encoding='utf-8') as f:
            self.workflow = json.load(f)
        return self.workflow
    
    def queue_prompt(self, prompt):
        """
        Send a prompt to ComfyUI for processing.
        
        Args:
            prompt: Modified workflow dictionary
            
        Returns:
            prompt_id: ID of the queued prompt
        """
        p = {"prompt": prompt, "client_id": self.client_id}
        data = json.dumps(p).encode('utf-8')
        req = urllib.request.Request(f"http://{self.server_address}/prompt", data=data)
        response = urllib.request.urlopen(req)
        return json.loads(response.read())['prompt_id']
    
    def get_image(self, filename, subfolder, folder_type):
        """
        Download generated image from ComfyUI.
        
        Args:
            filename: Name of the image file
            subfolder: Subfolder path
            folder_type: Type of folder (e.g., 'output')
            
        Returns:
            PIL Image object
        """
        data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        url_values = urllib.parse.urlencode(data)
        url = f"http://{self.server_address}/view?{url_values}"
        with urllib.request.urlopen(url) as response:
            return Image.open(io.BytesIO(response.read()))
    
    def get_history(self, prompt_id):
        """
        Get execution history for a prompt.
        
        Args:
            prompt_id: ID of the prompt
            
        Returns:
            History dictionary
        """
        url = f"http://{self.server_address}/history/{prompt_id}"
        with urllib.request.urlopen(url) as response:
            return json.loads(response.read())
    
    def track_progress(self, prompt_id, ws):
        """
        Track progress via WebSocket and return outputs when complete.
        
        Args:
            prompt_id: ID of the prompt to track
            ws: WebSocket connection
            
        Returns:
            Dictionary of outputs when complete, None if failed
        """
        while True:
            out = ws.recv()
            if isinstance(out, str):
                message = json.loads(out)
                if message['type'] == 'executing':
                    data = message['data']
                    if data['node'] is None and data['prompt_id'] == prompt_id:
                        # Execution complete
                        break
            else:
                continue
        
        # Get history to retrieve outputs
        history = self.get_history(prompt_id)
        return history.get(prompt_id, {}).get('outputs', {})
    
    def generate_single(self, lora_name, weight, prompt, seed, negative_prompt=None):
        """
        Generate a single image with specified parameters.
        
        Args:
            lora_name: Name of the LoRA file (e.g., "my_lora.safetensors")
            weight: LoRA weight (strength_model)
            prompt: Positive prompt text
            seed: Random seed for generation
            negative_prompt: Negative prompt text (optional)
            
        Returns:
            Dictionary with image path and metadata, or None if failed
        """
        if self.workflow is None:
            self.load_workflow()
        
        # Clone workflow to avoid modifying original
        workflow_instance = json.loads(json.dumps(self.workflow))
        
        # Modify LoraLoader (Node 11)
        workflow_instance["11"]["inputs"]["lora_name"] = lora_name
        workflow_instance["11"]["inputs"]["strength_model"] = weight
        
        # Modify KSampler (Node 3)
        workflow_instance["3"]["inputs"]["seed"] = seed
        
        # Modify CLIPTextEncode positive prompt (Node 6)
        workflow_instance["6"]["inputs"]["text"] = prompt
        
        # Modify negative prompt if provided (Node 7)
        if negative_prompt:
            workflow_instance["7"]["inputs"]["text"] = negative_prompt
        
        # Queue prompt and track via WebSocket
        ws = websocket.WebSocket()
        ws.connect(f"ws://{self.server_address}/ws?clientId={self.client_id}")
        
        try:
            prompt_id = self.queue_prompt(workflow_instance)
            outputs = self.track_progress(prompt_id, ws)
            
            # Extract image from outputs (SaveImage node is "9")
            if "9" in outputs and "images" in outputs["9"]:
                image_info = outputs["9"]["images"][0]
                image = self.get_image(
                    image_info["filename"],
                    image_info.get("subfolder", ""),
                    image_info.get("type", "output")
                )
                
                # Generate output filename
                safe_prompt = "".join(c if c.isalnum() or c in (' ', '_') else '_' for c in prompt[:30])
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"{safe_prompt}_w{weight}_s{seed}_{timestamp}.png"
                output_path = os.path.join(self.output_dir, output_filename)
                
                # Save image
                image.save(output_path)
                
                return {
                    "image_path": output_path,
                    "lora_name": lora_name,
                    "weight": weight,
                    "prompt": prompt,
                    "seed": seed,
                    "timestamp": timestamp
                }
            else:
                print(f"No image output found for prompt_id: {prompt_id}")
                return None
                
        except Exception as e:
            print(f"Error generating image: {e}")
            return None
        finally:
            ws.close()
    
    def generate_batch(self, lora_name, weight_range, prompts, base_seed=None, negative_prompt=None, progress_callback=None):
        """
        Generate a batch of images for weight x prompt combinations.
        
        Args:
            lora_name: Name of the LoRA file
            weight_range: Tuple of (min, max, step) for weights
            prompts: List of prompt strings
            base_seed: Base seed (will increment for each generation)
            negative_prompt: Negative prompt text (optional)
            progress_callback: Function to call with progress updates (current, total)
            
        Returns:
            List of result dictionaries
        """
        if base_seed is None:
            base_seed = int(time.time())
        
        # Generate weight values
        min_weight, max_weight, step = weight_range
        weights = []
        current = min_weight
        while current <= max_weight:
            weights.append(round(current, 2))
            current += step
        
        total_generations = len(weights) * len(prompts)
        results = []
        
        # Initialize CSV log
        log_path = os.path.join(self.output_dir, "generation_log.csv")
        csv_exists = os.path.exists(log_path)
        
        with open(log_path, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['timestamp', 'image_path', 'lora_name', 'weight', 'prompt', 'seed']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            if not csv_exists:
                writer.writeheader()
            
            current_generation = 0
            seed = base_seed
            
            for weight in weights:
                for prompt in prompts:
                    current_generation += 1
                    
                    if progress_callback:
                        progress_callback(current_generation, total_generations)
                    
                    result = self.generate_single(lora_name, weight, prompt, seed, negative_prompt)
                    
                    if result:
                        results.append(result)
                        writer.writerow(result)
                        csvfile.flush()  # Ensure data is written immediately
                    
                    seed += 1
                    time.sleep(0.5)  # Small delay to avoid overwhelming ComfyUI
        
        return results
    
    def generate_baseline(self, prompt, seed, negative_prompt=None):
        """
        Generate a baseline image without LoRA (weight=0).
        
        Args:
            prompt: Positive prompt text
            seed: Random seed
            negative_prompt: Negative prompt text (optional)
            
        Returns:
            Dictionary with image path and metadata
        """
        return self.generate_single("none.safetensors", 0.0, prompt, seed, negative_prompt)
