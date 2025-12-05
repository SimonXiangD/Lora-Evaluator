# Session Management System (v2)

## Overview
The application now supports **multiple independent evaluation sessions**. Each session is stored in its own timestamped folder with all related metadata and generated images.

## Directory Structure
```
./sessions/
├── 20240115_143022/          # Session folder (timestamp format: YYYYMMDD_HHMMSS)
│   ├── session_meta.json     # Session metadata and state
│   └── output/               # Generated images for this session
│       ├── baseline_0_0.png
│       ├── lora_0_0.png
│       └── ...
├── 20240115_150530/          # Another session
│   ├── session_meta.json
│   └── output/
└── ...
```

## Features

### Session Management UI (Sidebar)
Located in the sidebar, the session management panel includes:

- **Current Session Display**: Shows the active session ID and current step
- **🆕 New Button**: Creates a new session with fresh state (clears all progress)
- **💾 Save Button**: Saves current session progress to disk
- **📂 Load Session Dropdown**: 
  - Lists all available sessions sorted by most recent
  - Format: `{session_id} - {lora_filename} (MM/DD HH:MM)`
  - Click "📥 Load Selected" to restore that session

### Session Workflow

1. **Start New Session**
   - Click "🆕 New" to begin a fresh evaluation
   - New session ID is automatically generated (timestamp format)
   - Previous session state is cleared

2. **Setup Phase** (Tab 1)
   - Configure LoRA file, weight range
   - Add prompts (manual or AI-generated)
   - Select evaluation metrics
   - Choose random seeds

3. **Generation Phase** (Tab 2)
   - Images are saved to `./sessions/{session_id}/output/`
   - Each session has its own isolated output folder
   - Generated images persist with the session

4. **Evaluation Phase** (Tab 3)
   - Perform blind A/B comparison
   - Scores are tracked per session
   - Progress is maintained in session state

5. **Report Phase** (Tab 4)
   - View aggregated evaluation results
   - Generate comparison charts and statistics

6. **Save Progress**
   - Click "💾 Save" at any time to persist current state
   - Session metadata includes all configuration and progress

7. **Switch Sessions**
   - Select any previous session from the dropdown
   - Click "📥 Load Selected" to restore
   - **Warning**: Unsaved changes in current session will be lost

### Session Data Structure

Each `session_meta.json` contains:

```json
{
  "session_id": "20240115_143022",
  "timestamp": 1705308622.5,
  "step_1_setup": {
    "lora_filename": "my_lora.safetensors",
    "weight_range": [0.0, 0.5, 1.0],
    "seeds": [898938],
    "metrics": ["Quality", "Style", "Coherence"],
    "prompt_pairs": [...]
  },
  "step_2_generation": {
    "generation_complete": true,
    "results": [...],
    "shuffled_results": [...]
  },
  "step_3_evaluation": {
    "evaluation_complete": false,
    "current_eval_index": 5,
    "current_metric_index": 0,
    "scores": [...]
  },
  "step_4_report": {
    "report_generated": false
  }
}
```

### Key Functions (Internal API)

#### Session Identification
- `get_session_id()`: Returns current or creates new session ID
- `get_session_dir(session_id)`: Returns `./sessions/{session_id}/`
- `get_session_output_dir(session_id)`: Returns `./sessions/{session_id}/output/`

#### Session Management
- `list_available_sessions()`: Lists all sessions with metadata, sorted by timestamp
- `save_session_state(session_id)`: Saves current state to `session_meta.json`
- `load_session_state(session_id)`: Loads specific session into `st.session_state`
- `create_new_session()`: Creates new session ID and clears current state

#### Helper Functions
- `get_current_step()`: Determines current step (1-4) based on progress flags

### ComfyRunner Integration

The backend `ComfyRunner` class now accepts a custom `output_dir` parameter:

```python
# In main()
session_id = get_session_id()
session_output_dir = get_session_output_dir(session_id)
runner = ComfyRunner(
    server_address=comfyui_url, 
    output_dir=session_output_dir
)
```

This ensures all generated images are automatically organized by session folder.

## Migration from v1

**Old System** (v1):
- Single file: `./sessions/current_session.json`
- Auto-loads on startup
- Single session only

**New System** (v2):
- Multiple sessions: `./sessions/{session_id}/session_meta.json`
- Manual session selection via dropdown
- Each session has dedicated output folder
- Images are isolated per session

**Migration Steps**:
1. Old `current_session.json` is no longer used
2. Start a new session using "🆕 New" button
3. Previous data can be manually migrated if needed

## Usage Tips

### Best Practices
- **Save frequently**: Click "💾 Save" after completing each major step
- **Descriptive LoRA names**: Helps identify sessions in the dropdown
- **One session per LoRA**: Create new session for each LoRA evaluation
- **Clean up old sessions**: Manually delete old session folders to save space

### Common Workflows

**Quick Evaluation**:
1. New Session → Setup → Generate → Evaluate → Save

**Comparison Study**:
1. Session A: LoRA v1 → Complete evaluation → Save
2. New Session → Session B: LoRA v2 → Complete evaluation → Save
3. Compare reports from both sessions

**Interrupted Work**:
1. Working on evaluation → App crashes
2. Restart app → Load Session dropdown → Select session
3. Continue from last saved state

### Troubleshooting

**Images not found after loading session**:
- Ensure the session folder still exists in `./sessions/`
- Check that `output/` subfolder contains the images
- Verify ComfyUI output directory matches session path

**Session not appearing in dropdown**:
- Ensure `session_meta.json` exists in the session folder
- Check JSON file is valid (not corrupted)
- Verify timestamp field exists in metadata

**Can't create new session**:
- Check write permissions for `./sessions/` directory
- Ensure sufficient disk space

## Technical Details

### Session ID Format
- Format: `YYYYMMDD_HHMMSS` (e.g., `20240115_143022`)
- Generated using: `time.strftime("%Y%m%d_%H%M%S")`
- Ensures unique IDs when creating sessions seconds apart

### Auto-save Behavior
Unlike v1, v2 does **not** auto-save. Users must manually click "💾 Save" to persist progress.

Future enhancement: Add configurable auto-save intervals.

### Session Isolation
Each session is completely independent:
- Separate metadata file
- Separate output directory
- No shared state between sessions
- Switching sessions clears `st.session_state` and loads new data

### Performance Considerations
- Large sessions (100+ images) may take longer to load
- Images are cached using `@st.cache_data` decorator
- Consider archiving old sessions to external storage

## Future Enhancements

Potential improvements:
- [ ] Session deletion UI
- [ ] Session export/import (zip archive)
- [ ] Session comparison view
- [ ] Auto-save with configurable intervals
- [ ] Session search/filter by LoRA name or date
- [ ] Session notes/descriptions
- [ ] Thumbnail preview in session dropdown
