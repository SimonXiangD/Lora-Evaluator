# Session Management Guide

## 🎯 Overview

The LoRA Evaluator now includes automatic session saving and restoration, allowing you to:
- **Resume interrupted work**: If the app crashes or closes, your progress is saved
- **Continue later**: Come back and pick up exactly where you left off
- **No data loss**: Every step is automatically saved

## 📂 How It Works

### Automatic Saving
The app automatically saves your progress at these key moments:

1. **Setup Phase**: When you click "Start Generation"
2. **Generation Phase**: When image generation completes
3. **Evaluation Phase**: 
   - Every 5 evaluations (auto-save)
   - When you skip an evaluation
   - When you click "Skip to End"
   - When all evaluations are complete

### Auto-Restore on Startup
When you launch the app, it automatically:
1. Checks for saved session in `./sessions/current_session.json`
2. Restores all your previous state if found
3. Shows you which step you're on
4. Lets you continue from where you left off

## 💾 Session Data Structure

The session file stores:

### Step 1: Setup
- LoRA filename
- Weight range (min, max, step)
- Seeds list
- Evaluation metrics
- Prompt pairs

### Step 2: Generation
- Generation status (complete/in-progress)
- Generated image results
- Shuffled evaluation order
- Generation parameters

### Step 3: Evaluation
- Evaluation progress (current pair and metric)
- All scores submitted so far
- Evaluation completion status

### Step 4: Report
- Report generation status

## 🎮 Manual Controls

In the sidebar, you'll find:

### 💾 Save Button
- Manually save your current progress
- Useful before closing the app
- Shows success message when saved

### 🗑️ Reset Button
- Clear all saved data
- Start completely fresh
- ⚠️ This deletes all progress!

### Current Step Indicator
Shows which step you're currently on:
- Step 1: Setup
- Step 2: Generation
- Step 3: Evaluation
- Step 4: Report

### Last Saved Timestamp
Displays when the session was last saved

## 📁 File Location

Session data is stored in:
```
./sessions/current_session.json
```

This file contains all your progress in JSON format.

## 🔄 Typical Workflow

### First Time
1. Open app → Setup parameters
2. Click "Start Generation" → **Auto-saved**
3. Generation completes → **Auto-saved**
4. Start evaluation → Auto-saves every 5 evaluations
5. Complete evaluation → **Auto-saved**
6. View report

### After Crash/Close
1. Open app → **Auto-restores** previous session
2. See info message: "📂 Session restored from previous run"
3. Continue from exact point where you left off
4. Current step shown in sidebar

### Starting Over
1. Click "🗑️ Reset" in sidebar
2. All progress cleared
3. Start fresh with new parameters

## 🛡️ Safety Features

- **Auto-save frequency**: Every 5 evaluations during the evaluation phase
- **Backup on critical operations**: Before starting generation, after completion
- **Error handling**: If save fails, app continues normally
- **Non-blocking**: Saving happens in background, doesn't slow down the app

## 💡 Tips

1. **Manual save before closing**: Click 💾 Save before you exit
2. **Check timestamp**: Verify "Last saved" time in sidebar
3. **Reset when needed**: If things seem stuck, use Reset button
4. **Don't delete sessions folder**: Keep `./sessions/` for auto-restore to work

## 🐛 Troubleshooting

**Q: Session not restoring?**
- Check if `./sessions/current_session.json` exists
- Try manual save button
- Check console for error messages

**Q: Want to start completely fresh?**
- Click "🗑️ Reset" button in sidebar
- Or manually delete `./sessions/` folder

**Q: Can I share my session?**
- Yes! The JSON file is portable
- Copy `current_session.json` to another machine
- Place in `./sessions/` folder before launching app

**Q: How much disk space does it use?**
- Very small! Only ~10-50KB per session
- Contains metadata, not images
- Images stay in `./output/` folder

## 🔧 Advanced: Session File Format

Example structure:
```json
{
  "timestamp": 1701734400.0,
  "step_1_setup": {
    "lora_filename": "my_lora.safetensors",
    "weight_range": [0.0, 1.0, 0.5],
    "seeds": [123456, 789012],
    "metrics": ["Visual Quality", "Style Consistency"],
    "prompt_pairs": [...]
  },
  "step_2_generation": {
    "generation_complete": true,
    "results": [...],
    "shuffled_results": [...]
  },
  "step_3_evaluation": {
    "evaluation_complete": false,
    "current_eval_index": 15,
    "current_metric_index": 0,
    "scores": [...]
  },
  "step_4_report": {
    "report_generated": false
  }
}
```

## 📊 Benefits

✅ **Never lose progress** - Work at your own pace  
✅ **Resume anytime** - Close and come back later  
✅ **Auto-backup** - No manual intervention needed  
✅ **Fast restore** - Instant resume on startup  
✅ **Portable** - Move sessions between machines  
✅ **Transparent** - Always know what's saved  

---

**Note**: This feature requires write permissions to the application directory. Make sure `./sessions/` folder can be created.
