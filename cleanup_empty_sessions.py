"""
Clean up empty session folders in the sessions directory.
"""

import os
import shutil

SESSIONS_DIR = "./sessions"

def cleanup_empty_sessions():
    """Remove empty session folders that have no files."""
    if not os.path.exists(SESSIONS_DIR):
        print("Sessions directory does not exist.")
        return
    
    removed_count = 0
    
    for folder_name in os.listdir(SESSIONS_DIR):
        folder_path = os.path.join(SESSIONS_DIR, folder_name)
        
        # Skip if not a directory
        if not os.path.isdir(folder_path):
            continue
        
        # Check if folder is empty (including subdirectories)
        has_files = False
        for root, dirs, files in os.walk(folder_path):
            if files:
                has_files = True
                break
        
        # Remove if empty
        if not has_files:
            print(f"Removing empty session: {folder_name}")
            shutil.rmtree(folder_path)
            removed_count += 1
    
    print(f"\nCleanup complete. Removed {removed_count} empty session folders.")

if __name__ == "__main__":
    cleanup_empty_sessions()
