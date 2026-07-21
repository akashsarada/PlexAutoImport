import os
import shutil
import time
import uuid

# --- CONFIGURATION ---
SRC_DIR = r'C:\Test'
DEST_DIR = r'D:\Test'
# ---------------------

def cleanup_deleteme_files(directory):
    """Final pass to remove any files marked for deletion."""
    print("\n--- Running Cleanup Pass ---")
    current_files = list(os.listdir(directory))
    count = 0
    
    for file in current_files:
        if file.endswith(".deleteme"):
            path = os.path.join(directory, file)
            try:
                os.remove(path)
                print(f"Cleanup: Removed {file}")
                count += 1
            except PermissionError:
                # If still locked, we leave it for the next run
                pass
    print(f"Cleanup finished. Removed {count} files.")

def process_test_move():
    if not os.path.exists(SRC_DIR):
        print(f"Source {SRC_DIR} not found.")
        return
    
    if not os.path.exists(DEST_DIR):
        os.makedirs(DEST_DIR)
        print(f"Created destination: {DEST_DIR}")

    # Snapshot the directory list
    files = list(os.listdir(SRC_DIR))
    
    for file in files:
        src_path = os.path.join(SRC_DIR, file)
        
        # Skip folders and already-marked trash files
        if not os.path.isfile(src_path) or file.endswith(".deleteme"):
            continue

        dest_path = os.path.join(DEST_DIR, file)
        print(f"Moving: {file}")

        try:
            # 1. Collision Check on D:\Test
            if os.path.exists(dest_path):
                base, ext = os.path.splitext(file)
                dest_path = os.path.join(DEST_DIR, f"{base}_{uuid.uuid4().hex[:6]}{ext}")
                print(f"  - Destination conflict. Renamed to: {os.path.basename(dest_path)}")

            # 2. Copy (Since it's cross-drive)
            shutil.copy2(src_path, dest_path)
            print("  - Copy successful.")

            # 3. Rename Source to "Kill" the file handle
            # This is the secret sauce for Windows locks
            delete_path = src_path + ".deleteme"
            
            # If an old .deleteme file exists, try to clear it first
            if os.path.exists(delete_path):
                try: os.remove(delete_path)
                except: pass

            os.rename(src_path, delete_path)
            print("  - Marked for deletion.")

            # 4. Try immediate delete
            try:
                os.remove(delete_path)
                print("  - Deleted successfully.")
            except PermissionError:
                print("  - Immediate delete blocked (lock persistent). Deferring.")

        except Exception as e:
            print(f"  - FAILED to process {file}: {e}")

    # Run the final sweep
    cleanup_deleteme_files(SRC_DIR)

if __name__ == "__main__":
    process_test_move()