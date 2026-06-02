import os
import sys
import re
import time
import json
import subprocess
import urllib.parse
import tempfile
from datetime import datetime
from collections import deque

import yt_dlp
from faster_whisper import WhisperModel
import requests
from bs4 import BeautifulSoup

# =================
#   Config Values
# =================
BASE_DIR = "Data_Collection"
CLEAN_DIR = os.path.join(BASE_DIR, "clean_data")
LORA_DIR = os.path.join(BASE_DIR, "LoRA_Transcripts")

#Plans to change where these are put later
MEMORY_FILE = os.path.join(BASE_DIR, "known_speakers.json")
EXCLUSION_FILE = os.path.join(BASE_DIR, "exclusion_list.json")

os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(CLEAN_DIR, exist_ok=True)
os.makedirs(LORA_DIR, exist_ok=True)

CONFIG = {
    "max_pages": 50,
    "crawl_delay": 1.0, 
    "chunk_duration": 15,
    "whisper_vod_model": "medium.en",
    "whisper_live_model": "small.en"
}
LOADED_MODELS = {}

# Heuristic mappings for dynamic action injection
ACTION_KEYWORDS = {
    "Smile": [r"\b(happy|glad|awesome|nice|good|cool|love|congrats|congratulations)\b", r":\)"],
    "Laugh": [r"\b(haha|lol|lmao|griefing|funny|xd)\b"],
    "Blink": [r"\b(what|look|see|view|drone|glitch)\b", r"\?"],
    "Wave": [r"\b(hello|hi|hey|goodbye|bye|see ya|welcome)\b"],
    "Sigh": [r"\b(pain|die|sorry|sad|nonsense|fault|stuck|hurt|lost)\b"]
}

def get_whisper_engine(config_key):
    model_name = CONFIG[config_key]
    if model_name not in LOADED_MODELS:
        print(f"\n[ SYSTEM ] Spinning up Whisper Model '{model_name}'...")
        try:
            print("  -> Testing GPU (CUDA) hardware initialization...")
            test_model = WhisperModel(model_name, device="cuda", compute_type="float16")
            import numpy as np
            dummy_silence = np.zeros(4000, dtype=np.float32) 
            list(test_model.generate_segments(dummy_silence))
            LOADED_MODELS[model_name] = test_model
            print("  -> GPU Acceleration verified successfully.")
        except Exception as e:
            print(f" -> GPU/CUDA unavailable or library missing (Error: {e})")
            print("  -> Falling back to standard CPU thread allocation mode.")
            LOADED_MODELS[model_name] = WhisperModel(model_name, device="cpu", compute_type="int8")
    return LOADED_MODELS[model_name]

def sanitize_name(text):
    if not text: 
        return "Unknown"
    return re.sub(r'[\\/*?:"<>|]', "", str(text)).replace(" ", "_")

# =========================================
#   Dynamic Storage Configuration Engines
# =========================================
def load_speaker_memory():
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    return {"bot_name": "Cato", "tracked_humans": []}

def save_speaker_memory(memory):
    with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(memory, f, indent=4)

def load_exclusion_list():
    """
    Loads custom filters from JSON file in the master folder.
    Generates a default set if the configuration file is missing.
    """
    default_exclusions = [
        "A", "The", "I", "You", "What", "Oh", "City", "Los", "Santos", "There", "Here", 
        "This", "That", "On", "Checking", "Fine", "Good", "Talking", "Where", "Singing", 
        "Swinging", "Shooting", "Going", "True", "All", "Earn", "Dead", "Living", "About", 
        "Holding", "Such", "Actually", "Just", "Like", "How", "We", "They", "He", "She", 
        "It", "My", "Your", "Our", "And", "But", "Or", "If", "Because", "As", "Until", 
        "While", "Of", "At", "By", "For", "With", "Against", "Between", "Into", "Yeah",
        "Through", "During", "Before", "After", "Above", "Below", "To", "From", "Up", 
        "Down", "In", "Out", "Off", "Over", "Under", "Again", "Further", "Then", "Once"
    ]
    
    if os.path.exists(EXCLUSION_FILE):
        try:
            with open(EXCLUSION_FILE, 'r', encoding='utf-8') as f:
                return set(json.load(f))
        except:
            pass
            
    with open(EXCLUSION_FILE, 'w', encoding='utf-8') as f:
        json.dump(default_exclusions, f, indent=4)
    return set(default_exclusions)

def inject_dynamic_action(text):
    for action, patterns in ACTION_KEYWORDS.items():
        for pattern in patterns:
            if re.search(pattern, text.lower()):
                return f' {{"action": "{action}"}}'
    return ""

def discover_names(text, memory, exclusion_set):
    intro_patterns = [
        r"\b[Ii]'m\s+([A-Z][a-z]+)", 
        r"\b[Tt]his\s+is\s+([A-Z][a-z]+)",
        r"\b[Hh]i\b,?\s+([A-Z][a-z]+)",
        r"\b[Hh]ey\b,?\s+([A-Z][a-z]+)"
    ]
    for pattern in intro_patterns:
        matches = re.findall(pattern, text)
        for name in matches:
            if name not in exclusion_set:
                if name != memory["bot_name"] and name not in memory["tracked_humans"]:
                    print(f"   [ IDENTITY DISCOVERY ] Learned new speaker: '{name}'")
                    memory["tracked_humans"].append(name)
                    save_speaker_memory(memory)

def run_dataset_cleaner():
    print("\n=======================================================")
    print("          LOCAL TRANSCRIPT CLEANING ENGINE             ")
    print("=======================================================")
    
    txt_files = []
    
    for root, dirs, files in os.walk(BASE_DIR):
        if "clean_data" in root or "LoRA_Transcripts" in root:
            continue
            
        for file in files:
            if file.endswith(".txt"):
                full_path = os.path.join(root, file)
                display_path = os.path.relpath(full_path, BASE_DIR)
                txt_files.append((display_path, full_path, file))
    
    if not txt_files:
        print(f"[!] No raw text transcript files found inside '{BASE_DIR}'.")
        return

    print("Available transcripts found across all platforms:")
    for idx, (display_path, _, _) in enumerate(txt_files, start=1):
        print(f" [{idx}] {display_path}")
        
    try:
        selection = input("\nSelect file index to clean (or press Enter to cancel): ").strip()
        if not selection: return
        
        selected_idx = int(selection) - 1
        if selected_idx < 0 or selected_idx >= len(txt_files):
            print("[!] Invalid index selected.")
            return
            
        display_path, input_path, original_filename = txt_files[selected_idx]
    except ValueError:
        print("[!] Invalid index selected.")
        return

    flattened_name = display_path.replace(os.sep, "_").replace("/", "_").replace("\\", "_")
    output_filename = f"CLEAN_{flattened_name}"
    output_path = os.path.join(CLEAN_DIR, output_filename)

    memory = load_speaker_memory()
    exclusion_set = load_exclusion_list()
    bot_name = memory["bot_name"]
    
    print(f"\n[ SYSTEM ] Optimizing transcript: {display_path}")
    print(f"[ SYSTEM ] Exporting Unique Output Target: {output_filename}")

    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    cleaned_lines = []
    for line in lines:
        c = re.sub(r'\[source:\s*\d+\]', '', line).strip()
        if c:
            cleaned_lines.append(c)

    with open(output_path, 'w', encoding='utf-8') as out:
        is_bot_turn = False
        
        for text_line in cleaned_lines:
            discover_names(text_line, memory, exclusion_set)
            
            if is_bot_turn:
                speaker = bot_name
            else:
                speaker = memory["tracked_humans"][-1] if memory["tracked_humans"] else "User"
            
            action_payload = inject_dynamic_action(text_line)
            out.write(f"{speaker}: {text_line}{action_payload}\n")
            is_bot_turn = not is_bot_turn

    print(f"\n>>> SUCCESS: Cleaned data exported cleanly.")
    print(f"    Saved -> {output_path}")

# ==================================
#   LoRA Master Compilation Engine
# ==================================
def run_lora_compiler():
    print("\n=======================================================")
    print("             LORA MASTER DATA COMPILER                 ")
    print("=======================================================")
    
    if not os.path.exists(CLEAN_DIR):
        print("[!] Clean data folder does not exist yet.")
        return
        

    clean_files = [f for f in os.listdir(CLEAN_DIR) if f.endswith(".txt") and f.startswith("CLEAN_")]
    
    if not clean_files:
        print(f"[!] No valid cleaned files found inside '{CLEAN_DIR}'.")
        return

    selected_files = []
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        print("=======================================================")
        print("          SELECT CLEAN FILES FOR LORA BATCH            ")
        print("=======================================================")
        
        for idx, file_name in enumerate(clean_files, start=1):
            status = "[ SELECTED ]" if file_name in selected_files else "[   ---    ]"
            print(f" {status} [{idx}] {file_name}")
            
        print("-------------------------------------------------------")
        print(" * Enter a number to toggle selection.")
        print(" * Type 'DONE' to compile selected files.")
        print(" * Press Enter with no input to cancel out.")
        print("=======================================================")
        
        choice = input("Selection input: ").strip()
        
        if not choice:
            return
        if choice.upper() == "DONE":
            if not selected_files:
                print("\n[!] You haven't selected any files yet!")
                time.sleep(2)
                continue
            break
            
        try:
            target_idx = int(choice) - 1
            chosen_file = clean_files[target_idx]
            if chosen_file in selected_files:
                selected_files.remove(chosen_file)
            else:
                selected_files.append(chosen_file)
        except (ValueError, IndexError):
            print("\n[!] Selection index completely out of range.")
            time.sleep(1)

    current_date = datetime.now().strftime("%Y-%m-%d")
    master_lora_filename = f"LoRA_{current_date}.txt"
    final_output_path = os.path.join(LORA_DIR, master_lora_filename)
    
    print(f"\n[ COMPILING ] Merging {len(selected_files)} transcripts sequentially...")
    
    try:
        with open(final_output_path, "w", encoding="utf-8") as master_out:
            for clean_file in selected_files:
                file_path = os.path.join(CLEAN_DIR, clean_file)
                master_out.write(f"\n# --- START OF FILE: {clean_file} ---\n")
                with open(file_path, "r", encoding="utf-8") as infile:
                    master_out.write(infile.read())
                master_out.write(f"\n# --- END OF FILE: {clean_file} ---\n")
                
        print(f"\n>>> CONSOLIDATION COMPLETE! Master fine-tuning asset created successfully.")
        print(f"    Saved -> {final_output_path}")
        input("\nPress Enter to return back to dashboard setup...")
    except Exception as e:
        print(f"[ CRITICAL ERROR ] Failed compilation block build: {e}")
        input("\nPress Enter to return...")

# =========================================
#   Stubs for Existing Functional Modules
# =========================================
def run_media_scraper():
    print("\n[ Stub ] Running Media VOD Scraper...")
    time.sleep(1)

def run_live_chunker():
    print("\n[ Stub ] Running Live Stream Audio Chunker...")
    time.sleep(1)

def run_documentation_crawler():
    print("\n[ Stub ] Running Documentation Crawler...")
    time.sleep(1)

def run_settings_menu():
    print("\n[ Stub ] Accessing Settings Parameters...")
    time.sleep(1)

# =================
#   Control Panel
# =================
if __name__ == "__main__":
    while True:
        print("\n=======================================================")
        print("          UNIFIED DATA ACQUISITION ENGINE              ")
        print("=======================================================")
        print(" [1] Scrape Media VOD   - Extract & Transcribe Video (YT/Twitch)")
        print(" [2] Chunk Live Stream  - Record & Transcribe Live Feed Content")
        print(" [3] Crawl Doc Website  - Deep-Scrape Code Documentation Safely")
        print(" [4] Clean Transcripts  - Format LoRA Dataset & Learn Names (Cato)")
        print(" [5] Make LoRA Dataset  - Combine Clean Files into Master Dataset")
        print(" [6] Adjust Settings    - Configure Pipeline Hyperparameters")
        print(" [7] Exit Program       - Shut down interface")
        print("=======================================================")
        main_choice = input("Select operation parameter (1-7): ").strip()
        
        if main_choice == "1": run_media_scraper()
        elif main_choice == "2": run_live_chunker()
        elif main_choice == "3": run_documentation_crawler()
        elif main_choice == "4": run_dataset_cleaner()
        elif main_choice == "5": run_lora_compiler()
        elif main_choice == "6": run_settings_menu()
        elif main_choice == "7": 
            print("\nShutting down pipeline components safely.")
            break
