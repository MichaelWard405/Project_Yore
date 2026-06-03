import os
import sys
import re
import time
import json
import subprocess
import urllib.parse
import tempfile
import uuid
from datetime import datetime
import numpy as np

import yt_dlp
from faster_whisper import WhisperModel

# ========================================
#   Environment & Directory Architecture
# ========================================
BASE_DIR = "Data_Collection"
CLEAN_DIR = os.path.join(BASE_DIR, "clean_data")
LORA_DIR = os.path.join(BASE_DIR, "LoRA_Transcripts")
TEMP_DIR = os.path.join(BASE_DIR, "temp_downloads")
BIN_TRANSCRIPTS_DIR = os.path.join(BASE_DIR, "Binary_Transcripts")
LOCAL_DATA_DIR = os.path.join(BASE_DIR, "Local_Data")
NEURAL_DIR = "Neural_Net"
TOKENIZER_PATH = os.path.join(NEURAL_DIR, "custom_bpe.json")
VOCAB_SIZE = 32000

def ensure_environment():
    for directory in [BASE_DIR, CLEAN_DIR, LORA_DIR, TEMP_DIR, BIN_TRANSCRIPTS_DIR, LOCAL_DATA_DIR, NEURAL_DIR]:
        os.makedirs(directory, exist_ok=True)

ensure_environment()

MEMORY_FILE = os.path.join(BASE_DIR, "known_speakers.json")
EXCLUSION_FILE = os.path.join(BASE_DIR, "exclusion_list.json")

CONFIG = { 
    "max_pages": 50,
    "crawl_delay": 1.0,
    "chunk_duration": 15,
    "whisper_vod_model": "medium.en",
    "whisper_live_model": "small.en"
}
LOADED_MODELS = {}

ACTION_KEYWORDS = {
    "Smile": [r"\b(happy|glad|awesome|nice|good|cool|love|congrats|congratulations)\b", r":\)"],
    "Laugh": [r"\b(haha|lol|lmao|griefing|funny|xd|clown|clowns)\b"],
    "Blink": [r"\b(what|look|see|view|drone|glitch|pumpkin|stuck|hp)\b", r"\?"],
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
            dummy_audio = np.zeros(16000, dtype=np.float32) 
            segments, _ = test_model.transcribe(dummy_audio)
            list(segments)
            LOADED_MODELS[model_name] = test_model
            print("  -> GPU Acceleration verified successfully.")
        except Exception as e:
            print(f"  -> GPU/CUDA unavailable or library missing (Error: {e})")
            print("  -> Falling back safely to standard CPU thread allocation mode.")
            LOADED_MODELS[model_name] = WhisperModel(model_name, device="cpu", compute_type="int8")
    return LOADED_MODELS[model_name]

def extract_text_from_pdf(pdf_path):
    try:
        import pypdf
        reader = pypdf.PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted: text += extracted + "\n"
        return text
    except ImportError:
        print("\n[ ERROR ] Local PDF extraction requires the pypdf parser library.")
        print(" -> Please install it via terminal: pip install pypdf")
        return None

def sanitize_name(text):
    if not text: return "Unknown"
    return re.sub(r'[\\/*?:"<>|]', "", str(text)).replace(" ", "_").strip()

def load_speaker_memory():
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, 'r', encoding='utf-8') as f: return json.load(f)
        except: pass
    return {"bot_name": "Cato", "tracked_humans": []}

def save_speaker_memory(memory):
    with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(memory, f, indent=4)

def load_exclusion_list():
    default_exclusions = ["A", "The", "I", "You", "What", "Oh", "City", "There", "Here", "This", "That", "On", "Checking", "Fine", "Good", "Talking", "Where", "Singing", "Swinging", "Shooting", "Going", "True", "All", "Earn", "Dead", "Living", "About", "Holding", "Such", "Actually", "Just", "Like", "How", "We", "They", "He", "She", "It", "My", "Your", "Our", "And", "But", "Or", "If", "Because", "As", "Until", "While", "Of", "At", "By", "For", "With", "Against", "Between", "Into", "Yeah", "Through", "During", "Before", "After", "Above", "Below", "To", "From", "Up", "Down", "In", "Out", "Off", "Over", "Under", "Again", "Further", "Then", "Once"]
    if os.path.exists(EXCLUSION_FILE):
        try:
            with open(EXCLUSION_FILE, 'r', encoding='utf-8') as f: return set(json.load(f))
        except: pass
    with open(EXCLUSION_FILE, 'w', encoding='utf-8') as f:
        json.dump(default_exclusions, f, indent=4)
    return set(default_exclusions)

def inject_dynamic_action(text):
    for action, patterns in ACTION_KEYWORDS.items():
        for pattern in patterns:
            if re.search(pattern, text.lower()): return f' {{"action": "{action}"}}'
    return ""

def discover_names(text, memory, exclusion_set):
    intro_patterns = [r"\b[Ii]'m\s+([A-Z][a-z]+)", r"\b[Tt]his\s+is\s+([A-Z][a-z]+)", r"\b[Hh]i\b,?\s+([A-Z][a-z]+)", r"\b[Hh]ey\b,?\s+([A-Z][a-z]+)"]
    for pattern in intro_patterns:
        matches = re.findall(pattern, text)
        for name in matches:
            if name not in exclusion_set and name != memory["bot_name"] and name not in memory["tracked_humans"]:
                print(f"   [ IDENTITY DISCOVERY ] Learned new speaker: '{name}'")
                memory["tracked_humans"].append(name)
                save_speaker_memory(memory)

def run_dataset_cleaner():
    print("\n=======================================================")
    print("      LOCAL TRANSCRIPT CLEANING ENGINE (JSONL MODE)    ")
    print("=======================================================")
    
    txt_files = []
    for root, dirs, files in os.walk(BASE_DIR):
        if any(d in root for d in ["clean_data", "LoRA_Transcripts", "temp_downloads", "Binary_Transcripts", "Local_Data"]):
            continue
        for file in files:
            if file.endswith(".txt"):
                full_path = os.path.join(root, file)
                txt_files.append((os.path.relpath(full_path, BASE_DIR), full_path, file))
    
    if not txt_files:
        print(f"[!] No raw text transcript files found inside '{BASE_DIR}'.")
        return

    print("Available transcripts found across all platforms:")
    for idx, (display_path, _, _) in enumerate(txt_files, start=1):
        print(f" [{idx}] {display_path}")
        
    try:
        selection = input("\nSelect file index to clean (or press Enter to cancel): ").strip()
        if not selection: return
        display_path, input_path, original_filename = txt_files[int(selection) - 1]
    except:
        return print("[!] Invalid index selected.")

    base_name = os.path.splitext(display_path.replace(os.sep, "_").replace("/", "_").replace("\\", "_"))[0]
    output_path_jsonl = os.path.join(CLEAN_DIR, f"CLEAN_{base_name}.jsonl")
    output_path_txt = os.path.join(CLEAN_DIR, f"CLEAN_{base_name}.txt")

    memory = load_speaker_memory()
    exclusion_set = load_exclusion_list()
    
    print(f"\n[ SYSTEM ] Optimizing transcript: {display_path}")

    with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
        raw_text = f.read().replace('\x00', '')

    cleaned_lines = []
    for line in raw_text.splitlines():
        clean_line = re.sub(r'\\', '', line).strip()
        clean_line = clean_line.replace('"', '').strip() 
        if clean_line: cleaned_lines.append(clean_line)

    with open(output_path_jsonl, 'w', encoding='utf-8') as out_jsonl, open(output_path_txt, 'w', encoding='utf-8') as out_txt:
        is_bot_turn = False
        for text_line in cleaned_lines:
            discover_names(text_line, memory, exclusion_set)
            speaker = memory["bot_name"] if is_bot_turn else (memory["tracked_humans"][-1] if memory["tracked_humans"] else "User")
            combined_message = f"{text_line}{inject_dynamic_action(text_line)}"
            
            jsonl_entry = {
                "text": f"{speaker}: {combined_message}",
                "metadata": {"speaker": speaker, "target_user": memory["tracked_humans"][-1] if memory["tracked_humans"] else "Unknown", "location": "Brisbane", "topic": "General"}
            }
            out_jsonl.write(json.dumps(jsonl_entry) + "\n")
            out_txt.write(combined_message + "\n")
            is_bot_turn = not is_bot_turn

    print(f"\n>>> SUCCESS: Cleaned data exported to JSONL (LoRA) and TXT (Base Pretrain).")

def run_bin_compiler():
    print("\n=======================================================")
    print("      BASE MODEL .BIN COMPILER (MEMORY MAP)            ")
    print("=======================================================")
    
    clean_files = [f for f in os.listdir(CLEAN_DIR) if f.endswith(".txt") and f.startswith("CLEAN_")]
    if not clean_files: return print(f"[!] No valid TXT files found in '{CLEAN_DIR}'.")

    selected_files = []
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        print("=======================================================")
        for idx, file_name in enumerate(clean_files, start=1):
            status = "[ SELECTED ]" if file_name in selected_files else "[   ---    ]"
            print(f" {status} [{idx}] {file_name}")
        print("-------------------------------------------------------")
        print(" * Enter a number to toggle selection. Type 'DONE' to compile.")
        choice = input("Selection input: ").strip()
        
        if not choice: return
        if choice.upper() == "DONE":
            if selected_files: break
            time.sleep(1)
            continue
        try:
            chosen_file = clean_files[int(choice) - 1]
            if chosen_file in selected_files: selected_files.remove(chosen_file)
            else: selected_files.append(chosen_file)
        except: time.sleep(1)

    print(f"\n[ SYSTEM ] Spinning up tokenizer. This process handles memory securely.")
    try:
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import Whitespace
    except ImportError:
        print("[ CRITICAL ERROR ] Tokenizers module missing for .bin compilation.")
        print(" -> Run: pip install tokenizers")
        return

    if not os.path.exists(TOKENIZER_PATH):
        print(f"[ SYSTEM ] Building a NEW BPE tokenizer from selected files...")
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"], min_frequency=2)
        tokenizer.train([os.path.join(CLEAN_DIR, f) for f in selected_files], trainer)
        tokenizer.save(TOKENIZER_PATH)
    else:
        tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
        
    encode = lambda s: tokenizer.encode(s).ids
    target_bin_path = os.path.join(BIN_TRANSCRIPTS_DIR, f"Pretrain_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.bin")

    try:
        total_tokens = 0
        file_paths = [os.path.join(CLEAN_DIR, f) for f in selected_files]
        for file_path in file_paths:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as infile:
                for line in infile:
                    if line.strip(): total_tokens += len(encode(line.strip()))
                    
        data_mmap = np.memmap(target_bin_path, dtype=np.int32, mode='w+', shape=(total_tokens,))
        
        current_idx = 0
        for file_path in file_paths:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as infile:
                for line in infile:
                    if line.strip():
                        tokens = encode(line.strip())
                        num_tokens = len(tokens)
                        data_mmap[current_idx : current_idx + num_tokens] = tokens
                        current_idx += num_tokens
                        
        data_mmap.flush()
        del data_mmap
                
        print(f"\n>>> CONSOLIDATION COMPLETE! Core Base Model .bin constructed.")
        print(f"    Saved -> {target_bin_path}")
        input("\nPress Enter to return...")
    except Exception as e: print(f"[ CRITICAL ERROR ] {e}")

def run_lora_compiler():
    print("\n=======================================================")
    print("             LORA MASTER DATA COMPILER                 ")
    print("=======================================================")
    
    clean_files = [f for f in os.listdir(CLEAN_DIR) if f.endswith(".jsonl") and f.startswith("CLEAN_")]
    if not clean_files:
        print(f"[!] No valid cleaned JSONL files found inside '{CLEAN_DIR}'.")
        return

    selected_files = []
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        print("=======================================================")
        for idx, file_name in enumerate(clean_files, start=1):
            status = "[ SELECTED ]" if file_name in selected_files else "[   ---    ]"
            print(f" {status} [{idx}] {file_name}")
        print("-------------------------------------------------------")
        print(" * Enter a number to toggle selection. Type 'DONE' to compile.")
        choice = input("Selection input: ").strip()
        
        if not choice: return
        if choice.upper() == "DONE":
            if selected_files: break
            time.sleep(1)
            continue
        try:
            chosen_file = clean_files[int(choice) - 1]
            if chosen_file in selected_files: selected_files.remove(chosen_file)
            else: selected_files.append(chosen_file)
        except: time.sleep(1)

    final_output_path = os.path.join(LORA_DIR, f"LoRA_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.jsonl")
    
    try:
        with open(final_output_path, "w", encoding="utf-8") as master_out:
            for clean_file in selected_files:
                with open(os.path.join(CLEAN_DIR, clean_file), "r", encoding="utf-8") as infile:
                    for line in infile:
                        if line.strip(): master_out.write(line.strip() + "\n")
        print(f"\n>>> CONSOLIDATION COMPLETE! Saved -> {final_output_path}")
        input("\nPress Enter to return...")
    except Exception as e: print(f"[ CRITICAL ERROR ] {e}")

def extract_metadata(url):
    ydl_opts = {'quiet': True, 'nocheckcertificate': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
            return sanitize_name(info.get('extractor', 'web').capitalize()), sanitize_name(info.get('uploader', 'Unknown_Channel')), sanitize_name(info.get('title', 'Unknown_Title')), info
        except Exception as e: 
            print(f"[ WARNING ] Metadata extractor parsing dropped: {e}")
            return "Platform", "Unknown_Channel", "Unknown_Title", {}

def run_media_scraper():
    url = input("\nEnter YouTube or Twitch VOD URL: ").strip()
    if not url: return
    platform, channel, title, _ = extract_metadata(url)
    target_folder_path = os.path.join(BASE_DIR, f"{platform}_{channel}")
    os.makedirs(target_folder_path, exist_ok=True)
    
    task_id = f"scrape_{uuid.uuid4().hex[:8]}"
    raw_template = os.path.join(TEMP_DIR, f"{task_id}.%(ext)s")
    audio_path = os.path.join(TEMP_DIR, f"{task_id}.wav")
    
    ydl_opts = {
        'format': 'bestaudio/best', 
        'outtmpl': raw_template, 
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav', 'preferredquality': '192'}], 
        'quiet': False,
        'nocheckcertificate': True
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl: 
            ydl.download([url])
        
        if not os.path.exists(audio_path):
            return print("[ ERROR ] Audio download payload failed.")

        print("\n[ WHISPER ] Parsing audio file. Commencing full structural transcription...")
        print("            (If running on CPU, you will see text appear slowly below as it generates)\n")
        
        model = get_whisper_engine("whisper_vod_model")
        segments, _ = model.transcribe(audio_path, beam_size=5)
        
        with open(os.path.join(target_folder_path, f"TRANSCRIPT_{title}.txt"), "w", encoding="utf-8") as f:
            for segment in segments: 
                text_chunk = segment.text.strip()
                print(f" -> [ {segment.start:.2f}s - {segment.end:.2f}s ] {text_chunk}")
                f.write(text_chunk + "\n")
                f.flush() # Ensure it writes to the file continuously
                
        print(f"\n>>> SUCCESS: Raw transcript fully generated inside {target_folder_path}!")
    except Exception as e: 
        print(f"[ CRITICAL ERROR ] Transcription halted: {e}")
    finally:
        if os.path.exists(audio_path): 
            try: os.remove(audio_path)
            except: pass
    input("\nPress Enter to return to main menu...")

def run_live_chunker():
    url = input("\nEnter Live Stream URL (YouTube Live / Twitch): ").strip()
    if not url: return
    platform, channel, title, info = extract_metadata(url)
    
    stream_url = info.get('url') or (info.get('formats', [{}])[-1].get('url') if 'formats' in info else None)
    if not stream_url: 
        return print(f"[ ERROR ] Could not extract direct live stream feed.")
    
    target_folder_path = os.path.join(BASE_DIR, f"{platform}_{channel}")
    os.makedirs(target_folder_path, exist_ok=True)
    live_log_path = os.path.join(target_folder_path, f"TRANSCRIPT_LIVE_{title}.txt")

    chunk_idx, task_id = 0, f"live_{uuid.uuid4().hex[:6]}"
    model = get_whisper_engine("whisper_live_model")

    print(f"\n[ RUNNING ] Real-time Stream Chunker started for channel: '{channel}'")
    print(" -> Appending live transcript data continuously. Press Ctrl+C to halt.")
    print("----------------------------------------------------------------------")

    try:
        with open(live_log_path, "a", encoding="utf-8") as log_file:
            while True:
                temp_chunk = os.path.join(TEMP_DIR, f"{task_id}_slice_{chunk_idx}.wav")
                subprocess.run(['ffmpeg', '-y', '-i', stream_url, '-t', str(CONFIG["chunk_duration"]), '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', temp_chunk], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if os.path.exists(temp_chunk) and os.path.getsize(temp_chunk) > 0:
                    segments, _ = model.transcribe(temp_chunk, beam_size=1)
                    for segment in segments:
                        if segment.text.strip():
                            print(f" [ Live Segment ] {segment.text.strip()}")
                            log_file.write(segment.text.strip() + "\n")
                            log_file.flush()
                    try: os.remove(temp_chunk)
                    except: pass
                else: 
                    time.sleep(2)
                chunk_idx += 1
    except KeyboardInterrupt: 
        print(f"\n[ SYSTEM ] Capture pipeline closed safely.")
    input("\nPress Enter to return...")

def run_local_scan():
    print("\n=======================================================")
    print("          LOCAL DATA ACQUISITION & SCANNER             ")
    print("=======================================================")
    
    local_files = []
    supported_extensions = (".wav", ".mp3", ".mp4", ".m4a", ".flac", ".pdf")
    
    if os.path.exists(LOCAL_DATA_DIR):
        local_files = [f for f in os.listdir(LOCAL_DATA_DIR) if f.lower().endswith(supported_extensions)]
        
    if not local_files:
        print(f"[!] No valid media records or documents found inside '{LOCAL_DATA_DIR}'.")
        input("\nPress Enter to return...")
        return

    print("Discovered file indices inside Local_Data directory:")
    for idx, file_name in enumerate(local_files, start=1):
        print(f" [{idx}] {file_name}")
        
    try:
        selection = input("\nSelect entry number to process (or Enter to cancel): ").strip()
        if not selection: return
        chosen_filename = local_files[int(selection) - 1]
    except:
        return print("[!] Target selection out of bounds.")

    full_input_path = os.path.join(LOCAL_DATA_DIR, chosen_filename)
    base_name, extension = os.path.splitext(chosen_filename)
    output_text_file = os.path.join(LOCAL_DATA_DIR, f"TRANSCRIPT_LOCAL_{base_name}.txt")

    if extension.lower() == ".pdf":
        print(f"\n[ PARSING ] Unpacking text from local document: {chosen_filename}...")
        extracted_text = extract_text_from_pdf(full_input_path)
        if extracted_text:
            with open(output_text_file, "w", encoding="utf-8") as text_out:
                text_out.write(extracted_text)
            print(f"\n>>> SUCCESS: Text stripped and written -> {output_text_file}")
    else:
        print(f"\n[ WHISPER ] Transcribing local media file: {chosen_filename}...")
        print("            (You will see text generate line-by-line below)\n")
        try:
            model = get_whisper_engine("whisper_vod_model")
            segments, _ = model.transcribe(full_input_path, beam_size=5)
            
            with open(output_text_file, "w", encoding="utf-8") as text_out:
                for segment in segments:
                    text_chunk = segment.text.strip()
                    print(f" -> [ {segment.start:.2f}s - {segment.end:.2f}s ] {text_chunk}")
                    text_out.write(text_chunk + "\n")
                    text_out.flush()
            print(f"\n>>> SUCCESS: Transcription complete -> {output_text_file}")
        except Exception as e:
            print(f"[ CRITICAL ERROR ] Transcription failed: {e}")

    input("\nPress Enter to return...")

if __name__ == "__main__":
    while True:
        print("\n=======================================================")
        print("          UNIFIED DATA ACQUISITION ENGINE              ")
        print("=======================================================")
        print(" [1] Scrape Media VOD   - Extract & Transcribe Video")
        print(" [2] Chunk Live Stream  - Record & Transcribe Live Feed")
        print(" [3] Clean Transcripts  - Format LoRA Dataset & Learn Names")
        print(" [4] Build Pretrain BIN - Compile Text to Base Model Memory-Map")
        print(" [5] Make LoRA Dataset  - Combine Clean JSONL into Master Dataset")
        print(" [6] Local Data Scan    - Process Offline Media / Read PDF Assets")
        print(" [7] Exit Program       - Shut down interface")
        print("=======================================================")
        main_choice = input("Select operation parameter (1-7): ").strip()
        
        if main_choice == "1": run_media_scraper()
        elif main_choice == "2": run_live_chunker()
        elif main_choice == "3": run_dataset_cleaner()
        elif main_choice == "4": run_bin_compiler()
        elif main_choice == "5": run_lora_compiler()
        elif main_choice == "6": run_local_scan()
        elif main_choice == "7": break
