import os
import sys
import re
import time
import json
import subprocess
import uuid
from datetime import datetime
import numpy as np

import urllib.request
import urllib.error
try:
    import yt_dlp
except ImportError:
    yt_dlp = None

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

BASE_DIR = "Data_Collection"
CLEAN_DIR = os.path.join(BASE_DIR, "clean_data")
TEMP_DIR = os.path.join(BASE_DIR, "temp_downloads")
BIN_TRANSCRIPTS_DIR = os.path.join(BASE_DIR, "Binary_Transcripts")
LOCAL_DATA_DIR = os.path.join(BASE_DIR, "Local_Data")
NEURAL_DIR = "Neural_Net"
TOKENIZER_PATH = os.path.join(NEURAL_DIR, "custom_bpe.json")
VOCAB_SIZE = 32000

def ensure_environment():
    for directory in [BASE_DIR, CLEAN_DIR, TEMP_DIR, BIN_TRANSCRIPTS_DIR, LOCAL_DATA_DIR, NEURAL_DIR]:
        os.makedirs(directory, exist_ok=True)

ensure_environment()

CONFIG = { 
    "chunk_duration": 15,
    "whisper_vod_model": "medium.en",
    "whisper_live_model": "small.en"
}
LOADED_MODELS = {}

def get_whisper_engine(config_key):
    if WhisperModel is None:
        print("[ ERROR ] 'faster_whisper' is not installed. Action canceled.")
        return None
    model_name = CONFIG[config_key]
    if model_name not in LOADED_MODELS:
        print(f"\n[ SYSTEM ] Spinning up Whisper Model '{model_name}'...")
        try:
            test_model = WhisperModel(model_name, device="cuda", compute_type="float16")
            dummy_audio = np.zeros(16000, dtype=np.float32) 
            list(test_model.transcribe(dummy_audio)[0])
            LOADED_MODELS[model_name] = test_model
        except Exception:
            print("[ SYSTEM WARNING ] CUDA isolated. Diverting to localized CPU threads.")
            LOADED_MODELS[model_name] = WhisperModel(model_name, device="cpu", compute_type="int8")
    return LOADED_MODELS[model_name]

def extract_text_from_pdf(pdf_path):
    try:
        import pypdf
        reader = pypdf.PdfReader(pdf_path)
        return "".join([page.extract_text() + "\n" for page in reader.pages if page.extract_text()])
    except ImportError:
        print("[ ERROR ] 'pypdf' package missing.")
        return None

def sanitize_name(text):
    if not text: return "Unknown"
    return re.sub(r'[\\/*?:"<>|]', "", str(text)).replace(" ", "_").strip()

def run_dataset_cleaner():
    print("\n=======================================================")
    print("        GLOBAL TRANSCRIPT LINE DEDUPLICATOR           ")
    print("=======================================================")
    
    txt_files = []
    for root, dirs, files in os.walk(BASE_DIR):
        if any(d in root for d in ["clean_data", "temp_downloads", "Binary_Transcripts", "Local_Data"]): continue
        for file in files:
            if file.endswith(".txt"):
                full_path = os.path.join(root, file)
                txt_files.append((os.path.relpath(full_path, BASE_DIR), full_path))
    
    if not txt_files:
        print(f"[!] No raw text files discovered inside '{BASE_DIR}'.")
        return

    for idx, (display_path, _) in enumerate(txt_files, start=1):
        print(f" [{idx}] {display_path}")
        
    try:
        selection = input("\nSelect entry to clean: ").strip()
        if not selection: return
        display_path, input_path = txt_files[int(selection) - 1]
    except: return print("[!] Invalid index selected.")

    base_name = os.path.splitext(display_path.replace(os.sep, "_"))[0]
    output_path_jsonl = os.path.join(CLEAN_DIR, f"CLEAN_{base_name}.jsonl")
    output_path_txt = os.path.join(CLEAN_DIR, f"CLEAN_{base_name}.txt")

    with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
        raw_text = f.read().replace('\x00', '')

    cleaned_lines = []
    seen_lines = set()
    for line in raw_text.splitlines():
        c_line = line.strip()
        if c_line and c_line.lower() not in seen_lines:
            cleaned_lines.append(c_line)
            seen_lines.add(c_line.lower())

    with open(output_path_jsonl, 'w', encoding='utf-8') as out_jsonl, open(output_path_txt, 'w', encoding='utf-8') as out_txt:
        for text_line in cleaned_lines:
            out_txt.write(text_line + "\n")
            out_jsonl.write(json.dumps({"text": text_line}) + "\n")

    print(f"\n>>> SUCCESS: Records cleaned without duplication at:\n -> {output_path_txt}\n -> {output_path_jsonl}")
    input("\nPress Enter to continue...")

def run_bin_compiler():
    print("\n=======================================================")
    print("      BASE MODEL .BIN COMPILER (MEMORY MAP)            ")
    print("=======================================================")
    
    clean_files = [f for f in os.listdir(CLEAN_DIR) if f.endswith(".txt") and f.startswith("CLEAN_")]
    if not clean_files: return print(f"[!] No functional text elements located in '{CLEAN_DIR}'.")

    selected_files = []
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        print("=======================================================")
        for idx, file_name in enumerate(clean_files, start=1):
            status = "[ SELECTED ]" if file_name in selected_files else "[   ---    ]"
            print(f" {status} [{idx}] {file_name}")
        print("-------------------------------------------------------")
        choice = input("Select number to toggle (Type 'DONE' to compile): ").strip()
        
        if choice.upper() == "DONE" and selected_files: break
        if not choice: return
        try:
            chosen = clean_files[int(choice) - 1]
            selected_files.remove(chosen) if chosen in selected_files else selected_files.append(chosen)
        except: pass

    try:
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import Whitespace
    except ImportError:
        print("[ ERROR ] Tokenizers library missing. Run: pip install tokenizers")
        return

    if not os.path.exists(TOKENIZER_PATH):
        print(f"[ SYSTEM ] Formulating a completely new BPE profile tokenizer schema...")
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
        for path in file_paths:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if line.strip(): total_tokens += len(encode(line.strip()))
                    
        data_mmap = np.memmap(target_bin_path, dtype=np.int32, mode='w+', shape=(total_tokens,))
        
        current_idx = 0
        for path in file_paths:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if line.strip():
                        tokens = encode(line.strip())
                        num_t = len(tokens)
                        data_mmap[current_idx : current_idx + num_t] = tokens
                        current_idx += num_t
                        
        data_mmap.flush()
        del data_mmap
        print(f"\n>>> PIPELINE INTEGRITY EXCELLENT: Memory Map generated at:\n -> {target_bin_path}")
        input("\nPress Enter to return...")
    except Exception as e: print(f"[ CRITICAL MATRIX FAILURE ] {e}")

def run_llm_formatting_pipeline():
    print("\n=======================================================")
    print("             Server & FORMATTING PIPELINE            ")
    print("=======================================================")
    
    SERVER_BASE_URL = "http://localhost:5000"
    SERVER_FORMAT_URL = "http://localhost:5000/format"

    boot_choice = input("Launch background Server framework instance? (y/n) [Default: y]: ").strip().lower()
    server_process = None
    if boot_choice != 'n':
        if os.path.exists("Formatting_Server.py"):
            try:
                server_process = subprocess.Popen([sys.executable, "Formatting_Server.py"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f"[ SERVER RUNNING ] Initialized Background Daemon PID: {server_process.pid}")
            except Exception as e: print(f"[ EXCEPTION ] Cannot run script natively: {e}")

    print("\n[ SYSTEM ] Pinging engine synchronization hooks...")
    server_ready = False
    for i in range(45):
        try:
            req = urllib.request.Request(SERVER_BASE_URL, method='GET')
            with urllib.request.urlopen(req, timeout=2) as r:
                if r.status == 200:
                    print("\n[ SUCCESS ] Local Formatting Engine is fully synchronized.\n")
                    server_ready = True
                    break
        except: pass
        sys.stdout.write("·")
        sys.stdout.flush()
        time.sleep(2)

    if not server_ready:
        if input("\n[ WARNING ] Server unreached. Continue blindly? (y/n): ").strip().lower() != 'y':
            if server_process: server_process.terminate()
            return

    clean_files = [f for f in os.listdir(CLEAN_DIR) if f.endswith(".jsonl") and not f.startswith("LLM_FORMATTED_")]
    if not clean_files:
        print("[!] No clean operational vectors detected.")
        if server_process: server_process.terminate()
        return

    for idx, f_name in enumerate(clean_files, start=1):
        print(f" [{idx}] {f_name}")
        
    selection = input("\nSelect comma-separated indexes to structure or type 'all': ").strip()
    selected_files = clean_files if selection.lower() == 'all' else []
    if not selected_files:
        try: selected_files = [clean_files[int(x.strip()) - 1] for x in selection.split(',')]
        except: return print("[!] Error parsing indices.")

    combined_lines = []
    for file in selected_files:
        with open(os.path.join(CLEAN_DIR, file), 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try: combined_lines.append(json.loads(line.strip()).get("text", ""))
                    except: pass

    total_lines = len(combined_lines)
    if total_lines == 0:
        if server_process: server_process.terminate()
        return

    output_filename = f"LLM_FORMATTED_COMBINED_DATASET_{int(time.time())}.jsonl"
    output_path = os.path.join(CLEAN_DIR, output_filename)
    
    success_count = 0
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for idx, text in enumerate(combined_lines, start=1):
            try:
                payload = json.dumps({"text": text}).encode("utf-8")
                req = urllib.request.Request(SERVER_FORMAT_URL, data=payload, headers={'Content-Type': 'application/json'}, method='POST')
                with urllib.request.urlopen(req, timeout=120) as resp:
                    res = json.loads(resp.read().decode("utf-8"))
                    formatted = res.get("formatted_text", "")
                    if formatted:
                        outfile.write(json.dumps({"text": formatted}) + "\n")
                        outfile.flush()
                        print(f" [{idx}/{total_lines}] Transformed -> {formatted[:60]}...")
                        success_count += 1
            except Exception as e:
                print(f" [{idx}/{total_lines}] Line execution fault: {e}")
                
    print(f"\n>>> OPERATION SUCCESSFUL: Parsed strings written directly into:\n -> {output_path}")
    if server_process: server_process.terminate()
    input("\nPress Enter to return...")

def run_media_scraper():
    if yt_dlp is None or WhisperModel is None:
        return print("[!] Essential libraries ('yt_dlp' or 'faster_whisper') are missing.")
    url = input("\nEnter Media Stream or Video URL: ").strip()
    if not url: return
    platform, channel, title, _ = extract_metadata(url)
    folder = os.path.join(BASE_DIR, f"{platform}_{channel}")
    os.makedirs(folder, exist_ok=True)
    
    task_id = f"scrape_{uuid.uuid4().hex[:8]}"
    audio_path = os.path.join(TEMP_DIR, f"{task_id}.wav")
    
    ydl_opts = {
        'format': 'bestaudio/best', 
        'outtmpl': os.path.join(TEMP_DIR, f"{task_id}.%(ext)s"), 
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav', 'preferredquality': '192'}], 
        'quiet': True,
        'nocheckcertificate': True
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl: ydl.download([url])
        if not os.path.exists(audio_path): return
        
        model = get_whisper_engine("whisper_vod_model")
        segments, _ = model.transcribe(audio_path, beam_size=5)
        
        with open(os.path.join(folder, f"TRANSCRIPT_{title}.txt"), "w", encoding="utf-8") as f:
            for segment in segments: 
                print(f" -> {segment.text.strip()}")
                f.write(segment.text.strip() + "\n")
        print("\n>>> SUCCESS: Transcription complete.")
    except Exception as e: print(f"[ ERROR ] {e}")
    finally:
        if os.path.exists(audio_path): os.remove(audio_path)
    input("\nPress Enter to return..."):

def run_live_chunker():
    if yt_dlp is None or WhisperModel is None: return print("[ ERROR ] Subsystem modules missing.")
    url = input("\nEnter Active Live Feed URL: ").strip()
    if not url: return
    platform, channel, title, info = extract_metadata(url)
    stream_url = info.get('url') or (info.get('formats', [{}])[-1].get('url') if 'formats' in info else None)
    if not stream_url: return
    
    folder = os.path.join(BASE_DIR, f"{platform}_{channel}")
    os.makedirs(folder, exist_ok=True)
    log_path = os.path.join(folder, f"TRANSCRIPT_LIVE_{title}.txt")
    model = get_whisper_engine("whisper_live_model")

    chunk_idx, task_id = 0, f"live_{uuid.uuid4().hex[:6]}"
    print("\n[ ACTIVE ] Streaming slice interception sequence deployed. CTRL+C to halt.")
    try:
        with open(log_path, "a", encoding="utf-8") as log_file:
            while True:
                chunk_file = os.path.join(TEMP_DIR, f"{task_id}_{chunk_idx}.wav")
                subprocess.run(['ffmpeg', '-y', '-i', stream_url, '-t', str(CONFIG["chunk_duration"]), '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', chunk_file], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if os.path.exists(chunk_file) and os.path.getsize(chunk_file) > 0:
                    segments, _ = model.transcribe(chunk_file, beam_size=1)
                    for seg in segments:
                        if seg.text.strip():
                            print(f" [ Live ] {seg.text.strip()}")
                            log_file.write(seg.text.strip() + "\n")
                            log_file.flush()
                    os.remove(chunk_file)
                else: time.sleep(2)
                chunk_idx += 1
    except KeyboardInterrupt: print("\n[ HALTED ] Stream interception disengaged gracefully.")
    input("\nPress Enter to return...")

def run_local_scan():
    print("\n=======================================================")
    print("          LOCAL DATA ACQUISITION & SCANNER             ")
    print("=======================================================")
    supported = (".wav", ".mp3", ".mp4", ".m4a", ".flac", ".pdf")
    files = [f for f in os.listdir(LOCAL_DATA_DIR) if f.lower().endswith(supported)] if os.path.exists(LOCAL_DATA_DIR) else []
    
    if not files: return print(f"[!] Target directory '{LOCAL_DATA_DIR}' contains no supported files.")
    for idx, name in enumerate(files, start=1): print(f" [{idx}] {name}")
        
    try:
        sel = input("\nSelect index: ").strip()
        chosen = files[int(sel) - 1]
    except: return
    
    input_path = os.path.join(LOCAL_DATA_DIR, chosen)
    out_file = os.path.join(LOCAL_DATA_DIR, f"TRANSCRIPT_LOCAL_{os.path.splitext(chosen)[0]}.txt")

    if chosen.lower().endswith(".pdf"):
        txt = extract_text_from_pdf(input_path)
        if txt:
            with open(out_file, "w", encoding="utf-8") as f: f.write(txt)
            print(f">>> PDF conversion complete: {out_file}")
    else:
        model = get_whisper_engine("whisper_vod_model")
        segments, _ = model.transcribe(input_path, beam_size=5)
        with open(out_file, "w", encoding="utf-8") as f:
            for s in segments:
                print(f" -> {s.text.strip()}")
                f.write(s.text.strip() + "\n")
    input("\nPress Enter to continue...")

def extract_metadata(url):
    if yt_dlp is None: return "Platform", "Unknown_Channel", "Unknown_Title", {}
    with yt_dlp.YoutubeDL({'quiet': True, 'nocheckcertificate': True}) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
            return sanitize_name(info.get('extractor', 'web').capitalize()), sanitize_name(info.get('uploader', 'Unknown_Channel')), sanitize_name(info.get('title', 'Unknown_Title')), info
        except: return "Platform", "Unknown_Channel", "Unknown_Title", {}

if __name__ == "__main__":
    while True:
        print("\n=======================================================")
        print("          STREAMLINED DATA ACQUISITION ENGINE          ")
        print("=======================================================")
        print(" [1] Scrape Media VOD   - Extract & Transcribe Video")
        print(" [2] Chunk Live Stream  - Record & Transcribe Live Feed")
        print(" [3] Clean Transcripts  - Deduplicate & Save as TXT + JSONL")
        print(" [4] Build Pretrain BIN - Compile Text to Base Model Memory-Map")
        print(" [5] LLM Formatter Pipeline - Boot LLaMA & Process JSONL File")
        print(" [6] Local Data Scan    - Process Offline Media / Read PDF Assets")
        print(" [7] Exit Program")
        print("=======================================================")
        choice = input("Select action (1-7): ").strip()
        if choice == "1": run_media_scraper()
        elif choice == "2": run_live_chunker()
        elif choice == "3": run_dataset_cleaner()
        elif choice == "4": run_bin_compiler()
        elif choice == "5": run_llm_formatting_pipeline()
        elif choice == "6": run_local_scan()
        elif choice == "7": sys.exit(0)
