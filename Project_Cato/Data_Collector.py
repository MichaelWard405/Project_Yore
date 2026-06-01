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
os.makedirs(BASE_DIR, exist_ok=True)

CONFIG = {
    "max_pages": 50,                  # Max pages for documentation crawler per session run
    "crawl_delay": 1.0,               # Seconds to pause between web page requests to protect your IP
    "chunk_duration": 15,             # Seconds per saved live audio slice
    "whisper_vod_model": "medium.en", # AI engine for pre-recorded media (high accuracy)
    "whisper_live_model": "small.en"  # AI engine for live feeds (low latency)
}
LOADED_MODELS = {}

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
            print("  -> SUCCESS: Engine running on Dedicated GPU Acceleration.")
        
        except Exception as gpu_err:
            print(f"  -> GPU/CUDA unavailable or library missing (Error: {gpu_err})")
            print("  -> Rolling over pipeline execution to standard CPU mode...")
            try:
                LOADED_MODELS[model_name] = WhisperModel(model_name, device="cpu", compute_type="int8")
                print("  -> SUCCESS: Engine running on Universal CPU mode.")
            except Exception as cpu_err:
                print(f"[ CRITICAL ERROR ] Target device could not launch CPU fallback model: {cpu_err}")
                raise cpu_err
                
    return LOADED_MODELS[model_name]

def sanitize_name(text):
    if not text: 
        return "Unknown"
    return re.sub(r'[\\/*?:"<>|]', "", str(text)).replace(" ", "_")

# =====================
#   Video Transcriber
# =====================
def run_media_scraper():
    url = input("\nEnter YouTube/Twitch Video or VOD URL: ").strip()
    if not url: return

    # Creates a secure temporary environment that auto-destructs when done
    with tempfile.TemporaryDirectory() as safe_temp_dir:
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(safe_temp_dir, '%(id)s.%(ext)s'),
            'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav'}],
            'noplaylist': True, 
            'quiet': True,
            'remote_components': ['ejs:github']
        }
        
        try:
            print(f"\n[ CONNECTING ] Interrogating media link via yt-dlp...")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                
                extractor = info.get("extractor_key", "").lower()
                platform = "Youtube" if "youtube" in extractor else "Twitch"
                channel = sanitize_name(info.get("uploader") or info.get("channel") or "Unknown_Channel")
                
                destination_dir = os.path.join(BASE_DIR, f"{platform}_{channel}")
                os.makedirs(destination_dir, exist_ok=True)
                
                wav_path = os.path.join(safe_temp_dir, f"{info.get('id')}.wav")
                if not os.path.exists(wav_path):
                    print("[ ERROR ] Audio capture layout missing.")
                    return
                
                whisper_engine = get_whisper_engine("whisper_vod_model")
                print(f"[ TRANSCRIBING ] Processing audio tracking for: '{info.get('title')}'...")
                
                segments, _ = whisper_engine.transcribe(wav_path, beam_size=5, vad_filter=True)
                
                safe_title = sanitize_name(info.get('title', 'Untitled_Video'))
                out_file = os.path.join(destination_dir, f"{safe_title}.txt")
                
                with open(out_file, "w", encoding="utf-8") as f:
                    for segment in segments:
                        clean_text = segment.text.strip()
                        if clean_text:
                            print(f"  [{segment.start:.1f}s] {clean_text}")
                            f.write(clean_text + "\n")
                
                print(f"\n>>> SUCCESS: Saved transcript document to: {out_file} <<<")
                
        except Exception as e:
            print(f"\n[ ERROR ] Media pipeline failed: {e}")
        print("[ CLEANUP ] Secure temp space destroyed.")

# ==================
#   Stream Chunker
# ==================
def run_live_chunker():
    url = input("\nEnter Live YouTube or Twitch Stream URL: ").strip()
    if not url: return

    try:
        print(f"\n[ CONNECTING ] Parsing live feed headers...")
        live_opts = {
            'quiet': True, 
            'noplaylist': True, 
            'remote_components': ['ejs:github']
        }
        with yt_dlp.YoutubeDL(live_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            raw_stream_url = info.get('url')
            channel = sanitize_name(info.get("uploader") or "Unknown_Live_Host")
            extractor = info.get("extractor_key", "").lower()
            platform = "Youtube" if "youtube" in extractor else "Twitch"
    except Exception as e:
        print(f"[ ERROR ] Failed to hook live feed stream matrix: {e}")
        return

    session_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session_dir = os.path.join(BASE_DIR, f"{platform}_{channel}_LIVE", f"Session_{session_time}")
    os.makedirs(session_dir, exist_ok=True)
    master_transcript = os.path.join(session_dir, "master_transcript.txt")
    
    print(f"\n[ RECORDING ] Connected successfully!")
    print(f"[ STORAGE ] Transcript securely writing to: {session_dir}")
    print("Press Ctrl+C to stop recording safely.\n")
    
    whisper_engine = get_whisper_engine("whisper_live_model")
    chunk_idx = 1
    
    with tempfile.TemporaryDirectory() as safe_temp_dir:
        try:
            while True:
                chunk_file = os.path.join(safe_temp_dir, f"chunk_{chunk_idx:03d}.wav")
                
                ffmpeg_cmd = [
                    'ffmpeg', '-y', '-i', raw_stream_url, '-t', str(CONFIG['chunk_duration']), 
                    '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', 
                    '-loglevel', 'quiet', chunk_file
                ]
                subprocess.run(ffmpeg_cmd, check=True)
                
                segments, _ = whisper_engine.transcribe(chunk_file, beam_size=5)
                with open(master_transcript, "a", encoding="utf-8") as f:
                    for segment in segments:
                        clean_text = segment.text.strip()
                        if clean_text:
                            print(f"  [{datetime.now().strftime('%H:%M:%S')}] {clean_text}")
                            f.write(clean_text + "\n")
                            
                if os.path.exists(chunk_file):
                    os.remove(chunk_file)
                    
                chunk_idx += 1
                
        except KeyboardInterrupt:
            print(f"\n[ SYSTEM ] Live capture processing loop terminated cleanly.")
        except subprocess.CalledProcessError:
            print("\n[ DISCONNECTED ] Stream feed terminated or lost link signal.")
        print("[ CLEANUP ] Erasing unhandled temporary live chunk sequences...")

# =================
#   State Crawler
# =================
def run_documentation_crawler():
    start_url = input("\nEnter Base Documentation URL: ").strip()
    if not start_url: return

    parsed_start = urllib.parse.urlparse(start_url)
    base_domain = parsed_start.netloc
    base_path = parsed_start.path
    
    domain_folder = sanitize_name(base_domain.replace("www.", ""))
    destination_dir = os.path.join(BASE_DIR, f"Website_{domain_folder}")
    os.makedirs(destination_dir, exist_ok=True)
    
    state_file = os.path.join(destination_dir, "crawl_state.json")
    
    resume_session = False
    if os.path.exists(state_file):
        print(f"\n[ SYSTEM ] Found an active crawl state log file inside '{destination_dir}'.")
        choice = input("Would you like to resume from where you left off? (y/n): ").strip().lower()
        if choice == 'y':
            resume_session = True

    if resume_session:
        try:
            with open(state_file, 'r', encoding='utf-8') as f:
                state_data = json.load(f)
            visited = set(state_data.get("visited", []))
            queue = deque(state_data.get("queue", []))
            print(f"[ SYSTEM ] Session Loaded: Skipping {len(visited)} old files. Processing next {CONFIG['max_pages']} from the queue.")
        except Exception as e:
            print(f"[ ERROR ] Failed to read session log ({e}). Starting fresh instead.")
            queue = deque([start_url])
            visited = set()
    else:
        queue = deque([start_url])
        visited = set()
    
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Dataset-Collector'}
    pages_scraped = 0
    
    print(f"\n[ CRAWLING ] Operational Window Opened for: {base_domain}")
    print(f"[ PROFILE ] Batch Target: {CONFIG['max_pages']} pages | Network Delay: {CONFIG['crawl_delay']}s\n")
    
    try:
        while queue and pages_scraped < CONFIG["max_pages"]:
            current_url = queue.popleft()
            
            url_clean = current_url.split('#')[0]
            if url_clean in visited: 
                continue
            visited.add(url_clean)
            
            try:
                print(f" [{pages_scraped + 1}/{CONFIG['max_pages']}] Fetching: {current_url}")
                response = requests.get(current_url, headers=headers, timeout=10)
                
                if response.status_code != 200 or 'text/html' not in response.headers.get('Content-Type', ''):
                    continue
                    
                soup = BeautifulSoup(response.content, 'html.parser')
                
                for element in soup(["script", "style"]):
                    element.extract()
                    
                page_text = soup.get_text(separator='\n', strip=True)
                
                if page_text:
                    page_title = sanitize_name(soup.title.string if soup.title else f"page_{len(visited)}")
                    out_file = os.path.join(destination_dir, f"{page_title}.txt")
                    
                    with open(out_file, "w", encoding="utf-8") as f:
                        f.write(f"SOURCE URL: {current_url}\n")
                        f.write(f"TITLE: {soup.title.string if soup.title else 'Untitled'}\n")
                        f.write("="*40 + "\n\n")
                        f.write(page_text)
                        
                    pages_scraped += 1
                    
                for anchor in soup.find_all('a', href=True):
                    next_url = urllib.parse.urljoin(current_url, anchor['href'])
                    parsed_next = urllib.parse.urlparse(next_url)
                    
                    if parsed_next.netloc == base_domain and parsed_next.path.startswith(base_path):
                        if not any(parsed_next.path.lower().endswith(ext) for ext in ['.zip', '.pdf', '.png', '.jpg', '.mp4', '.tar.gz']):
                            url_to_queue = next_url.split('#')[0]
                            if url_to_queue not in visited:
                                queue.append(url_to_queue)
                                
                time.sleep(CONFIG["crawl_delay"])
                
            except Exception as e:
                continue
                
    except KeyboardInterrupt:
        print("\n[ EMERGENCY INTERRUPT ] Crawl loop halted early by user request.")

    try:
        state_data = {
            "visited": list(visited),
            "queue": list(queue)
        }
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(state_data, f, indent=4)
        print(f"\n>>> STATE SECURED: Crawl status logged to '{state_file}' <<<")
        print(f"Total Site Map Progress: {len(visited)} absolute pages processed. {len(queue)} pages remaining in queue.")
    except Exception as e:
        print(f"[ ERROR ] Failed to preserve state backup logs: {e}")

# ==============
#    InterFace
# ==============
def run_settings_menu():
    while True:
        print("\n=======================================================")
        print("            PIPELINE SYSTEM SETTINGS CONTROL           ")
        print("=======================================================")
        print(f" [1] Max Scraping Pages Cap   : {CONFIG['max_pages']} pages")
        print(f" [2] Crawl Request Delay Time : {CONFIG['crawl_delay']} seconds")
        print(f" [3] Live Stream Audio Chunk  : {CONFIG['chunk_duration']} seconds")
        print(f" [4] Whisper Model (VOD/Media): '{CONFIG['whisper_vod_model']}'")
        print(f" [5] Whisper Model (Live Feed): '{CONFIG['whisper_live_model']}'")
        print(" [6] Return to Main Hub Dashboard")
        print("=======================================================")
        choice = input("Select value parameter to alter (1-6): ").strip()
        
        if choice == "1":
            val = input("Enter new max pages cap: ").strip()
            if val.isdigit(): CONFIG["max_pages"] = int(val)
        elif choice == "2":
            val = input("Enter new crawl delay interval (seconds): ").strip()
            try: CONFIG["crawl_delay"] = float(val)
            except ValueError: pass
        elif choice == "3":
            val = input("Enter live capture audio slice duration (seconds): ").strip()
            if val.isdigit(): CONFIG["chunk_duration"] = int(val)
        elif choice == "4":
            print("\nAvailable profiles: tiny.en, base.en, small.en, medium.en, large-v3")
            val = input("Enter new VOD Whisper model key: ").strip().lower()
            if val in ["tiny.en", "base.en", "small.en", "medium.en", "large-v3"]: CONFIG["whisper_vod_model"] = val
        elif choice == "5":
            print("\nAvailable profiles: tiny.en, base.en, small.en, medium.en, large-v3")
            val = input("Enter new Live Whisper model key: ").strip().lower()
            if val in ["tiny.en", "base.en", "small.en", "medium.en", "large-v3"]: CONFIG["whisper_live_model"] = val
        elif choice == "6":
            break

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
        print(" [4] Adjust Settings    - Configure Pipeline Hyperparameters")
        print(" [5] Exit Program       - Shut down interface")
        print("=======================================================")
        main_choice = input("Select operation parameter (1-5): ").strip()
        
        if main_choice == "1": run_media_scraper()
        elif main_choice == "2": run_live_chunker()
        elif main_choice == "3": run_documentation_crawler()
        elif main_choice == "4": run_settings_menu()
        elif main_choice == "5": 
            print("\nShutting down pipeline components safely.")
            break 
