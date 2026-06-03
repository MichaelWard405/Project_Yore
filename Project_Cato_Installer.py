import os
import sys
import subprocess
import urllib.request

# ===================
#   Config Registry
# ===================
PROJECT_ROOT = "Project_Cato"
MASTER_DIR = os.path.join(PROJECT_ROOT, "Master")
VENV_DIR = os.path.join(MASTER_DIR, ".venv")

# Unified pipeline dependencies
MASTER_DEPENDENCIES = [
    "torch", 
    "tokenizers", 
    "fastapi", 
    "uvicorn", 
    "yt-dlp", 
    "faster-whisper", 
    "requests", 
    "beautifulsoup4", 
    "numpy",
    "textual",
    "pypdf"
]

#===========================
#  Python Files To Install
#===========================
PIPELINE_MODULES = {
    "1": {
        "name": "Neural Net Block",
        "files": {
            "Cato_Neural_Net.py": "https://raw.githubusercontent.com/MichaelWard405/Project_Yore/Project_Yore/Cato_Neural_Net.py"
        }
    },
    "2": {
        "name": "Data Collection Block",
        "files": {
            "Data_Collector.py": "https://raw.githubusercontent.com/MichaelWard405/Project_Yore/Project_Yore/Data_Collector.py"
        }
    },
    "3": {
        "name": "Master Interface",
        "files": {
            "Master_Interface.py": "https://raw.githubusercontent.com/MichaelWard405/Project_Yore/Project_Yore/Master_Interface.py"
        }
    }
}

# ===============
#   Helper Func
# ===============
def run_cmd(cmd, cwd=None):
    print(f"[*] Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)

def get_venv_python(venv_dir):
    if sys.platform == "win32":
        return os.path.join(venv_dir, "Scripts", "python.exe")
    else:
        return os.path.join(venv_dir, "bin", "python")

def download_file(url, dest_path):
    print(f"  -> Downloading {os.path.basename(dest_path)}...")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(dest_path, 'wb') as out_file:
            out_file.write(response.read())
    except Exception as e:
        print(f"\n[ CRITICAL ERROR ] Failed to download {url}")
        print(f"Details: {e}")
        sys.exit(1)

# =========================
#   Main Install PipeLine
# =========================
def main():
    print("\n=======================================================")
    print("           PROJECT YORE - Installation Wizard          ")
    print("=======================================================")
    
    print("Select modules to install:")
    for key, module in PIPELINE_MODULES.items():
        print(f" [{key}] {module['name']}")
    print(" [A] Install ALL Modules")
    print(" [Q] Quit")
    
    choice = input("\nEnter your choice: ").strip().upper()
    
    if choice == 'Q':
        sys.exit(0)
        
    target_keys = []
    if choice == 'A':
        target_keys = list(PIPELINE_MODULES.keys())
    else:
        target_keys = [k.strip() for k in choice.split(",") if k.strip() in PIPELINE_MODULES]
        
    if not target_keys:
        print("[ ERROR ] Invalid selection. Terminating.")
        sys.exit(1)

    print("\n[ SYSTEM ] Initializing Project_Cato Environment Build...\n")

    os.makedirs(MASTER_DIR, exist_ok=True)
    print(f"[+] Verified Master directory: {MASTER_DIR}")
    
    if not os.path.exists(VENV_DIR):
        print("\n--- Provisioning Unified Master Environment ---")
        run_cmd([sys.executable, "-m", "venv", VENV_DIR])
    else:
        print("\n--- Master Environment already exists. Verifying packages ---")
        
    venv_python = get_venv_python(VENV_DIR)
    
    run_cmd([venv_python, "-m", "pip", "install", "--upgrade", "pip", "--quiet"])
    run_cmd([venv_python, "-m", "pip", "install"] + MASTER_DEPENDENCIES)

    print("\n--- Fetching Target Modules ---")
    os.makedirs(PROJECT_ROOT, exist_ok=True)
    
    for key in target_keys:
        module = PIPELINE_MODULES[key]
        print(f"\n[+] Processing: {module['name']}")
        
        for file_name, file_url in module["files"].items():
            dest_path = os.path.join(PROJECT_ROOT, file_name)
            download_file(file_url, dest_path)

    # --- FINALIZATION ---
    print("\n=======================================================")
    print("               SYSTEM SETUP COMPLETE                   ")
    print("=======================================================")
    print(f"Your architecture is ready inside ./{PROJECT_ROOT}\n")
    print("To run your scripts using the Master Control Interface, activate it with:")
    
    if sys.platform == "win32":
        print(f"  > {os.path.join(PROJECT_ROOT, 'Master', '.venv', 'Scripts', 'activate')}")
    else:
        print(f"  > source {os.path.join(PROJECT_ROOT, 'Master', '.venv', 'bin', 'activate')}")
        
    print("\nOnce activated, execute your scripts directly from the root directory:")
    print(f"  > cd {PROJECT_ROOT}")
    print("  > python Master_Interface.py")

if __name__ == "__main__":
    main()
