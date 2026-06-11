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
    "pypdf",
    "llama-cpp-python",
    "transformers", # Fixed typo from "tranformers"
    "accelerate",
    "bitsandbytes",
    "peft",
    "unsloth @ git+https://github.com/unslothai/unsloth.git"
]
#===========================
#  Python Files To Install
#===========================
PIPELINE_MODULES = {
    "1": {
        "name": "Neural Net Block",
        "files": {
            "Cato_Neural_Net.py": "https://raw.githubusercontent.com/MichaelWard405/Project_Yore/Project_Yore/Project_Cato/Cato_Neural_Net.py"
        }
    },
    "2": {
        "name": "Data Collection Block",
        "files": {
            "Data_Collector.py": "https://raw.githubusercontent.com/MichaelWard405/Project_Yore/Project_Yore/Project_Cato/Data_Collector.py"
        }
    },
    "3": {
        "name": "Master Interface",
        "files": {
            "Master_Interface.py": "https://raw.githubusercontent.com/MichaelWard405/Project_Yore/Project_Yore/Project_Cato/Master_Interface.py"
        }
    },
    "4": {
        "name": "Formatting Server",
        "files": {
            "Formatting_Server.py": "https://raw.githubusercontent.com/MichaelWard405/Project_Yore/Project_Yore/Project_Cato/Formatting_Server.py"
        }
    },
    "5": {
        "name": "download_local.py",
        "files": {
            "download_local.py": "https://raw.githubusercontent.com/MichaelWard405/Project_Yore/Project_Yore/Project_Cato/download_local.py"
        }
    },
    "6": {
        "name": "Context Data Configuration",
        "files": {
            "Neural_Net/Context/Context.txt": "https://raw.githubusercontent.com/MichaelWard405/Project_Yore/Project_Yore/Project_Cato/Neural_Net/Context/Context.txt"
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
            # Automatically build nested subdirectories (e.g. Neural_Net/Context/) if they don't exist yet
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            download_file(file_url, dest_path)

    # =====================================
    #   POST-INSTALL AUTOMATION & CLEANUP
    # =====================================
    print("\n=======================================================")
    print("           EXECUTING POST-INSTALL SEQUENCE             ")
    print("=======================================================")
    
    # Store absolute paths before changing working directories
    installer_path = os.path.abspath(__file__)
    venv_python = os.path.abspath(venv_python)

    # Move into the project directory immediately so everything runs inside it
    os.chdir(PROJECT_ROOT)

    download_local_path = "download_local.py"
    if os.path.exists(download_local_path):
        print("\n[*] Launching download_local.py using the virtual environment...")
        try:
            subprocess.run([venv_python, download_local_path], check=True)
            print("[+] download_local.py completed successfully.")
            print(f"[*] Deleting {os.path.basename(download_local_path)}...")
            os.remove(download_local_path)
        except subprocess.CalledProcessError:
            print("[ ERROR ] download_local.py failed during execution. Halting sequence.")
            sys.exit(1)
        except Exception as e:
            print(f"[ ERROR ] Could not execute or delete download_local.py: {e}")
    else:
        print("\n[!] download_local.py not found. Skipping execution.")
        
    print(f"\n[*] Self-destructing installer: {os.path.basename(installer_path)}...")
    try:
        os.remove(installer_path)
    except Exception as e:
        print(f"[ WARNING ] Could not delete installer file automatically: {e}")
        
    master_interface_path = "Master_Interface.py"
    if os.path.exists(master_interface_path):
        print(f"\n[*] Launching Master_Interface.py...")
        print("=======================================================\n")
        subprocess.run([venv_python, "Master_Interface.py"])
    else:
        print("\n[ ERROR ] Master_Interface.py not found. Cannot launch.")

if __name__ == "__main__":
    main()
