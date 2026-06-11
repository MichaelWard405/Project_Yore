import os
from huggingface_hub import snapshot_download

LOCAL_MODEL_DIR = "./Neural_Net/Local_Model"
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

print("Initiating direct raw file transfer from Hugging Face...")
model_id = "unsloth/Llama-3.2-3B-Instruct-unsloth-bnb-4bit"

# This downloads the raw files EXACTLY as they are on the server, bypassing the transformers unpack/repack issue.
snapshot_download(
    repo_id=model_id,
    local_dir=LOCAL_MODEL_DIR,
    local_dir_use_symlinks=False,  # Forces actual file downloads, not cache shortcuts
    ignore_patterns=["*.msgpack", "*.h5", "coreml/*"]  # Ignores unnecessary extra bloat
)

print(f"\n>>> ALL UTILITIES EXPORTED SUCCESSFULLY. Environment is safe for offline deployment.")
