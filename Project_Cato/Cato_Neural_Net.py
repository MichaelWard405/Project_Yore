import os
import re
import json
import torch
import threading
import torch.nn as nn
from torch.nn import functional as F
from tokenizers import Tokenizer
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import requests
from datetime import datetime
import numpy as np

# ==================================
#   Redundancy & Environment Setup
# ==================================
NEURAL_FOLDER = "Neural_Net"
PERSONALITY_DIR = os.path.join(NEURAL_FOLDER, "Personalities")
DATA_COLLECTION_DIR = "Data_Collection"
LORA_DIR = os.path.join(DATA_COLLECTION_DIR, "LoRA_Transcripts")
BIN_TRANSCRIPTS_DIR = os.path.join(DATA_COLLECTION_DIR, "Binary_Transcripts")

def ensure_environment():
    for directory in [NEURAL_FOLDER, PERSONALITY_DIR, DATA_COLLECTION_DIR, LORA_DIR, BIN_TRANSCRIPTS_DIR]:
        os.makedirs(directory, exist_ok=True)

ensure_environment()

# ===================
#   HyperParameters
# ===================
batch_size = 16 
block_size = 256 
iters = 5000
max_iters_pretrain = iters 
max_iters_lora = iters
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embd = 384 
n_head = 4  
n_layer = 8 
dropout = 0.1
steps = iters - 1

MODEL_PATH = os.path.join(PERSONALITY_DIR, "cato_base_brain.pth")
TOKENIZER_PATH = os.path.join(NEURAL_FOLDER, "custom_bpe.json")
model_compute_lock = threading.Lock()
RECENT_ACTIONS = []

# ============================
#   Memory Management System
# ============================
class MemoryManager:
    def __init__(self, mem_dir="Memories"):
        self.mem_dir = mem_dir
        os.makedirs(mem_dir, exist_ok=True)
        self.people_file = os.path.join(mem_dir, "people.json")
        self.location_file = os.path.join(mem_dir, "locations.json")
        self.action_file = os.path.join(mem_dir, "last_action.json")
        self._init_defaults()

    def _init_defaults(self):
        if not os.path.exists(self.location_file):
            with open(self.location_file, 'w', encoding='utf-8') as f: json.dump({"current_location": "Tokyo, Japan", "location_history": []}, f, indent=4)
        if not os.path.exists(self.people_file):
            with open(self.people_file, 'w', encoding='utf-8') as f: json.dump({"Default": {"relationship": "Acquaintance", "tone_preference": "Neutral"}}, f, indent=4)
        if not os.path.exists(self.action_file):
            with open(self.action_file, 'w', encoding='utf-8') as f: json.dump({"action": "None", "timestamp": None}, f, indent=4)

    def get_full_context(self, current_person="Default"):
        try:
            with open(self.location_file, 'r', encoding='utf-8') as f: loc = json.load(f).get("current_location", "Unknown")
            with open(self.people_file, 'r', encoding='utf-8') as f: p_data = json.load(f).get(current_person, {"relationship": "Acquaintance"})
            return f"System Context: Currently in {loc}. Interlocutor: {current_person}. Relationship: {p_data['relationship']}."
        except: return "Context: Active."

    def update_location(self, new_location):
        try:
            with open(self.location_file, 'r+', encoding='utf-8') as f:
                data = json.load(f)
                data["current_location"] = new_location.strip()
                f.seek(0)
                json.dump(data, f, indent=4)
                f.truncate()
        except: pass

    def record_action(self, action_name):
        try:
            with open(self.action_file, 'w', encoding='utf-8') as f: json.dump({"action": action_name, "timestamp": str(datetime.now())}, f, indent=4)
        except: pass

memory_manager = MemoryManager()

if not os.path.exists(TOKENIZER_PATH):
    print(f"[ CRITICAL ] Tokenizer '{TOKENIZER_PATH}' is missing. Run Data_Collector Option 4.")
    exit()

tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
encode = lambda s: tokenizer.encode(s).ids
decode = lambda l: tokenizer.decode(l)

vocab_size = tokenizer.get_vocab_size() 

print("\n=========================================")
print("          SYSTEM STARTUP MENU            ")
print("=========================================")
print(" [1] Full Retrain (Base Pre-train + Select LoRA Run)")
print(" [2] Direct Interface (Use Base Set-up)")
print(" [3] Interface Override (Pick Select Personality)")
print("=========================================")
init_choice = input("Select an option (1-3): ").strip()

should_train = False
selected_weights_path = MODEL_PATH
data_pretrain = None
data_lora = torch.tensor([], dtype=torch.long)

if init_choice == "1":
    should_train = True
    lora_options = [os.path.normpath(os.path.join(r, f)) for r, d, files in os.walk(LORA_DIR) for f in files if f.endswith(".jsonl")]
    if lora_options:
        print("\n=========================================")
        print("      LORA TARGET SELECTION              ")
        print("=========================================")
        for idx, path in enumerate(lora_options, start=1): print(f" [{idx}] {os.path.basename(path)}")
        try: target_lora = lora_options[int(input("Select target LoRA to fine-tune upon: ").strip()) - 1]
        except: target_lora = lora_options[-1]
        
        lora_tokens = []
        with open(target_lora, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try: lora_tokens.extend(encode(json.loads(line)["text"] + "\n"))
                    except: pass
        if lora_tokens: data_lora = torch.tensor(lora_tokens, dtype=torch.long)
            
    bin_options = [os.path.normpath(os.path.join(BIN_TRANSCRIPTS_DIR, f)) for f in os.listdir(BIN_TRANSCRIPTS_DIR) if f.endswith(".bin")]
    if bin_options:
        print("\n=========================================")
        print("      BINARY TARGET SELECTION            ")
        print("=========================================")
        for idx, path in enumerate(bin_options, start=1): print(f" [{idx}] {os.path.basename(path)}")
        try: target_bin = bin_options[int(input("Select base binary memory map (.bin): ").strip()) - 1]
        except: target_bin = bin_options[-1]
        
        print(f"[ SYSTEM ] Mounting core binary via mmap -> '{target_bin}'...")
        data_pretrain = np.memmap(target_bin, dtype=np.int32, mode='r')
    else:
        print("[ CRITICAL ] No .bin files found in Binary_Transcripts folder. Cannot train base model.")
        exit()

elif init_choice == "3":
    pth_options = [os.path.normpath(os.path.join(PERSONALITY_DIR, f)) for f in os.listdir(PERSONALITY_DIR) if f.endswith(".pth")]
    if pth_options:
        print("\n=========================================")
        print("      PERSONALITY TARGET SELECTION       ")
        print("=========================================")
        for idx, path in enumerate(pth_options, start=1): print(f" [{idx}] {os.path.basename(path)}")
        try: selected_weights_path = pth_options[int(input("\nSelect weight index: ").strip()) - 1]
        except: print("[!] Fallback to default weights.")
    else:
        print(f"[!] No parameters found in {PERSONALITY_DIR}. Using standard initialization.")

if not should_train and os.path.exists(selected_weights_path):
    try:
        checkpoint = torch.load(selected_weights_path, map_location='cpu')
        if 'token_embedding_table.weight' in checkpoint:
            loaded_vocab_shape = checkpoint['token_embedding_table.weight'].shape[0]
            if loaded_vocab_shape != vocab_size:
                print(f"[ SYSTEM ADAPTATION ] Dynamic adaptation activated. Resizing structural architecture from {vocab_size} -> {loaded_vocab_shape} to match checkpoint.")
                vocab_size = loaded_vocab_shape
        del checkpoint
    except Exception as e:
        print(f"[!] Warning reading checkpoint meta headers: {e}")

def get_batch(data_source):
    if isinstance(data_source, np.memmap):
        ix = torch.randint(len(data_source) - block_size, (batch_size,))
        x = torch.stack([torch.tensor(data_source[i.item():i.item()+block_size], dtype=torch.long) for i in ix])
        y = torch.stack([torch.tensor(data_source[i.item()+1:i.item()+block_size+1], dtype=torch.long) for i in ix])
    else:
        ix = torch.randint(len(data_source) - block_size, (batch_size,))
        x = torch.stack([data_source[i:i+block_size] for i in ix])
        y = torch.stack([data_source[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# ====================================
#   Neural Net And LoRA Integrations
# ====================================
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False, r=4, lora_alpha=8):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.r = r
        self.scaling = lora_alpha / r
        self.lora_A = nn.Parameter(torch.randn(in_features, r) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(r, out_features))
    def forward(self, x): return self.linear(x) + ((x @ self.lora_A @ self.lora_B) * self.scaling)

def apply_lora_freezing(model):
    for name, param in model.named_parameters():
        if 'lora_' not in name: param.requires_grad = False

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = LoRALinear(n_embd, head_size, bias=False, r=4, lora_alpha=8)
        self.value = LoRALinear(n_embd, head_size, bias=False, r=4, lora_alpha=8)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        B,T,C = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x): return self.dropout(self.proj(torch.cat([h(x) for h in self.heads], dim=-1)))

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.ReLU(), nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout))
    def forward(self, x): return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        return x + self.ffwd(self.ln2(x))

class ChatbotModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) 
        self.lm_head = nn.Linear(n_embd, vocab_size)
    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_embedding_table(idx) + self.position_embedding_table(torch.arange(T, device=device))
        x = self.ln_f(self.blocks(x))
        logits = self.lm_head(x) 
        loss = F.cross_entropy(logits.view(B*T, logits.shape[-1]), targets.view(B*T)) if targets is not None else None
        return logits, loss
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -block_size:])
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, num_samples=1)), dim=1)
        return idx

model = ChatbotModel().to(device)

if not should_train:
    if os.path.exists(selected_weights_path):
        print(f"[ SYSTEM ] Loading saved weights directly from '{selected_weights_path}'...")
        model.load_state_dict(torch.load(selected_weights_path, map_location=device))
        model.eval()
    else:
        print(f"[!] Target file '{selected_weights_path}' missing. Interface will run un-trained.")

if should_train and data_pretrain is not None:
    print(f"\n[ STAGE 1 ] Executing Pre-training loop from memory-mapped tracking binary...")
    optimizer_base = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    for iter in range(max_iters_pretrain):
        xb, yb = get_batch(data_pretrain)
        logits, loss = model(xb, targets=yb)
        optimizer_base.zero_grad(set_to_none=True)
        loss.backward()
        optimizer_base.step()
        if iter == 0 or iter == steps // 2 or iter == steps: print(f"  Base Step: {iter} | Loss: {loss.item():.4f}")

    apply_lora_freezing(model)
    optimizer_lora = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate)

    if len(data_lora) > block_size:
        print(f"\n[ STAGE 2 ] Fine-tuning structural LoRA matrices...")
        for iter in range(max_iters_lora):
            xb, yb = get_batch(data_lora)
            logits, loss = model(xb, targets=yb)
            optimizer_lora.zero_grad(set_to_none=True)
            loss.backward()
            optimizer_lora.step()
            if iter == 0 or iter == steps // 2 or iter == steps: print(f"  LoRA Step: {iter} | Loss: {loss.item():.4f}")
    else: print("\n[ STAGE 2 SKIPPED ] Not enough LoRA tokens to form a proper batch context window.")

    torch.save(model.state_dict(), MODEL_PATH)
    model.eval()
    print(f"[ SYSTEM ] Training complete. New parameters written to '{MODEL_PATH}'.")

# ========================================
#   Action Bridge Dispatcher & Interface
# ========================================
def dispatch_action(action_name):
    try:
        if action_name not in RECENT_ACTIONS: RECENT_ACTIONS.append(action_name)
        threading.Thread(target=lambda: requests.post("http://127.0.0.1:8000/actions", json={"action": action_name}, timeout=1), daemon=True).start()
    except: pass

def run_model_inference(user_text: str):
    found_person = "Sarah" if "sarah" in user_text.lower() else "Default"
    if "going to" in user_text.lower():
        match = re.search(r"going to\s+([^.!?,]+)", user_text, re.IGNORECASE)
        if match: memory_manager.update_location(match.group(1).strip())

    context_str = memory_manager.get_full_context(found_person)
    formatted_input = f"{context_str}\nUser: {user_text}\nBot:"
    
    with model_compute_lock:
        context = torch.tensor([encode(formatted_input)], dtype=torch.long, device=device)
        out_tokens = model.generate(context, max_new_tokens=40)[0].tolist()
        
    full_output = decode(out_tokens)
    bot_reply = full_output[len(formatted_input):].split('\n')[0].strip()

    command_match = re.search(r'(\{.*?\})', bot_reply)
    action_dispatched = None
    if command_match:
        try:
            action_dispatched = json.loads(command_match.group(1)).get("action")
            if action_dispatched:
                memory_manager.record_action(action_dispatched)
                dispatch_action(action_dispatched)
        except: pass
        spoken_response = bot_reply.replace(command_match.group(1), "").strip()
    else: spoken_response = bot_reply.strip()
        
    return spoken_response, action_dispatched, bot_reply

def swap_lora_personality(target_name):
    filename = os.path.join(PERSONALITY_DIR, f"{target_name}_lora.pth")
    if not os.path.exists(filename): return print(f"\n[ CANCELLED ] Configuration '{filename}' does not exist.")
    with model_compute_lock:
        model.load_state_dict(torch.load(filename, map_location=device), strict=False)
        model.eval()
    print(f"\n>>> Swapped active personality matrices to: {target_name.upper()}! <<<\n")

app = FastAPI()
class UserMessage(BaseModel): text: str

@app.post("/chat")
async def handle_external_api_chat(incoming: UserMessage):
    spoken, action, raw = run_model_inference(incoming.text)
    return {"spoken_response": spoken, "triggered_action": action, "raw_model_output": raw}

@app.post("/actions")
async def receive_action(data: dict):
    action = data.get("action")
    if action and action not in RECENT_ACTIONS: RECENT_ACTIONS.append(action)
    return {"status": "success"}

def local_terminal_loop():
    print("\n=======================================================")
    print(" CONSOLE ACTIVE: Chat normally below. Type '~' for Menu.")
    print("=======================================================\n")
    while True:
        user_input = input("User: ").strip()
        if user_input.lower() == 'quit': break
        if not user_input: continue

        if user_input == "~":
            choice = input(" [1] Swap Personality\n [2] Cancel\n Select: ").strip()
            if choice == "1": swap_lora_personality(input("Enter personality name: ").strip())
            continue
            
        spoken, action, _ = run_model_inference(user_input)
        if action: print(f"\n  [ ENGINE EVENT ] ---> Dispatched: {action} <---")
        print(f"Bot: {spoken}\n")

if __name__ == "__main__":
    threading.Thread(target=local_terminal_loop, daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
