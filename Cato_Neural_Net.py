import os
import re
import json
import torch
import threading
import torch.nn as nn
from torch.nn import functional as F
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# ===================
#   HYPERPARAMETERS
# ===================
batch_size = 32 #Expand values, With larger Data set #16 
block_size = 128 #512
max_iters_pretrain = 4000
max_iters_lora = 4000
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embd = 256 #768
n_head = 4  #6
n_layer = 8 #12
dropout = 0.1
vocab_size = 10000
steps = 3000


MODEL_PATH = "Neural_Net/my_chatbot_brain.pth"
TOKENIZER_PATH = "Neural_Net/custom_bpe.json"
PRETRAIN_FILE = "Neural_Net/pretrain_data.txt"
LORA_FILE = "Neural_Net/lora_data.txt"
NEURAL_FOLDER = "Neural_Net"

# Threading lock to prevent the model from training and inferencing simultaneously
model_compute_lock = threading.Lock()

# ========================================
#   AUTO-DATASET CREATION & VERIFICATION
# ========================================
def verify_datasets():
    if not os.path.exists(NEURAL_FOLDER):
        print(f"[SYSTEM] {NEURAL_FOLDER} not found. Generating {NEURAL_FOLDER}")
    if not os.path.exists(PRETRAIN_FILE):
        print(f"[ SYSTEM ] '{PRETRAIN_FILE}' not found. Generating default text...")
        dummy_pretrain = "The AI is learning to speak English. It learns grammar and sentence structure. " * 500
        with open(PRETRAIN_FILE, 'w', encoding='utf-8') as f: f.write(dummy_pretrain)
            
    if not os.path.exists(LORA_FILE):
        print(f"[ SYSTEM ] '{LORA_FILE}' not found. Generating default context commands...")
        dummy_lora = (
            "User: Hello!\nBot: Hi there! I am happy to see you. {\"action\": \"Smile\"}\n"
            "User: Look at this!\nBot: Wow, what is that? {\"action\": \"Blink\"}\n"
            "User: Goodbye!\nBot: See you later! {\"action\": \"Wave\"}\n"
        ) * 200
        with open(LORA_FILE, 'w', encoding='utf-8') as f: f.write(dummy_lora)

verify_datasets()

# ==========================================
#  TOKENIZER ARCHITECTURE
# ==========================================
use_existing_bpe = False
if os.path.exists(TOKENIZER_PATH):
    choice = input(f"Found existing '{TOKENIZER_PATH}'. Load it? (y/n): ").strip().lower()
    if choice == 'y': use_existing_bpe = True

if use_existing_bpe:
    print(f"[ SYSTEM ] Loading existing tokenizer from '{TOKENIZER_PATH}'...")
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
else:
    print(f"[ SYSTEM ] Building a NEW BPE tokenizer from your text documents...")
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"], min_frequency=2)
    tokenizer.train([PRETRAIN_FILE, LORA_FILE], trainer)
    tokenizer.save(TOKENIZER_PATH)

vocab_size = tokenizer.get_vocab_size()
encode = lambda s: tokenizer.encode(s).ids
decode = lambda l: tokenizer.decode(l)

# Read and vectorize baseline contents
with open(PRETRAIN_FILE, 'r', encoding='utf-8') as f: data_pretrain = torch.tensor(encode(f.read()), dtype=torch.long)
with open(LORA_FILE, 'r', encoding='utf-8') as f: data_lora = torch.tensor(encode(f.read()), dtype=torch.long)

def get_batch(data_source):
    ix = torch.randint(len(data_source) - block_size, (batch_size,))
    x = torch.stack([data_source[i:i+block_size] for i in ix])
    y = torch.stack([data_source[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# ==========================================
#  NEURAL NETWORK WITH INTEGRATED LORA
# ==========================================
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False, r=4, lora_alpha=8):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.r = r
        self.scaling = lora_alpha / r
        self.lora_A = nn.Parameter(torch.randn(in_features, r) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(r, out_features))
    def forward(self, x):
        return self.linear(x) + ((x @ self.lora_A @ self.lora_B) * self.scaling)

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

# ==========================================
#  RETRAINING & INITIALIZATION CHOICES
# ==========================================
should_train = True
if os.path.exists(MODEL_PATH):
    train_choice = input(f"Found existing weights file '{MODEL_PATH}'. Retrain model parameters? (y/n): ").strip().lower()
    if train_choice == 'n': should_train = False

if not should_train:
    print(f"[ SYSTEM ] Loading saved weights directly from '{MODEL_PATH}'...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
else:
    print(f"\n[ STAGE 1 ] Executing Pre-training loop from '{PRETRAIN_FILE}'...")
    optimizer_base = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    for iter in range(max_iters_pretrain):
        xb, yb = get_batch(data_pretrain)
        logits, loss = model(xb, targets=yb)
        optimizer_base.zero_grad(set_to_none=True)
        loss.backward()
        optimizer_base.step()
        if iter % steps == 0: print(f"  Base Step {iter} | Loss: {loss.item():.4f}")

    apply_lora_freezing(model)
    optimizer_lora = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate)

    print(f"\n[ STAGE 2 ] Fine-tuning structural LoRA matrices from '{LORA_FILE}'...")
    for iter in range(max_iters_lora):
        xb, yb = get_batch(data_lora)
        logits, loss = model(xb, targets=yb)
        optimizer_lora.zero_grad(set_to_none=True)
        loss.backward()
        optimizer_lora.step()
        if iter % steps == 0: print(f"  LoRA Step {iter} | Loss: {loss.item():.4f}")#Steps for the LoRA Pass 

    torch.save(model.state_dict(), MODEL_PATH)
    model.eval()
    print(f"[ SYSTEM ] Training complete. New parameters written to '{MODEL_PATH}'.")

# ==========================================
#  CENTRAL INFERENCE LOGIC Engine
# ==========================================
def run_model_inference(user_text: str):
    """Feeds text to the transformer model, logs interactions, and parses JSON actions."""
    formatted_input = f"User: {user_text}\nBot:"
    
    with model_compute_lock:
        context = torch.tensor([encode(formatted_input)], dtype=torch.long, device=device)
        out_tokens = model.generate(context, max_new_tokens=40)[0].tolist()
        
    bot_reply = decode(out_tokens)[len(formatted_input):].split('\n')[0].strip()
    
    # Track the raw interaction for future continuous learning calls
    with open("live_history.txt", "a", encoding="utf-8") as f:
        f.write(f"User: {user_text}\nBot: {bot_reply}\n\n")
        
    # Execute structural parsing for actions
    command_match = re.search(r'(\{.*?\})', bot_reply)
    action_dispatched = None
    if command_match:
        command_text = command_match.group(1)
        try:
            command_data = json.loads(command_text)
            action_dispatched = command_data.get("action")
        except json.JSONDecodeError:
            pass
        spoken_response = bot_reply.replace(command_text, "").strip()
    else:
        spoken_response = bot_reply.strip()
        
    return spoken_response, action_dispatched, bot_reply

# ==========================================
#  ADMINISTRATIVE TOOLSET MANAGEMENT
# ==========================================
def swap_lora_personality(target_personality_name):
    filename = f"{target_personality_name}_lora.pth"
    if not os.path.exists(filename):
        print(f"\n[ CANCELLED ] Configuration target '{filename}' does not exist.")
        return
    with model_compute_lock:
        lora_state = torch.load(filename, map_location=device)
        model.load_state_dict(lora_state, strict=False)
        model.eval()
    print(f"\n>>> SYSTEM: Swapped active personality matrices to: {target_personality_name.upper()}! <<<\n")

def run_evolution_merge():
    if not os.path.exists("live_history.txt") or os.path.getsize("live_history.txt") == 0:
        print("\n[ CANCELLED ] 'live_history.txt' contains no content to process.\n")
        return
    print("\n[ EVOLVING ] Merging runtime interactions and retraining LoRA parameters...")
    with model_compute_lock:
        with open("combined_training.txt", "w", encoding="utf-8") as master:
            if os.path.exists(LORA_FILE):
                with open(LORA_FILE, "r", encoding="utf-8") as core: master.write(core.read() + "\n")
            with open("live_history.txt", "r", encoding="utf-8") as history: master.write(history.read())

        with open("combined_training.txt", "r", encoding="utf-8") as f:
            data_evolve = torch.tensor(encode(f.read()), dtype=torch.long)

        model.train()
        optimizer_evolve = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate)
        for iter in range(400):
            xb, yb = get_batch(data_evolve)
            _, loss = model(xb, targets=yb)
            optimizer_evolve.zero_grad(set_to_none=True)
            loss.backward()
            optimizer_evolve.step()

        torch.save(model.state_dict(), MODEL_PATH)
        model.eval()
        open("live_history.txt", "w").close() # Clear temporary historical buffer
    print(">>> SYSTEM: Evolution processing sequence completed. <<<\n")

# ==========================================
#  PARALLEL CONTROL INTERFACES (CLI & API)
# ==========================================
app = FastAPI()

class UserMessage(BaseModel):
    text: str

@app.post("/chat")
async def handle_external_api_chat(incoming: UserMessage):
    """Allows external apps (Unity, Games, Scripts) to send inputs and read JSON commands."""
    spoken_text, action, raw_reply = run_model_inference(incoming.text)
    return {
        "spoken_response": spoken_text,
        "triggered_action": action,
        "raw_model_output": raw_reply
    }

def local_terminal_loop():
    """Direct console window execution thread allowing simultaneous admin menu actions."""
    print("\n=======================================================")
    print(" CONSOLE ACTIVE: Chat normally below.")
    print(" Type '~' to open the CLI System Admin Menu.")
    print("=======================================================\n")
    
    while True:
        user_input = input("User: ").strip()
        if user_input.lower() == 'quit': break
        if not user_input: continue

        if user_input == "~":
            print("\n=======================================================")
            print("             INTERNAL SYSTEM ADMIN MENU                ")
            print("=======================================================")
            print(" [1] Remember - Merge live history & retrain LoRA weights")
            print(" [2] Swap     - Swap active LoRA personality profile")
            print(" [3] Exit     - Close menu and return to active chat")
            print("=======================================================")
            choice = input("Select an action (1-3): ").strip()
            
            if choice == "1": run_evolution_merge()
            elif choice == "2":
                target = input("Enter target personality name (e.g., sassy): ").strip()
                swap_lora_personality(target)
            continue

        # Process a regular conversation string locally
        spoken, action, _ = run_model_inference(user_input)
        if action:
            print(f"\n  [ ENGINE EVENT ] ---> Action Dispatched: {action} <---")
        print(f"Bot: {spoken}\n")

if __name__ == "__main__":
    # Fire up the local terminal controller as a parallel background worker
    cli_thread = threading.Thread(target=local_terminal_loop, daemon=True)
    cli_thread.start()

    # Bind the main program thread to the FastAPI Web Server on your local network
    # Quiet uvicorn logging so it does not clutter your conversation console window
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
