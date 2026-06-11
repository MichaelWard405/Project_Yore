import os
import re
import json
from numpy._core.multiarray import scalar
from pydantic.type_adapter import P
import torch
import threading
import warnings
import logging
from torch.fx.node import Target
import uvicorn
import torch.nn as nn
import numpy as np
from datetime import datetime
from transformers import logging as transformers_logging
from torch.nn import functional as F 
from tokenizers import Tokenizer
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
#===================================
#  Warning And Logging Suppression
#===================================
os.environ["TOEKNIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("peft").setLevel(logging.ERROR)
transformers_logging.set_verbosity_error()
#=====================
#  Environment Setup
#=====================
#[Neural Net Files]
NEURAL_FOLDER = "Neural_Net"
MEMORIES_DIR = "Memories"
PERSONALITY_DIR = os.path.join(NEURAL_FOLDER, "Personalities")
LOCAL_MODEL_DIR = os.path.join(NEURAL_FOLDER, "Local_Model")
CONTEXT_DIR = os.path.join(NEURAL_FOLDER, "Context")
#[Data Collection Paths]
DATA_COLLECTION_DIR = "Data_Collection"
CLEAN_DATA = os.path.join(DATA_COLLECTION_DIR, "Clean_Data")
BIN_TRANSCRIPTS = os.path.join(DATA_COLLECTION_DIR, "Binary_Transcripts")
#[Custom Model File/Paths][Due to Change]
MODEL_PATH = os.path.join(PERSONALITY_DIR, "custom_base.pth")
CUSTOM_LORA_PATH = os.path.join(PERSONALITY_DIR, "custom_lora.jsonl")
TOKENIZER_PATH = os.path.join(NEURAL_FOLDER, "custom_bpe.json")
#[Redundancy]
def redundant_enviroment():
    for directory in [NEURAL_FOLDER, MEMORIES_DIR, PERSONALITY_DIR, LOCAL_MODEL_DIR, CONTEXT_DIR, DATA_COLLECTION_DIR, CLEAN_DATA, BIN_TRANSCRIPTS]:
        os.makedirs(directory, exist_ok=True)
redundant_enviroment()
#===================
#  HyperParameters
#===================
batch_size = 8
block_size = 256
iterrations = 2000
max_iters_pretrain = iterrations
max_iters_LoRA = iterrations
learning_rate = 3e-4
n_embd = 768
n_head = 12
n_layer = 12
dropout = 0.1
device = "cuda" if torch.cuda.is_available() else "cpu"
#[Global State]
model_compute_lock = threading.Lock()
MODEL_MODE = "CATO"
local_model = None
local_tokenizer = None
custom_model = None
custom_tokenizer = None
vocab = 32000
TRUE_TEXT = False
#=================
#  Memory System
#=================
class MemoryManager:
#[Folder Init]
    def __init__(self, mem_dir = MEMORIES_DIR):
        self.mem_dir = mem_dir
        self.people_matrix = os.path.join(mem_dir, "PEOPLE[MATRIX].json")
        self.location_matrix = os.path.join(mem_dir, "LOCATIONS[MATRIX].json")
        self._init_defaults()
#[Memory Default init]
    def _init_defaults(self):
    #[Construct location Defaults]        
        if not os.path.exists(self.location_matrix):
            with open(self.location_matrix, 'w', encoding = 'utf-8') as f:
                json.dump({"Current_Location": "Default", "History": []}, f, indent = 4)
    #[Construct People Defaults]
        if not os.path.exists(self.people_matrix):
            with open(self.people_matrix, 'w', encoding = 'utf-8') as f:
                json.dump({"Default":{"Relationship": "Unknown", "Memories":[], "Emotions":[]}}, f, indent = 4)
#[Context System]
    def context(self, current_person = "Default"):
        try:
    #[Location Context]
            with open(self.location_matrix, 'r', encoding = 'utf-8') as f:
                location_matrix = json.load(f).get("Current_Location", "Unknown")
    #[People Context]
            with open(self.people_matrix, 'r', encoding = 'utf-8') as f:
                people_data = json.load(f)

                if not isinstance(people_data, dict):
                    people_data = {"Default": {"Relationship": "Unknown", "Memories":[], "Emotions":[]}}
                user_data = people_data.get(current_person, {"Relationship": "Unknown", "Memories":[], "Emotions":[]})
    #[Memory Read]
            mems = " | ".join(user_data.get('Memories', [])) or "None"
            emos = " | ".join(user_data.get('Emotions', [])) or "None"
            return f"Location: {location_matrix} | User Relationship: {user_data.get('Relationship')} | Recent Memories: {mems} | Emotions Towards User: {emos}"
        except Exception as e:
            return"[Systems Context Online]"
#[Append User Context]
    def update_interaction(self, person, mem_text=None, emo_text=None):
        try:
            with open(self.people_matrix, 'r', encoding = 'utf-8') as f:
                data = json.load(f)
            if not isinstance(data, dict): data = {}
            if person not in data:
                data[person] = {"Relationship": "Acquaintance", "Memories": [], "Emotions": []}
            extracted_mems = mem_text if mem_text else f"Met with {person}"
            extracted_emos = emo_text if emo_text else "Neutral"
            data[person]["Memories"].append(extracted_mems)
            data[person]["Emotions"] = [extracted_emos]
            with open(self.people_matrix, 'w', encoding = 'utf-8') as f:
                json.dump(data, f, indent = 4)
        except Exception as e:
            pass
memory_manager = MemoryManager()
#========================
#  Architectural layers
#========================
class LoRALinear(nn.Module):
#[Layer Initialization]
    def __init__(self, in_features, out_features, r = 8, lora_alpha = 16, dropout_rate = 0.05):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.r = r
        self.scaling = lora_alpha / r
        self.lora_A = nn.Parameter(torch.randn(in_features, r) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(r, out_features))
        self.dropout = nn.Dropout(dropout_rate)
#[LoRA Forward Pass]
    def forward(self, x):
        base_out = self.linear(x)
        lora_out = (self.dropout(x) @ self.lora_A @ self.lora_B) * self.scaling
        return base_out + lora_out
class Head(nn.Module):
#[Head Initialization]
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = LoRALinear(n_embd, head_size, r=8, lora_alpha=16)
        self.value = LoRALinear(n_embd, head_size, r=8, lora_alpha=16)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
#[Attention Foward Pass]
    def forward(self, x):
        B, T, C = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        return self.dropout(wei @ v)
class MultiHeadAttention(nn.Module):
#[Multi-Head Initialization]
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x): return self.dropout(self.proj(torch.cat([h(x) for h in self.heads], dim=-1)))
class FeedForward(nn.Module):
#[Feed Foward Initialization]
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.ReLU(), nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout))
    def forward(self, x): return self.net(x)
class Block(nn.Module):
#[Transformer Block Initialization]
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
#[Residual Block Pass]
    def forward(self, x): return x + self.sa(self.ln1(x)) + self.ffwd(self.ln2(x))
class CustomModel(nn.Module):
#[Model Initialization]
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) 
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights) 
#[Weight Initialization]
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None: torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
#[LoRA Parameter Freeze]
    def freeze_base_for_lora(self):
        for name, param in self.named_parameters():
            if 'lora_A' not in name and 'lora_B' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
#[Model Foward & loss]
    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_embedding_table(idx) + self.position_embedding_table(torch.arange(T, device=device))
        x = self.ln_f(self.blocks(x))
        logits = self.lm_head(x) 
        loss = F.cross_entropy(logits.view(B*T, logits.shape[-1]), targets.view(B*T)) if targets is not None else None
        return logits, loss
#[Token Generation Loop]
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -block_size:])
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, num_samples=1)), dim=1)
        return idx
#=============================
#  DataSet Loading Utilities
#=============================
#[Generate Data Batch]
def get_batch(data_source, source_type = "memmap", b_size = batch_size):
    #[Memmap Slicing]
    if source_type == "memmap":
        ix = torch.randint(0, len(data_source) - block_size, (b_size))
        x = torch.stack([torch.from_numpy((data_source[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data_source[i+1:i+block_size+1]).astype(np.int64)) for i in ix])
    #[Tensor Slicing]
    else:
        ix = torch.randint(0, len(data_source) - block_size, (b_size))
        x = torch.stack([data_source[i:i+block_size] for i in ix])
        y = torch.stack([data_source[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)
#[Tokenize JSONL DataSet]
def get_jsonl_data(filepath, encode_func):
    tokens = []
    #[Parse & Encode]
    with open(filepath, 'r', encoding = 'utf-8') as f:
        for line in f:
            if line.strip():
                try: tokens.extend(encode_func(json.loads(line)["text"] + "\n"))
                except: pass
    return torch.tensor(tokens, dtype = torch.long)
#================================
#  Selection Menu's & Ecosystem
#================================
print("\n==============================")
print("        System StartUp Menu     ")
print("================================")
print(" [1] Local Model Cato (4-Bit)")
print(" [2] Custom Model Generation (WIP)")
print("================================")
ecosystem_choice = input("Select An Option: ").strip()
#[Cato Ecosystem Branch]
if ecosystem_choice == "1":
    MODEL_MODE = "CATO"
    if not os.path.exists(LOCAL_MODEL_DIR) or len(os.listdir(LOCAL_MODEL_DIR)) == 0:
        exit()
    print("\n=============================")
    print("     Cato Operational Mode     ")
    print("===============================")
    print(" [1] Boot Base Model")
    print(" [2] Boot With LoRA Adapter")
    print(" [3] Train New LoRA Adapter")
    print("===============================")
    local_mode = input("Select a Mode: ").strip()
#[Configure 4-Bit Quantization]
    bnb_config = BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_compute_dtypes=torch.bfloat16,
        bnb_4bit_use_double_quant = True
    )
#[Initialize Tokenizer & Base Weights]
    local_tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR, local_files_only = True)
    base_model = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL_DIR,
        quantization_config = bnb_config,
        device_map = "auto",
        local_files_only = True
    )
    if local_tokenizer.pad_token is None:
        local_tokenizer.pad_token = local_tokenizer.eos_token
    local_tokenizer.padding_side = 'left'
    #[Mode 1: Boot Base Model]
    if local_mode == "1":
        local_model = base_model
    #[Mode 2: Boot With LoRA Adapter]
    elif local_mode == "2":
        adapters = [f for f in os.listdir(CLEAN_DATA) if os.path.isdir(os.path.join(CLEAN_DATA, f))]
        if not adapters:
            local_model = base_model
        else:
            print("\nAvailiable Adapters:")
            for i, name in enumerate(adapters, 1): print(f"[{i}] {name}")
            try: target = os.path.join(CLEAN_DATA, adapters[int(input("Selection: ").strip()) - 1])
            except: target = os.path.join(CLEAN_DATA, adapters[0])
            local_model = PeftModel.from_pretrained(base_model, target)
    #[Mode 3: Train New LoRA Adapter]
    elif local_mode == "3":
        jsonls = [f for f in os.listdir(CLEAN_DATA) if f.endswith(".jsonl")]
        if not jsonls:
            exit()
        print("\n[CATO] Available Datasets:")
        for i, name in enumerate(jsonls, 1):print(f" [{i}] {name}")
        try: target_data = os.path.join(CLEAN_DATA, jsonls[int(input("Selection: ").strip()) - 1])
        except: exit()
#[Prepare for Quantized Training]
        torch.cuda.empty_cache()
        base_model = prepare_model_for_kbit_training(base_model)
#[Configure Adapter Topography]
        peft_config = LoRAConfig(
            r = 16, lora_alpha = 32, target_modules = ["q_proj", "v_proj"],
            lora_dropout = 0.05, bias = "none", task_type = "CASUAL_LM"
        )
        local_model = get_peft_model(base_model, peft_config)
        encoded_inputs = []
        with open(target_data, 'r', encoding = 'utf-8') as f:
            for line in f:
                if line.strip():
                    try: encoded_inputs.append(local_tokenizer(json.loads(line)["text"], return_tensors = 'pt', max_length = block_size, truncation = True))
                    except: pass
        local_model.train()
        optimizer = torch.optim.AdamW(local_model.parameters(), lr = learning_rate)
        scaler = torch.cuda.amp.GradScaler()
        local_train_batch_size = 2
#[Local LoRA Training Loop]
        for step in range(max_iters_LoRA):
            b_ix = torch.randint(0, len(encoded_inputs), (local_train_batch_size))
            i_ids = [encoded_inputs[i]['input_ids'][0] for i in b_ix]
            a_mask = [encoded_inputs[i]['attention_mask'][0] for i in b_ix]
            m_len = max(len(x) for x in i_ids)
            p_inputs = torch.stack([torch.cat([x, torch.full((m_len - len(x), ), local_tokenizer.pad_token_id, dtype = torch.long)]) for x in i_ids]).to(device)
            p_masks = torch.stack([torch.cat([m, torch.zeros(m_len - len(m), dtype=torch.long)]) for m in a_mask]).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                loss = local_model(input_ids=p_inputs, attention_mask=p_masks, labels=p_inputs).loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if step % 100 == 0: print(f"Step {step:04d} | Adapter Loss: {loss.item():.4f}")
        out_dir = os.path.join(CLEAN_DATA, f"[CATO] Adapter_{int(datetime.now().timestamp())}")
        local_model.save_pretrained(out_dir)
        local_model.eval()
#[Custom Model Ecosystem Branch]
elif ecosystem_choice == "2":
    MODEL_MODE = "custom"
#[Tokenizer Initialization]
    if os.path.exists(TOKENIZER_PATH):
        custom_tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
        custom_vocab_size = custom_tokenizer.get_vocab_size()
    else:
        exit()
    custom_model = CustomModel(custom_vocab_size).to(device)
#[Operational Menu]
    print("\n====================================")
    print("   Custom Model Operational Menu")
    print("====================================")
    print(" [1] Boot Model")
    print(" [2] Pretrain Base Brain")
    print(" [3] Fine-Tune LoRA Personality")
    print("====================================")
    c_mode = input("Select A Mode: ").strip().upper()
    if c_mode in ["1", "3"]:
        if os.path.exists(MODEL_PATH):
            custom_model.load_state_dict(torch.load(MODEL_PATH, map_location = device), strict = False)
            if os.path.exists(CUSTOM_LORA_PATH):
                custom_model.load_state_dict(torch.load(CUSTOM_LORA_PATH, map_location = device), strict = False)
    if c_mode == "2":
        bins = [f for f in os.listdir(BIN_TRANSCRIPTS) if f.endswith(".bin")]
        if not bins: exit()
        print("\nSelect Pretain Base")
        for i, f in enumerate(bins, 1): print(f" [{i}] {f}")
        try: target_bin = os.path.join(BIN_TRANSCRIPTS, bins[int(input("Selection: ").strip()) - 1])
        except: exit()
        for name, param in custom_model.named_parameters():
            if 'lora_' in name: param.requires_grad = False
            else: param.requires_grad = True
        data_mmap = np.memmap(target_bin, dtype = np.int32, mode = 'r')
        opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, custom_model.parameters()), lr = learning_rate)
        for step in range(max_iters_pretrain):
            xb, yb = get_batch(data_mmap, "memmap")
            logits, loss = custom_model(xb, targets = yb)
            opt.zero_grad(set_to_none = True)
            loss.backward()
            opt.step()
            if step % 100 == 0: print(f"  Base Pretrain Step {step:04d} | Loss: {loss.item():.4f}")
        torch.save(custom_model.state_dict(), MODEL_PATH)
    elif c_mode == "3":
        jsonls = [f for f in os.listdir(CLEAN_DATA) if f.endswith(".jsonl")]
        if not jsonls: exit()
        print("\nSelect Fine-Tune Data:")
        for i, f in enumerate(jsonls, 1): print(f" [{i}] {f}")
        try: target_data = os.path.join(CLEAN_DATA, jsonls[int(input("Selection: ").strip()) - 1])
        except: exit()
        custom_model.freeze_base_for_lora()
        opt_lora = torch.optim.AdamW(filter(lambda p: p.requires_grad, custom_model.parameters()), lr = learning_rate)
        encoding_func = lambda s: custom_tokenizer.encode(s).ids
        lora_tensor = get_jsonl_data(target_data, encoding_func)
        if len(lora_tensor) > block_size:
            for step in range(max_iters_LoRA):
                xb, yb = get_batch(lora_tensor, "tensor")
                logits, loss = custom_model(xb, targets = yb)
                opt_lora.zero_grad(set_to_none = True)
                loss.backward()
                opt_lora.step()
                if step % 100 == 0: print(f"  LoRa Step {step:04d} | Loss: {loss.item():.4f}")
        torch.save(custom_model.state_dict(), CUSTOM_LORA_PATH)
    custom_model.eval()
#====================
#  Inference Engine
#====================
def run_model_inference(user_text: str, current_user: str = "Default", is_launch: bool = False):
    with model_compute_lock:
        context_str = memory_manager.context(current_user)
        eval_text = user_text if user_text.strip() else "Hello"
        if MODEL_MODE == "CATO":
            CONTEXT_FILE = os.path.join(CONTEXT_DIR, "Context.txt")
            with open(CONTEXT_FILE, 'r', encoding = 'utf-8') as f:
                contxt = f.read()
            contxt = contxt.replace("{current_user}", current_user)
            matrix_context = f"{contxt}\n\n[System Matrix Context]: {context_str}\n\n{current_user}: {eval_text}\nCato:"
            inputs = local_tokenizer(matrix_context, return_tensors = "pt").to(device)
            with torch.no_grad():
                outputs = local_model.generate(**inputs, max_new_tokens = 150, pad_token_id = local_tokenizer.eos_token_id)
            input_length = inputs['input_ids'].shape[1]
            raw_reply = local_tokenizer.decode(outputs[0][input_length:], skip_special_tokens = True).strip()
        else:
            contxt = f"System Context: The User is {current_user}, {context_str}\nUser: {eval_text}\nCato:"
            encoded_contxt = custom_tokenizer.encode(contxt).ids
            context = torch.tensor([encoded_contxt], dtype = torch.long, device = device)
            with torch.no_grad():
                out_tokens = custom_model.generate(context, max_new_tokens = 150)[0].tolist()
            raw_reply = custom_tokenizer.decode(out_tokens)[len(contxt):].strip()
        reply = re.sub(r"<\|.*?\|>", "", raw_reply).strip()
        truncation_pattern = re.compile(rf"(\n{current_user}:|\nUser:|{current_user}:|User:)", re.IGNORECASE)
        match = truncation_pattern.search(reply)
        if match:
            reply = reply[:match.start()].strip()
        #[Extract the Actions]
        action_match = re.search(r"\{Action:\s*(.*?)}|\{\"action\"\s*:\s*\"(.*?)\"}", reply, re.IGNORECASE)
        action = (action_match.group(1) or action_match.group(2)).strip() if action_match else None
        spoken_reply = re.sub(r"\{Action:\s*.*?\}|\{\"action\"\s*:\s*\".*?\"\}", "", reply, flags = re.IGNORECASE)
        spoken_reply = re.sub(r"\{\"Memories\"\s*:\s*\".*?\"\}", "", spoken_reply, flags = re.IGNORECASE)
        spoken_reply = re.sub(r"\{\"Emotions\"\s*:\s*\".*?\"\}", "", spoken_reply, flags = re.IGNORECASE)
        spoken_reply = re.sub(r"Note:.*", "", spoken_reply, flags = re.IGNORECASE)
        spoken_reply = re.sub(r'\s+', ' ', spoken_reply).strip()
        if user_text.strip(): memory_manager.update_interaction(current_user, user_text, reply)
        return spoken_reply, action, raw_reply if MODEL_MODE == "CATO" else contxt
#=======================
#  API Gateway Routing
#=======================
app = FastAPI()
class UserMessage(BaseModel):
    text:str
    user_name:str = "Default"
@app.post("/chat")
async def handle_chat(incoming: UserMessage):
    reply, action, raw = run_model_inference(incoming.text, current_user = incoming.user_name)
    return {"spoken_response": reply, "action": action}
#====================
#  Session Terminal
#====================
def Local_Terminal_Loop():
    global TRUE_TEXT
    active_user = input("\nEnter Your Name: ").strip() or "Michael"
    print(f'\n[CATO] [SYSTEM] FrameWork: "{MODEL_MODE.upper()}"')
    try:
        spoken_greeting, action, raw_reply = run_model_inference("", current_user = active_user, is_launch = True)
        print(f"\n[CATO]: {spoken_greeting}\n")
    except Exception as e:
        print("[CATO] System Online... Ready")
    while True:
        user_input = input(f"\n{active_user}: ").strip()
        if user_input.lower() == 'quit': break
        if not user_input: continue
        if user_input.lower() == 'admin':
            print("=================================")
            print(" [1] Toggle True Text [Debug]")
            print(" [2] Swap Active User")
            print(" [3] Return To Active Session")
            print("=================================")
            admin_choice = input("Select Choice Index (1 - 3): ").strip()
            if admin_choice == "1":
                TRUE_TEXT = not TRUE_TEXT
            elif admin_choice == "2":
                new_user = input("Enter New User Name: ").strip()
                if new_user:
                    active_user = new_user
                    spoken_greeting, action, raw_reply = run_model_inference("", current_user = active_user, is_launch = True)
                    print(f"[CATO] {spoken_greeting}\n")
            continue
        spoken, action, raw_reply = run_model_inference(user_input, current_user = active_user)
        if action: print(f"\n[CATO] [ENGINE] --> Dispatched Action: {action} <---")
        print(f"[CATO] {spoken}\n" if not TRUE_TEXT else f"\n [CATO] [RAW] -> {raw_reply}\n[CATO] {spoken}\n")
if __name__ == "__main__":
    threading.Thread(target = Local_Terminal_Loop, daemon = True).start()
    uvicorn.run(app, host = "127.0.0.1", port = 8000, log_level = "warning")
