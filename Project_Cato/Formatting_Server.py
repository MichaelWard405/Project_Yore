import json
from http.server import BaseHTTPRequestHandler, HTTPServer
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

MODEL_ID = "unsloth/Llama-3.2-3B-Instruct-unsloth-bnb-4bit"
PORT = 5000

print("Initializing local structural text processing network...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)
print("Formatting Server completely cached on device VRAM.")

class FormattingServerHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        """Standard pipeline sync checks."""
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({"status": "ready"}).encode('utf-8'))

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data.decode('utf-8'))
        raw_line = data.get("text", "").strip()
        
        system_prompt = (
            "You are Cato, a structured state-tracking engine. Analyze the dialogue text "
            "and convert it EXACTLY into the following multi-object tracking line format. Do not use markdown wrappers. "
            "Output everything strictly on a single continuous string line.\n\n"
            "REQUIRED LAYOUT:\n"
            '(Speaker_Type)"Cleaned Dialogue Text"{"action":"current action"}, {"Speaker":"person names", "target":"targeted party", "memories":"new memories", "emotions":"contextual emotion"}, {"Save_Person":"None"}, {"locations": "Brisbane", "Current_Location":"Brisbane", "memories":"None"}, {"save_Location":"None"}, {"emotion":"[happy: 100%]"}\n\n'
            "Rules:\n"
            "1. Choose either 'User' or 'bot' for Speaker_Type.\n"
            "2. Ensure all fields default to 'None' if context is unknown while keeping JSON syntax structurally pristine."
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Convert this raw string now: '{raw_line}'"}
        ]
        
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=350,
                eos_token_id=tokenizer.eos_token_id, 
                pad_token_id=tokenizer.pad_token_id,
                temperature=0.1,  
                do_sample=True
            )
        
        clean_output = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        clean_output = clean_output.replace("\n", " ").replace("\r", "")
        
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({"formatted_text": clean_output}).encode('utf-8'))

def run_server(port=PORT):
    server_address = ('', port)
    httpd = HTTPServer(server_address, FormattingServerHandler)
    print(f"Server executing cleanly on port {port}...")
    try: httpd.serve_forever()
    except KeyboardInterrupt: httpd.server_close()

if __name__ == "__main__":
    run_server()

