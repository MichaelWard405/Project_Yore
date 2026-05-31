import os
import sys
import subprocess
from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Header, Footer, Button, Select, RichLog, Input, Label

# =========================
#   Interactice Interface
# =========================
class ScriptPanel(Vertical):


    def __init__(self, script_files, panel_num, **kwargs):
        super().__init__(**kwargs)
        self.script_files = script_files
        self.panel_num = panel_num
        self.process = None

    def compose(self) -> ComposeResult:
        options = [(f, f) for f in self.script_files]
        
        yield Label(f"📦 Pipeline Slot #{self.panel_num}", id="panel-title")
        yield Select(options, prompt="Select Script...", id="script-selector")
        
        with Horizontal(id="controls"):
            yield Button("Run", variant="success", id="run-btn")
            yield Button("Kill", variant="error", id="stop-btn")
            yield Button("Close", variant="primary", id="close-btn")
            
        yield RichLog(highlight=True, markup=True, id="output-log")
        yield Input(placeholder="Type input & press Enter...", id="stdin-input")

    def on_click(self) -> None:
        self.query_one("#stdin-input").focus()

    @on(Button.Pressed, "#run-btn")
    def start_script(self) -> None:
        select = self.query_one("#script-selector", Select)
        log = self.query_one("#output-log", RichLog)
        
        if not select.value or select.value == Select.BLANK:
            log.write("[SYSTEM ERROR] Select a target module first.")
            return
            
        if self.process and self.process.poll() is None:
            log.write("[SYSTEM WARNING] Process already executing in this slot.")
            return
            
        log.clear()
        log.write(f"[SYSTEM] Streaming from: {select.value}...")
        self.execute_pipeline_worker(select.value, log)

    @work(thread=True)
    def execute_pipeline_worker(self, script_name: str, log_widget: RichLog) -> None:
        try:

            venv_dir = os.path.abspath(os.path.join("Master", ".venv"))
            
            if os.name == "nt":
                python_exe = os.path.join(venv_dir, "Scripts", "python.exe")
                bin_dir = os.path.join(venv_dir, "Scripts")
            else:
                python_exe = os.path.join(venv_dir, "bin", "python")
                bin_dir = os.path.join(venv_dir, "bin")


            current_env = os.environ.copy()

            if os.path.exists(python_exe):
                current_env["PATH"] = bin_dir + os.pathsep + current_env.get("PATH", "")
                current_env["VIRTUAL_ENV"] = venv_dir
                current_env.pop("PYTHONHOME", None)
            else:
                self.app.call_from_thread(
                    log_widget.write, 
                    f"[SYSTEM WARNING] '{python_exe}' not found.\nFalling back to primary system Python execution environment...\n"
                )
                python_exe = sys.executable

            self.process = subprocess.Popen(
                [python_exe, "-u", script_name],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=current_env
            )
            
            while self.process and self.process.poll() is None:
                line = self.process.stdout.readline()
                if line:
                    self.app.call_from_thread(log_widget.write, line.strip("\n"))
                    
            if self.process:
                code = self.process.wait()
                self.app.call_from_thread(log_widget.write, f"\n[PROCESS TERMINATED: Code {code}]")
        except Exception as e:
            self.app.call_from_thread(log_widget.write, f"\n[CRASH]: {str(e)}")

    @on(Button.Pressed, "#stop-btn")
    def stop_script(self) -> None:
        log = self.query_one("#output-log", RichLog)
        if self.process and self.process.poll() is None:
            self.process.terminate()
            log.write("[SYSTEM] Sent SIGTERM instruction.")
            self.process = None
        else:
            log.write("[SYSTEM] No active runtime thread found.")

    @on(Button.Pressed, "#close-btn")
    def close_panel(self) -> None:
        self.stop_script()
        self.app.remove_panel(self)

    @on(Input.Submitted, "#stdin-input")
    def handle_stdin_routing(self, event: Input.Submitted) -> None:
        input_widget = event.input
        user_text = event.value
        log = self.query_one("#output-log", RichLog)
        
        if self.process and self.process.poll() is None:
            try:
                self.process.stdin.write(user_text + "\n")
                self.process.stdin.flush()
                log.write(f"[INPUT] ➜ {user_text}")
            except Exception as e:
                log.write(f"[ROUTING ERROR]: {e}")
        else:
            log.write("[INPUT REJECTED] Process is not running.")
            
        input_widget.value = ""

    def on_unmount(self) -> None:
        if self.process and self.process.poll() is None:
            try:
                self.process.kill()
            except Exception:
                pass 

# ======================
#   Panel Auto-Scaling
# ======================
class MasterInterfaceApp(App):
    
    TITLE = "PROJECT YORE: GRUVBOX WRAPPING MATRIX"
    theme = "gruvbox"
    ENABLE_COMMAND_PALETTE = False
    
    BINDINGS = [
        ("n", "add_panel", "Spawn Panel Slot"),
        ("tab", "focus_next", "Cycle UI Focus"),
        ("q", "quit", "Exit Terminal Multiplexer")
    ]
    
    CSS = """
    #grid-container {
        layout: grid;
        grid-gutter: 1;       
        width: 100%;
        height: 100%;
        padding: 1;
        background: $background;
    }
    
    ScriptPanel {
        border: round $accent-darken-2;
        padding: 1;
        background: $surface;
        transition: border 0.1s, background 0.1s;
    }
    
    ScriptPanel:focus-within {
        border: double $success;
        background: $surface-lighten-1;
    }
    
    #panel-title {
        text-align: center;
        text-style: bold;
        color: $accent;
    }
    
    #controls {
        height: auto;
        margin: 1 0;
    }
    
    #controls Button {
        width: 1fr;
        margin: 0 1;
    }
    
    RichLog {
        height: 1fr;
        background: $background;
        border: solid $surface-darken-2;
    }
    
    Input {
        border: tall $primary;
        margin-top: 1;
    }
    """

    def scan_for_scripts(self):
        ignored_files = {"setup_cato.py", "Master_Interface.py"}
        return [
            f for f in os.listdir(".") 
            if f.endswith(".py") and f not in ignored_files
        ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Container(id="grid-container"):
            self.panel_count = 1
            py_files = self.scan_for_scripts()
            yield ScriptPanel(py_files, self.panel_count)
        yield Footer()

    def on_mount(self) -> None:
        self.recalculate_grid_dimensions()

    def recalculate_grid_dimensions(self) -> None:
        container = self.query_one("#grid-container")
        count = self.panel_count
        
        if count <= 4:
            cols = count
            rows = 1
        else:
            cols = 4
            rows = 2
            
        container.styles.grid_size_columns = cols
        container.styles.grid_size_rows = rows

    def action_add_panel(self) -> None:
        if self.panel_count >= 8:
            self.notify("Matrix Exhausted: Maximum structural limit of 8 dashboard panels reached.", title="Grid Full", severity="warning")
            return
            
        self.panel_count += 1
        container = self.query_one("#grid-container")
        py_files = self.scan_for_scripts()
        
        self.recalculate_grid_dimensions()
        
        new_panel = ScriptPanel(py_files, self.panel_count)
        container.mount(new_panel)
        new_panel.scroll_visible()
        
        new_panel.query_one("#stdin-input").focus()

    def remove_panel(self, panel: ScriptPanel) -> None:
        if self.panel_count <= 1:
            self.notify("Cannot close the final primary window slot.", severity="warning")
            return
            
        self.panel_count -= 1
        panel.remove()
        self.recalculate_grid_dimensions()
        
        remaining = [p for p in self.query(ScriptPanel) if p != panel]
        if remaining:
            remaining[0].query_one("#stdin-input").focus()

if __name__ == "__main__":
    app = MasterInterfaceApp()
    app.run()
