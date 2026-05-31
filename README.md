# Project Yore


Project Yore is a modular framework for building, training, and deploying a conversational AI, referred to as Project Cato. It consists of a data acquisition engine, a customizable neural network with LoRA fine-tuning, and a sophisticated terminal-based master interface for managing all components.

The system is designed to be highly interactive and extensible, allowing users to gather training data from various sources, evolve the AI's personality through live interaction, and integrate the model with external applications via a web API.

## Core Components

1.  **`Data_Collector.py` (Unified Data Acquisition Engine):** A powerful command-line tool for gathering text data from multiple sources. It can transcribe video/audio from YouTube and Twitch, record and transcribe live streams, and perform deep crawls of documentation websites to build datasets.

2.  **`Cato_Neural_Net.py` (Neural Net Core):** A transformer-based chatbot built with PyTorch. It features Low-Rank Adaptation (LoRA) for efficient fine-tuning and personality-swapping. The model can be trained on custom datasets, learn from live conversations, and expose a FastAPI endpoint for integration with games or other software.

3.  **`Master_Interface.py` (Terminal Multiplexer):** A Textual-based Terminal User Interface (TUI) that acts as a central control panel. It allows you to run, monitor, and interact with multiple instances of the data collector and neural network in a dynamic, auto-scaling grid of panels.

4.  **`Project_Cato_Installer.py` (Installation Wizard):** A setup script that automates the entire installation process, including creating a virtual environment, installing all dependencies, and downloading the core project files.

## Features

*   **Modular Architecture:** Each component can be run and used independently.
*   **Automated Setup:** A single installer script prepares the entire environment and dependencies.
*   **Advanced Data Collection:**
    *   Transcribe pre-recorded media from YouTube/Twitch using `yt-dlp`.
    *   Record and transcribe live streams in real-time.
    *   Crawl websites for text data with session resumption and politeness delays.
    *   GPU-accelerated (CUDA) audio transcription via `faster-whisper`, with a CPU fallback.
*   **Customizable Neural Network:**
    *   Transformer model built with PyTorch.
    *   Efficient fine-tuning using Low-Rank Adaptation (LoRA).
    *   "Evolution" feature to merge live conversation history and retrain LoRA weights.
    *   Supports swappable "personality" profiles (different LoRA weights).
    *   Parses JSON actions from model output (e.g., `{"action": "Smile"}`).
*   **Seamless Integration:**
    *   Built-in FastAPI server to allow external applications (e.g., Unity, game engines) to interact with the AI.
*   **Modern TUI Control Panel:**
    *   Run multiple scripts simultaneously in a single terminal window.
    *   Dynamically add and remove control panels (up to 8).
    *   Send input and view live output for each managed process.

## Installation

The project includes an installer script that handles all setup automatically.

1.  Clone or download this repository.
2.  Navigate to the repository's root directory in your terminal.
3.  Ensure you have Python 3 installed.
4.  Run the installer:
    ```bash
    python Project_Cato_Installer.py
    ```
    The script will guide you through selecting which modules to install. It will then:
    *   Create a `Project_Cato/` directory.
    *   Set up a dedicated Python virtual environment in `Project_Cato/Master/.venv`.
    *   Install all necessary dependencies (`torch`, `textual`, `faster-whisper`, etc.).
    *   Download the latest versions of the project's scripts into the `Project_Cato/` directory.

## Usage

After the installation is complete, you can run the entire system through the Master Interface.

**1. Activate the Virtual Environment**

The installer creates a self-contained environment. You must activate it before running the scripts.

*   On **Windows**:
    ```cmd
    Project_Cato\Master\.venv\Scripts\activate
    ```

*   On **macOS / Linux**:
    ```bash
    source Project_Cato/Master/.venv/bin/activate
    ```

**2. Navigate to the Project Directory**

All commands should be run from inside the `Project_Cato` folder.

```bash
cd Project_Cato
```

**3. Launch the Master Interface**

Start the main TUI control panel:

```bash
python Master_Interface.py
```

**4. Running Scripts**

Inside the Master Interface:
*   Use the `Select Script...` dropdown in a panel to choose a script to run (e.g., `Data_Collector.py` or `Cato_Neural_Net.py`).
*   Click the **Run** button to start the script. Its output will stream into the log window.
*   Use the input box at the bottom of the panel to send commands to the running script.
*   Use the **Kill** button to terminate the process.
*   Press `n` to spawn a new panel or `q` to quit the entire application.
