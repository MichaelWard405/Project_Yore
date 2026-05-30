# Project YoRe


The project is split into two primary components:
1.  **`Cato_Neural_Net.py`**: The core chatbot engine, featuring a custom transformer model, training loops, and a dual interface.
2.  **`Data_Collector.py`**: A versatile Multi-Media data acquisition utility designed to scrape websites and transcribe audio/video content to generate training data.

## Features

*   **Custom Transformer Model**: A from-scratch chatbot model built with PyTorch, including standard components like multi-head self-attention and feed-forward blocks.
*   **LoRA Integration**: Utilizes Low-Rank Adaptation for efficient fine-tuning. This allows for rapid specialization and "personality" swapping without retraining the entire model.
*   **Two-Stage Training**: Implements a pre-training stage on general text data and a fine-tuning stage using LoRA on specialized, instruction-based data.
*   **Dual Interface**:
    *   **Interactive CLI**: A terminal-based chat interface for direct interaction and administration.
    *   **FastAPI REST API**: Exposes a `/chat` endpoint, allowing external applications (e.g., game engines, scripts, web frontends) to communicate with the model.
*   **Runtime Evolution**: An admin function (`Remember`) allows the model to learn from the live conversation history by merging it with the existing LoRA dataset and retraining the LoRA weights.
*   **Versatile Data Collector**: A powerful script to build datasets from:
    *   **Media VODs**: Transcribes YouTube and Twitch videos using `yt-dlp` and `faster-whisper`.
    *   **Live Streams**: Connects to live YouTube/Twitch feeds, chunks the audio, and performs real-time transcription.
    *   **Websites**: A resumable, polite crawler for scraping text from documentation sites and other web pages.
*   **GPU Acceleration**: Automatically utilizes CUDA for both model training (`PyTorch`) and audio transcription (`faster-whisper`) if a compatible GPU is available, with a fallback to CPU.

### Planned Updates

## Core Components

### Cato Neural Net (`Cato_Neural_Net.py`)

This is the heart of the project. It defines, trains, and serves the chatbot model.

*   **Operation**: On first run, it builds a BPE tokenizer from your data, pre-trains the base model, and then fine-tunes the LoRA matrices. Subsequent runs can load the trained weights directly.
*   **Admin Menu**: While the chat is active, you can type `~` to access a system menu with the following options:
    *   **`Remember`**: Merges the `live_history.txt` with the core LoRA data and re-runs the fine-tuning process to incorporate new interaction patterns.
    *   **`Swap`**: Switches the active "personality" by loading different LoRA weights from a `.pth` file.
*   **API**: The script starts a Uvicorn server hosting a FastAPI application on `http://127.0.0.1:8000`. This allows for programmatic interaction with the chatbot.

### Data Collector (`Data_Collector.py`)

This script is your toolkit for creating the `pretrain_data.txt` and `lora_data.txt` files needed to train the model. It provides a menu-driven interface to:

1.  **Scrape Media VOD**: Provide a YouTube or Twitch video URL to download and transcribe its audio content.
2.  **Chunk Live Stream**: Provide a live stream URL to capture and transcribe the content in real-time.
3.  **Crawl Doc Website**: Provide a base URL to recursively scrape text content from a website. The crawler is polite (configurable delay) and can resume a previous session.
4.  **Adjust Settings**: Configure parameters like the `faster-whisper` model size (trading speed for accuracy), crawl depth, and audio chunk duration.

## Getting Started

### Prerequisites

You need Python 3 installed. You can install the required libraries using pip:

```bash
pip install torch tokenizers fastapi uvicorn yt-dlp faster-whisper requests beautifulsoup4 numpy
```
For GPU acceleration, ensure you have a CUDA-compatible GPU and install the appropriate version of PyTorch by following the instructions on the [official PyTorch website](https://pytorch.org/get-started/locally/).

### Step 1: Collect Data

First, run the data 

```bash
python Data_Collector.py
```
