# TinyStories-Small-Language-Model
# 🚀 Small Language Model (SLM) from Scratch

This project is my hands-on attempt at building a **GPT-style Transformer model** from scratch using **PyTorch**.  
The model is trained on the **TinyStories dataset** and is capable of generating short, coherent, and creative stories — all while being small enough to run on a single GPU or even Google Colab.  

Unlike most frameworks that hide the complexity, this project focuses on **clarity and transparency**, showing every moving part of a language model.

---

## ✨ Features
- **Custom Transformer Architecture**  
  - Token & position embeddings  
  - Multi-head causal self-attention  
  - GELU activation with MLP  
  - Residual connections & LayerNorm  
  - Shared embedding-output projection  

- **Lightweight yet capable**  
  - ~50M parameters (6 layers, 6 heads, hidden size 384, context length 128)  
  - Runs on a single GPU or Colab  

- **Training Pipeline**  
  - Dataset: [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)  
  - Tokenization with `tiktoken`  
  - Memory-mapped binary storage for efficient reads (`train.bin`, `validation.bin`)  
  - AdamW optimizer with learning rate warmup + cosine decay  
  - Mixed precision training (`torch.amp`) for speed and memory efficiency  

- **Story Generation**  
  - Generates fun, child-friendly short stories  
  - Demonstrates the core mechanics behind modern LLMs  

---

## 🛠 Tech Stack
- **Language**: Python  
- **Deep Learning Framework**: PyTorch  
- **Dataset**: Hugging Face TinyStories  
- **Tools**: tiktoken, datasets, numpy  
