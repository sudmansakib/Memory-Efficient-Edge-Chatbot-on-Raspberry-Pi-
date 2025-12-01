# Memory-Efficient-Edge-Chatbot-on-Raspberry-Pi-
A small LLM chatbot running fully on a Raspberry Pi, with memory-aware context management and benchmarking.

This repository implements a Transformer-based chatbot optimized for low-memory inference on a Raspberry Pi.  
This project follows the requirements of the Embedded Systems for AI course.

## Repository Status
Currently under development.  
Core folder structure is ready; code will be added progressively.

## Project Goals
- Deploy a small Transformer model (TinyLlama / DistilGPT2 / Phi-1.5 INT8)
- Implement memory-efficient KV-cache strategies:
  - Sliding Window
  - Paged / Ring Cache
  - Quantized Cache (INT8 K/V)
- Build CLI or Web chatbot UI
- Benchmark latency, tokens/sec, RAM usage, and performance

## Getting Started
```bash
git clone https://github.com/sudmansakib/Memory-Efficient-Edge-Chatbot-on-Raspberry-Pi.git
cd Memory-Efficient-Edge-Chatbot-on-Raspberry-Pi

