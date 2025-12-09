# Memory-Efficient-Edge-Chatbot-on-Raspberry-Pi-
A small LLM chatbot running fully on a Raspberry Pi, with memory-aware context management and benchmarking.

This repository implements a Transformer-based chatbot optimized for low-memory inference on a Raspberry Pi.  
This project follows the requirements of the Embedded Systems for AI course.

This project demonstrates how to deploy a small LLM (e.g., DistilGPT-2) on a memory-constrained edge device such as the Raspberry Pi.  
To support long-form interaction on limited hardware, several optimized KV-cache strategies are implemented:

- **Sliding-Window Cache** — retains only the last *N* tokens  
- **Paged / Ring Cache** — retains the last *K* dialogue turns  
- **Quantized KV-Cache (INT8)** — compresses K/V tensors to reduce RAM growth  

A custom inference backend enables token-by-token generation, cache injection, and detailed performance measurement.


## Getting Started
```bash
git clone https://github.com/sudmansakib/Memory-Efficient-Edge-Chatbot-on-Raspberry-Pi.git
cd Memory-Efficient-Edge-Chatbot-on-Raspberry-Pi

