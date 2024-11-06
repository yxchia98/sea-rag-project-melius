#!/bin/sh

ollama serve & 
sleep 5

cd /root/models/Qwen2.5-1.5B-Instruct-gguf
ollama create llm-model -f Modelfile
cd /root/models/multilingual-e5-base-gguf
ollama create embedding-model -f Modelfile

cd /root/
rm -rf models/
 
