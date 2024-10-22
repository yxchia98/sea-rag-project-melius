FROM ollama/ollama:latest
COPY ./ /root/models/
# RUN ollama serve & sleep 5 && cd /root/models/llama3-8b-cpt-sea-lionv2-instruct-gguf && ollama create llama3-8b-cpt-sea-lionv2-instruct -f Modelfile && cd /root/models/paraphrase-multilingual-MiniLM-L12-v2-gguf && ollama create paraphrase-multilingual-MiniLM-L12-v2 -f Modelfile && cd /root/ && rm -rf models/ && 
RUN chmod +x /root/models/build-ollama.sh
RUN /root/models/build-ollama.sh
EXPOSE 11434
ENTRYPOINT [ "/bin/bash" ]
CMD ["-c", "ollama serve & sleep 5 && ollama run llama3-8b-cpt-sea-lionv2.1-instruct"]