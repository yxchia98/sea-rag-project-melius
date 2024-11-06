FROM python:3.12.5-slim-bullseye

RUN pip install llama-index-embeddings-ollama llama-index-llms-ollama llama-index-readers-web llama-index-vector-stores-chroma llama-index gradio

RUN pip install pysqlite3-binary

COPY demo/ /demo/

ENTRYPOINT [ "/bin/bash" ]
CMD ["-c", "cd /demo/ && python rag-demo-melius.py"]
