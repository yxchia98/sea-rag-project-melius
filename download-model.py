from huggingface_hub import snapshot_download

snapshot_download(repo_id="SeaLLMs/SeaLLMs-v3-1.5B-Chat", repo_type="model", local_dir="./llm-models/hf-models/SeaLLMs-v3-1.5B-Chat")

snapshot_download(repo_id="jinaai/jina-embeddings-v3", repo_type="model", local_dir="./llm-models/hf-models/jina-embeddings-v3")


# snapshot_download(repo_id="yahma/alpaca-cleaned", repo_type="dataset", local_dir=".")