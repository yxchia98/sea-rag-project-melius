from huggingface_hub import snapshot_download

snapshot_download(repo_id="SeaLLMs/SeaLLMs-v3-1.5B", repo_type="model", local_dir="./models/hf-models/SeaLLMs-v3-1.5B")

snapshot_download(repo_id="SeaLLMs/SeaLLMs-v3-7B", repo_type="model", local_dir="./models/hf-models/SeaLLMs-v3-7B")

snapshot_download(repo_id="Qwen/Qwen2.5-3B-Instruct", repo_type="model", local_dir="./models/hf-models/Qwen2.5-3B-Instruct")

snapshot_download(repo_id="Qwen/Qwen2.5-1.5B-Instruct", repo_type="model", local_dir="./models/hf-models/Qwen2.5-1.5B-Instruct")

snapshot_download(repo_id="intfloat/multilingual-e5-large-instruct", repo_type="model", local_dir="./models/hf-models/multilingual-e5-large-instruct")


