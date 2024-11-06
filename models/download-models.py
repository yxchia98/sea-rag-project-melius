from huggingface_hub import snapshot_download

snapshot_download(repo_id="Qwen/Qwen2.5-1.5B-Instruct", repo_type="model", local_dir="./hf-models/Qwen2.5-1.5B-Instruct", token="")

snapshot_download(repo_id="intfloat/multilingual-e5-base", repo_type="model", local_dir="./hf-models/multilingual-e5-base", token="")

snapshot_download(repo_id="intfloat/multilingual-e5-small", repo_type="model", local_dir="./hf-models/multilingual-e5-small", token="")