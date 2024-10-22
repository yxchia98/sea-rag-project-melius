from huggingface_hub import snapshot_download

snapshot_download(repo_id="SeaLLMs/SeaLLMs-v3-1.5B", repo_type="model", local_dir="./hf-models/SeaLLMs-v3-1.5B", token="")

snapshot_download(repo_id="aisingapore/llama3-8b-cpt-sea-lionv2.1-instruct-gguf", repo_type="model", local_dir="./hf-models/llama3-8b-cpt-sea-lionv2.1-instruct", token="")

snapshot_download(repo_id="intfloat/multilingual-e5-large-instruct", repo_type="model", local_dir="./hf-models/multilingual-e5-large-instruct", token="")

snapshot_download(repo_id="intfloat/multilingual-e5-large", repo_type="model", local_dir="./hf-models/multilingual-e5-large", token="")

snapshot_download(repo_id="intfloat/multilingual-e5-small", repo_type="model", local_dir="./hf-models/multilingual-e5-small", token="")