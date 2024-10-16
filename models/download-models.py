from huggingface_hub import snapshot_download


snapshot_download(repo_id="aisingapore/llama3-8b-cpt-sea-lionv2.1-instruct-gguf", repo_type="model", local_dir="./hf-models/llama3-8b-cpt-sea-lionv2.1-instruct", token="")