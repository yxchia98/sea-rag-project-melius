from huggingface_hub import snapshot_download


snapshot_download(repo_id="intfloat/multilingual-e5-large-instruct", repo_type="model", local_dir="./hf-models/multilingual-e5-large-instruct", token="")