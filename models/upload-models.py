from huggingface_hub import HfApi
api = HfApi()

api.create_repo("yixuan-chia/multilingual-e5-base-gguf", token="")

api.upload_folder(
    folder_path="./gguf-models/multilingual-e5-base-gguf",
    repo_id="yixuan-chia/multilingual-e5-base-gguf",
    repo_type="model",
	token=""
)