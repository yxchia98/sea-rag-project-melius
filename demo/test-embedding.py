from llama_index.embeddings.ollama import OllamaEmbedding

ollama_embedding = OllamaEmbedding(
    model_name="paraphrase-multilingual",
    base_url="http://localhost:11434",
    ollama_additional_kwargs={"mirostat": 0},
)

pass_embedding = ollama_embedding.get_text_embedding_batch(
    ["This is a passage!", "This is another passage"], show_progress=True
)
# print(pass_embedding)

query_embedding1 = ollama_embedding.get_query_embedding("Where is blue?")
query_embedding2 = ollama_embedding.get_query_embedding("Where is blue?")
print(query_embedding1)
print(query_embedding1 == query_embedding2)