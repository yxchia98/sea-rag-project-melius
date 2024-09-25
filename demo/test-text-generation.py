from llama_index.llms.ollama import Ollama
llm = Ollama(
    model="yxchia/llama3-8b-cpt-sea-lionv2-instruct:Q4_K_M", 
    base_url="http://localhost:11434",
    request_timeout=120.0)
response = llm.stream_complete("What is hello in thai?")
for r in response:
    print(r.delta, end="")