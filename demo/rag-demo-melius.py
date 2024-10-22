# import
from llama_index.core import StorageContext, Settings, PromptTemplate, VectorStoreIndex, SimpleDirectoryReader, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.readers.web import SimpleWebPageReader
from llama_index.readers.web import BeautifulSoupWebReader
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
import time
import gradio as gr
from tqdm import tqdm
import shutil
import os
from pathlib import Path
import glob


os.environ['GRADIO_TEMP_DIR'] = "/tmp/gradio"

RAG_UPLOAD_FOLDER = "/demo/rag-documents/"


class Custom_Query_Engine():
    def __init__(self):
        self.SYSTEM_PROMPT = """You are an AI assistant that answers questions in a friendly manner. Here are some rules you always follow:
        - Generate human readable output, avoid creating output with gibberish text.
        - Don't include any other language before or after the requested output.
        - Make use of the additional context given to provide better answers.
        """
        
        self.llm = Ollama(
            model=f"{os.environ['LLM_MODEL_NAME']}", 
            base_url=f"{os.environ['LLM_MODEL_ENDPOINT']}",
            request_timeout=120.0
            )

        # self.embed_model = IpexLLMEmbedding(model_name="/llm-models/hf-models/bge-small-en-v1.5", trust_remote_code=True)
        self.embed_model = OllamaEmbedding(
            model_name=f"{os.environ['EMBED_MODEL_NAME']}",
            base_url=f"{os.environ['EMBED_MODEL_ENDPOINT']}",
            ollama_additional_kwargs={"mirostat": 0},
            )
            
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model

        self.documents = SimpleDirectoryReader("./data/").load_data()
        self.index = VectorStoreIndex.from_documents(self.documents, show_progress=True)
        self.query_engine = self.index.as_query_engine(streaming=True, similarity_top_k=3)
        self.use_rag = False

    def toggle_rag(self, toggle):
        self.use_rag = toggle
        return self.use_rag

    def get_rag_toggle(self):
        return self.use_rag

    def reload(self, path, other_docs):
        del self.query_engine
        del self.index
        self.documents = SimpleDirectoryReader(RAG_UPLOAD_FOLDER).load_data()
        self.documents += other_docs
        self.index = VectorStoreIndex.from_documents(self.documents, show_progress=True)
        self.query_engine = self.index.as_query_engine(streaming=True, similarity_top_k=3)

    def query(self, message):
        return self.query_engine.query(message)

    def query_without_rag(self, message):
        # self.query_engine = self.llm.as_query_engine(streaming=True)
        return self.llm.stream_complete(message)
        # return self.query_engine.query(message)

    def completion_to_prompt(self, completion):
        # print(f"<|im_start|>system\n{self.SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{completion}<|im_end|>\n<|im_start|>assistant\n")
        return f"<|im_start|>system\n{self.SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{completion}<|im_end|>\n<|im_start|>assistant\n"

    def messages_to_prompt(self, messages):
        prompt = ""
        for message in messages:
            if message.role == "system":
                prompt += f"<|im_start|>system\n{message.content}<|im_end|>\n"
            elif message.role == "user":
                prompt += f"<|im_start|>user\n{message.content}<|im_end|>\n"
            elif message.role == "assistant":
                prompt += f"<|im_start|>assistant\n{message.content}<|im_end|>\n"

        if not prompt.startswith("<|im_start|>system"):
            prompt = "<|im_start|>system\n" + prompt

        prompt = prompt + "<|im_start|>assistant\n"

        # print(prompt)

        return prompt


css = """
.app-interface {
    height:80vh;
}
.chat-interface {
    height: 75vh;
}
.file-interface {
    height: 40vh;
}
"""

query_engine = Custom_Query_Engine()

def stream_response(message, history):
    print(f"current RAG toggle is {query_engine.get_rag_toggle()}")
    if query_engine.get_rag_toggle():
        print('using RAG')
        response = query_engine.query(message)
        print(response.source_nodes[0].get_content())
        res = ""
        for token in response.response_gen:
            # print(token, end="")
            res = str(res) + str(token)
            yield res
    else:
        print('not using RAG')
        response = query_engine.query_without_rag(message)
        res = ""
        for token in response:
            # print(token, end="")
            res = str(res) + str(token.delta)
            yield res

def vectorize(files, progress=gr.Progress()):
    Path(RAG_UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)
    UPLOAD_FOLDER = RAG_UPLOAD_FOLDER

    prev_files = glob.glob(f"{UPLOAD_FOLDER}*")
    for f in prev_files:
        os.remove(f)

    if not files:
        return []
    
    file_paths = [file.name for file in files]
    for file in files:
        shutil.copy(file.name, UPLOAD_FOLDER)

    query_engine.reload(UPLOAD_FOLDER)
    
    return file_paths

def toggle_knowledge_base(use_rag):
    print(f"toggling use knowledge base to {use_rag}")
    query_engine.toggle_rag(use_rag)
    return use_rag


with gr.Blocks(css=css) as demo:
    gr.Markdown(
    """
    <h1 style="text-align: center;">Project Melius Document Chatbot 💻📑✨</h3>
    """)
    with gr.Row(equal_height=False, elem_classes=["app-interface"]):
        with gr.Column(scale=4, elem_classes=["chat-interface"]):
            test = gr.ChatInterface(fn=stream_response)
        with gr.Column(scale=1):
            # url_input = gr.Textbox(label="Reference File URL", lines="2")
            file_input = gr.File(elem_classes=["file-interface"], file_types=["pdf", "csv", "text", "html"], file_count="multiple")
            # upload_button = gr.UploadButton("Click to Upload a File", file_types=["image", "video", "pdf", "csv", "text"], file_count="multiple")
            # upload_button.upload(upload_file, upload_button, file_input)
            vectorize_button = gr.Button("Vectorize Files")
            vectorize_button.click(fn=vectorize, inputs=file_input, outputs=file_input)
            use_rag = gr.Checkbox(label="Use Knowledge Base")
            use_rag.select(fn=toggle_knowledge_base, inputs=use_rag)
            

demo.launch(server_name="0.0.0.0", ssl_verify=False)



# what difference does dell technologies make in on-premise inferencing?


