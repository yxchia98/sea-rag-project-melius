# import
from llama_index.core import StorageContext, Settings, PromptTemplate, VectorStoreIndex, SimpleDirectoryReader, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.readers.web import SimpleWebPageReader
from llama_index.readers.web import BeautifulSoupWebReader

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
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
        - Generate only the requested output, don't include any other language before or after the requested output.
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

        # self.remote_db = chromadb.HttpClient(host=f"https://chromadb.cnasg.dellcsc.com:443")
        self.remote_db = chromadb.HttpClient(host=f"{os.environ['CHROMA_DB_URL']}:{os.environ['CHROMA_DB_PORT']}")

        # check and recreate collection if exists
        if "rag_data" in [c.name for c in self.remote_db.list_collections()]:
            print('collection rag_data exists, deleting...')
            self.remote_db.delete_collection(name="rag_data")
            print('deleted collection rag_data!')

        self.chroma_collection = self.remote_db.get_or_create_collection("rag_data")
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)


        self.documents = SimpleDirectoryReader("./data/").load_data()
        
        #Settings.chunk_size = 512
        #Settings.chunk_overlap = 50
        
        self.index = VectorStoreIndex.from_documents(self.documents, storage_context=self.storage_context, show_progress=True)
        
        self.query_engine = self.index.as_query_engine(streaming=True, similarity_top_k=2)
        #self.query_engine = self.index.as_query_engine(streaming=True, similarity_top_k=4)

        self.use_rag = False

        sample_text = "This is a sample text."
        embedding_vector = self.embed_model.get_text_embedding(sample_text)
        print(f"Vector dimension size: {len(embedding_vector)}")
        print(embedding_vector[:5])


    def inspect_chroma_db(self):
        collection = self.chroma_collection
        
        # Retrieve all items
        documents = collection.get(include=['embeddings', 'metadatas', 'documents'])

        # Assuming you have retrieved documents
        total_size = query_engine.calculate_size(documents)

        #print(f"Content of documents: {documents}")
        print(f"Documents size: {total_size}")
        
        #embedding = documents.get('embeddings')
        #embedding_size = len(str(embedding).encode('utf-8')) * 4  # Assuming 32-bit floats (4 bytes each)
        #print(f"Embedding: {embedding}")
        #print(f"Embedding size: {embedding_size}")

    def calculate_size(self, doc):
        total_size = 0

        # Calculate the size of the embedding vector
        embedding = doc.get('embeddings')
        if embedding is not None:
            embedding_size = len(embedding) * 4  # Assuming 32-bit floats (4 bytes each)
        else:
            embedding_size = 0

        # Calculate the size of the document text
        document_text = doc.get('documents')
        if document_text is not None:
            document_size = len(str(document_text).encode('utf-8'))
        else:
            document_size = 0

        # Calculate the size of the metadata
        metadata = doc.get('metadatas')
        if metadata is not None:
            metadata_size = len(str(metadata).encode('utf-8'))
        else:
            metadata_size = 0

        # Calculate total size for this document
        doc_total_size = embedding_size + document_size + metadata_size
        total_size += doc_total_size

        # Print sizes for debugging
        print(f"Embedding size: {embedding_size} bytes")
        print(f"Document text size: {document_size} bytes")
        print(f"Metadata size: {metadata_size} bytes")
        print(f"Total size for this document: {doc_total_size} bytes")
        print("\n")

        print(f"Total size of all documents: {total_size} bytes")
        return total_size


    def toggle_rag(self, toggle):
        # Assuming query_engine is an instance of Custom_Query_Engine
        query_engine.inspect_chroma_db()

        self.use_rag = toggle
        return self.use_rag

    def get_rag_toggle(self):
        return self.use_rag

    def reload(self, path):
        # Delete existing index and query engine
        del self.query_engine
        del self.index

        # Load documents
        self.documents = SimpleDirectoryReader(RAG_UPLOAD_FOLDER).load_data()
        
        total_text_size = 0
        total_vector_size = 0
        total_metadata_size = 0
        
        for doc in self.documents:
            # Text chunk size
            text_chunk_size = len(doc.text.encode('utf-8'))
            total_text_size += text_chunk_size
            
            # Vector size
            embedding_vector = self.embed_model.get_text_embedding(doc.text)
            vector_size = len(embedding_vector) * 4  # Assuming 32-bit floats (4 bytes each)
            total_vector_size += vector_size
            
            # Metadata size
            metadata = doc.metadata if hasattr(doc, 'metadata') else {}
            metadata_size = sum(len(str(key).encode('utf-8')) + len(str(value).encode('utf-8')) for key, value in metadata.items())
            total_metadata_size += metadata_size
            
            # Optional: Print the sizes for each document (for debugging or detailed analysis)
            print(f"Document: {doc.doc_id if hasattr(doc, 'doc_id') else 'Unknown ID'}")
            print(f"  Text chunk size: {text_chunk_size} bytes")
            print(f"  Vector size: {vector_size} bytes")
            print(f"  Metadata size: {metadata_size} bytes")
            print(f"  Total size: {text_chunk_size + vector_size + metadata_size} bytes")

        # Print the overall total sizes
        print(f"Total text chunk size: {total_text_size} bytes")
        print(f"Total vector size: {total_vector_size} bytes")
        print(f"Total metadata size: {total_metadata_size} bytes")
        print(f"Total size of all data sent to vector DB: {total_text_size + total_vector_size + total_metadata_size} bytes")
        
        # Create index and query engine
        self.index = VectorStoreIndex.from_documents(self.documents, storage_context=self.storage_context, show_progress=True)
        self.query_engine = self.index.as_query_engine(streaming=True, similarity_top_k=2)


    def reload_simple(self, path):
        # Delete existing index and query engine
        del self.query_engine
        del self.index

        # Load documents
        self.documents = SimpleDirectoryReader(RAG_UPLOAD_FOLDER).load_data()
        
        # Calculate total size of the document text
        total_text_size = sum(len(doc.text.encode('utf-8')) for doc in self.documents)
        print(f"Total size of document text: {total_text_size} bytes")

        # Calculate total size of vectors
        total_vector_size = 0
        for doc in self.documents:
            embedding_vector = self.embed_model.get_text_embedding(doc.text)
            vector_size = len(embedding_vector) * 4  # Assuming 32-bit floats (4 bytes each)
            total_vector_size += vector_size

        print(f"Total size of vectors sent to vector DB: {total_vector_size} bytes")
        
        # Create index and query engine
        self.index = VectorStoreIndex.from_documents(self.documents, storage_context=self.storage_context, show_progress=True)
        self.query_engine = self.index.as_query_engine(streaming=True, similarity_top_k=2)


    def reload_original(self, path):
        del self.query_engine
        del self.index
        self.documents = SimpleDirectoryReader(RAG_UPLOAD_FOLDER).load_data()
        self.index = VectorStoreIndex.from_documents(self.documents, storage_context=self.storage_context, show_progress=True)
        self.query_engine = self.index.as_query_engine(streaming=True, similarity_top_k=2)
        #self.query_engine = self.index.as_query_engine(streaming=True, similarity_top_k=4)

    def query_chromadb(self, message):
        start_time = time.time()

        retriever = self.index.as_retriever(similarity_top_k=2)
        retrieval_results = retriever.retrieve(message)

        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f"Response: {retrieval_results}")
        print(f"ChromaDB query time: {elapsed_time * 1000} milliseconds")

        return retrieval_results

    def query(self, message):
        return self.query_engine.query(message)

    def query_without_rag(self, message):
        # self.query_engine = self.llm.as_query_engine(streaming=True)
        return self.llm.stream_complete(message)
        # return self.query_engine.query(message)

    def completion_to_prompt(self, completion):
        print(f"<|im_start|>system\n{self.SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{completion}<|im_end|>\n<|im_start|>assistant\n")
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
    

         


# SYSTEM_PROMPT = """You are an AI assistant that answers questions in a friendly manner, based on the given source documents. Here are some rules you always follow:
# - Generate human readable output, avoid creating output with gibberish text.
# - Generate only the requested output, don't include any other language before or after the requested output.
# - Never say thank you, that you are happy to help, that you are an AI agent, etc. Just answer directly.
# - Generate professional language typically used in business documents in North America.
# - Never generate offensive or foul language.
# """


# load documents
# documents = SimpleDirectoryReader("./data/paul_graham/").load_data()
# documents = SimpleWebPageReader(html_to_text=True).load_data(
#     ["http://paulgraham.com/worked.html", "https://jujutsu-kaisen.fandom.com/wiki/Satoru_Gojo"]
# )
# documents = BeautifulSoupWebReader().load_data(
#     ["http://paulgraham.com/worked.html", "https://jujutsu-kaisen.fandom.com/wiki/Satoru_Gojo"]
# )


css = """
.app-interface {
    height:90vh;
}
.chat-interface {
    height: 85vh;
}
.file-interface {
    height: 40vh;
}
.web-interface {
    height: 30vh;
}
"""

query_engine = Custom_Query_Engine()

def stream_response(message, history):
    print(f"current RAG toggle is {query_engine.get_rag_toggle()}")
    if query_engine.get_rag_toggle():
        print('using RAG')

        # Now query only the ChromaDB to check time
        self.query_chromadb(message)

        start_time = time.time()
        print(f"Start time: {start_time}")
        print(f"Message: {message}")
        response = query_engine.query(message)

        end_time = time.time()
        print(f"End time: {end_time}")
        elapsed_time = (end_time - start_time) * 1000
        print(f"Elapsed time: {elapsed_time} milliseconds")

        context = " ".join([node.dict()['node']['text'] for node in response.source_nodes])
        #print(context)
        
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

    for doc in query_engine.documents:
        print(f"Document length: {len(doc.text)}")
    
    return file_paths

def toggle_knowledge_base(use_rag):
    print(f"toggling use knowledge base to {use_rag}")
    query_engine.toggle_rag(use_rag)
    return use_rag


with gr.Blocks(css=css) as demo:
    # gr.Markdown(
    # """
    # # **Retrieval Augmented Generation with GPU**
    # """)
    gr.Markdown(
    """
    <h1 style="text-align: center;">Retrieval Augmented Generation with GPU 💻 📑 ✨</h3>
    """)
    with gr.Row(equal_height=True, elem_classes=["app-interface"]):
        with gr.Column(scale=4, elem_classes=["chat-interface"]):
            test = gr.ChatInterface(fn=stream_response)
        with gr.Column(scale=1):
            file_input = gr.File(elem_classes=["file-interface"], file_types=["pdf", "csv", "text", "html"], file_count="multiple")
            # upload_button = gr.UploadButton("Click to Upload a File", file_types=["image", "video", "pdf", "csv", "text"], file_count="multiple")
            # upload_button.upload(upload_file, upload_button, file_input)
            vectorize_button = gr.Button("Vectorize Files")
            vectorize_button.click(fn=vectorize, inputs=file_input, outputs=file_input)
            use_rag = gr.Checkbox(label="Use Knowledge Base")
            use_rag.select(fn=toggle_knowledge_base, inputs=use_rag)
            

demo.launch(server_name="0.0.0.0", ssl_verify=False)



# what difference does dell technologies make in on-premise inferencing?


