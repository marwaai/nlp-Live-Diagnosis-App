import threading
import streamlit as st
import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain_core.callbacks import BaseCallbackHandler
from queue import Queue

# Custom callback handler to update Streamlit placeholder
class MyCustomHandler(BaseCallbackHandler):
    def __init__(self, st_queue):
        super().__init__()
        self.st_queue = st_queue
        self.dialogue = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if token.strip():  # Ignore empty tokens
            self.dialogue += token + " "
            self.st_queue.put(self.dialogue)  # Put the dialogue into the queue

    def clear_dialogue(self):
        self.dialogue = ""

# Initialize persist directory
persist_directory = "db"

# Function to load documents and create embeddings
def load_documents_and_create_embeddings():
    texts = []
    for root, dirs, files in os.walk(persist_directory):
        for file in files:
            if file.endswith("txt"):
                loader = TextLoader(os.path.join(root, file))
                documents = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
                texts.extend(text_splitter.split_documents(documents))

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # Use HuggingFace embeddings
    db = FAISS.from_documents(texts)
    
    # Serialize FAISS index to bytes
    pkl = db.serialize_to_bytes()
    with open(os.path.join(persist_directory, "faiss_index.pkl"), "wb") as f:
        f.write(pkl)
    
    return db

# Load or create FAISS index
faiss_index_path = os.path.join(persist_directory, "faiss_index.pkl")
if os.path.exists(faiss_index_path):
    with open(faiss_index_path, "rb") as f:
        pkl = f.read()
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # Use HuggingFace embeddings
    db = FAISS.deserialize_from_bytes(serialized=pkl)
else:
    db = load_documents_and_create_embeddings()

# Initialize LLM and QA models
llm = CTransformers(
    model="TheBloke/Llama-2-7B-Chat-GGML",
    model_file='llama-2-7b-chat.ggmlv3.q2_K.bin', stream=True,
    prompt="You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
)

retriever = db.as_retriever(search_kwargs={"k": 1})

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="refine",
    retriever=retriever,
    return_source_documents=False
)

# Function to handle UI updates from the queue
def handle_ui_updates(st_placeholder, st_queue):
    while True:
        dialogue = st_queue.get()  # Get dialogue from the queue
        st_placeholder.markdown(dialogue)  # Update Streamlit placeholder with markdown

# Function to stream output to Streamlit placeholder
def stream_output(question, st_placeholder, st_queue):
    callback_handler = MyCustomHandler(st_queue)
    llm.callbacks = [callback_handler]
    callback_handler.clear_dialogue()
    threading.Thread(target=llm.predict, args=(question,)).start()

# Main Streamlit app logic
def main():
    st.title('Live Diagnosis App')
    st.write('Ask your medical questions here and get live answers from our AI assistant.')

    # Text input field for user to ask questions
    question = st.text_input('Enter your question:')
    placeholder = st.empty()

    # Create a queue for UI updates
    st_queue = Queue()
    # Start a thread to handle UI updates
    threading.Thread(target=handle_ui_updates, args=(placeholder, st_queue), daemon=True).start()

    if st.button('Ask'):
        stream_output(question, placeholder, st_queue)

# Start the Streamlit app
if __name__ == '__main__':
    main()
