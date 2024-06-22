
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

class MyCustomHandler(BaseCallbackHandler):
    def __init__(self, st_placeholder):
        super().__init__()
        self.st_placeholder = st_placeholder
        self.dialogue = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if token.strip():  # Ignore empty tokens
            self.dialogue += token + " "
            self.st_placeholder.text(self.dialogue)  # Update the Streamlit placeholder

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
    db = FAISS.from_documents(texts, embeddings)
    
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
    db = FAISS.deserialize_from_bytes(embeddings=embeddings, serialized=pkl)
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

def stream_output(question, st_placeholder):
    callback_handler = MyCustomHandler(st_placeholder)
    llm.callbacks = [callback_handler]
    callback_handler.clear_dialogue()
    llm.predict(question)

# Main Streamlit app logic
def main():
    st.title('Live Diagnosis App')
    st.write('Ask your medical questions here and get live answers from our AI assistant.')

    # Text input field for user to ask questions
    question = st.text_input('Enter your question:')
    placeholder = st.empty()
    
    if st.button('Ask'):
        stream_output(question, placeholder)

# Start the Streamlit app
if __name__ == '__main__':
    main()
