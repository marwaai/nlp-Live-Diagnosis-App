import streamlit as st
import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain_core.callbacks import BaseCallbackHandler

class MyCustomHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        st.write(token)

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

    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(texts, embeddings)
    db.save("faiss_index")
    return db

# Load or create FAISS index
if os.path.exists("faiss_index"):
    db = FAISS.load("faiss_index")
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

def stream_output(question):
    callback_handler = MyCustomHandler(st.empty())
    llm.callbacks = [callback_handler]
    llm.predict(question)

# Main Streamlit app logic
def main():
    st.title('Live Diagnosis App')
    st.write('Ask your medical questions here and get live answers from our AI assistant.')

    # Text input field for user to ask questions
    question = st.text_input('Enter your question:')
    if st.button('Ask'):
        stream_output(question)

# Start the Streamlit app
if __name__ == '__main__':
    main()
