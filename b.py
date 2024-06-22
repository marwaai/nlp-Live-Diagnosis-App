import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.callbacks import BaseCallbackHandler
from langchain.document_loaders import TextLoader

class MyCustomHandler(BaseCallbackHandler):
    def __init__(self, st_placeholder):
        self.st_placeholder = st_placeholder
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token + " "
        self.st_placeholder.text(self.text)

# Initialize persist directory
persist_directory = "db"
index_path = os.path.join(persist_directory, "faiss_index")

# Load documents and create embeddings
@st.cache_resource(experimental_allow_widgets=True)
def load_texts():
    texts = []
    for root, dirs, files in os.walk(r"db"):
        for file in files:
            if file.endswith("txt"):
                loader = TextLoader(os.path.join(root, file))
                documents = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
                texts.extend(text_splitter.split_documents(documents))
    return texts

texts = load_texts()

# Create embeddings
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Load or create vector store
if os.path.exists(index_path):
    db = FAISS.load_local(index_path, embeddings)
    print("FAISS index loaded from disk.")
else:
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(index_path)
    print("FAISS index created and saved to disk.")

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

# Main Streamlit app logic
def main():
    st.title('Live Diagnosis App')
    st.write('Ask your medical questions here and get live answers from our AI assistant.')

    # Text input field for user to ask questions
    question = st.text_input('Enter your question:')
    placeholder = st.empty()
    if st.button('Ask'):
        stream_output(question, placeholder)

# Function to stream output
def stream_output(question, st_placeholder):
    callback_handler = MyCustomHandler(st_placeholder)
    llm.callbacks = [callback_handler]
    llm.predict(question)

# Start the Streamlit app
if __name__ == '__main__':
    main()
